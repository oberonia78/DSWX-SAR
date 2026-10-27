"""Optional GUNW coherence input, grid alignment, and water evidence.

All HDF5 reads use the same local/S3 reader as GCOV. Raster operations are
windowed; neither a full GUNW nor the full GCOV mosaic is held in memory.
"""

from contextlib import ExitStack
import copy
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from dswx_sar.common.read_h5_s3 import open_h5

logger = logging.getLogger('dswx_sar')
IDENTIFICATION = '/science/LSAR/identification'
GRIDS = '/science/LSAR/GUNW/grids'
RAW_SUPPORT_METHODS = ('raw_support', 'raw_support_pooled')


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _time(value):
    # datetime supports NISAR nanosecond strings (truncated to microseconds).
    return datetime.fromisoformat(_text(value).replace('Z', '+00:00')).replace(
        tzinfo=None)


def _identity(src, product):
    ident = src[IDENTIFICATION]
    result = {key: _text(ident[key][()]) for key in
              ('trackNumber', 'frameNumber', 'orbitPassDirection')}
    result['trackNumber'] = int(result['trackNumber'])
    result['frameNumber'] = int(result['frameNumber'])
    result['orbitPassDirection'] = result['orbitPassDirection'].upper()
    prefixes = ('',) if product == 'GCOV' else ('reference', 'secondary')
    result['intervals'] = []
    for prefix in prefixes:
        start = prefix + 'ZeroDopplerStartTime' if prefix else 'zeroDopplerStartTime'
        end = prefix + 'ZeroDopplerEndTime' if prefix else 'zeroDopplerEndTime'
        result['intervals'].append((_time(ident[start][()]),
                                    _time(ident[end][()])))
    return result


def _matches(gcov, gunw):
    same_frame = all(gcov[key] == gunw[key] for key in
                     ('trackNumber', 'frameNumber', 'orbitPassDirection'))
    start, end = gcov['intervals'][0]
    return same_frame and any(start <= other_end and other_start <= end
                              for other_start, other_end in gunw['intervals'])


def split_inputs(paths):
    """Classify by HDF5 productType and reject every unmatched GUNW.

    GCOV coverage may be partial: each supplied GUNW must match at least one
    GCOV, but GCOV frames without a GUNW keep the original algorithm.
    """
    products = {'GCOV': [], 'GUNW': []}
    identities = {}
    for path in sorted(set(paths)):
        with open_h5(path) as src:
            if f'{IDENTIFICATION}/productType' not in src:
                raise ValueError(f'Missing NISAR identification/productType: {path}')
            product = _text(src[f'{IDENTIFICATION}/productType'][()])
            if product not in products:
                raise ValueError(f'Unsupported input product {product}: {path}')
            if f'/science/LSAR/{product}' not in src:
                raise ValueError(f'Missing {product} data group: {path}')
            products[product].append(path)
            # Preserve the GCOV-only path: extra InSAR matching metadata is
            # required only when GUNW is actually supplied.
    if not products['GCOV']:
        raise ValueError('At least one GCOV input is required.')
    if products['GUNW']:
        for product, files in products.items():
            for path in files:
                with open_h5(path) as src:
                    try:
                        identities[path] = _identity(src, product)
                    except (KeyError, ValueError) as error:
                        raise ValueError(f'Invalid {product} matching metadata: {path}: {error}') from error
    matches = {}
    for path in products['GUNW']:
        matches[path] = [gcov for gcov in products['GCOV']
                         if _matches(identities[gcov], identities[path])]
        if not matches[path]:
            raise ValueError(
                f'GUNW has no matching GCOV track/frame/direction and '
                f'reference or secondary acquisition: {path}')
    return products['GCOV'], products['GUNW'], matches


def enabled(cfg, stage):
    options = getattr(cfg.groups.processing, 'gunw_coherence', None)
    return bool(options and getattr(options, stage).enabled and
                getattr(cfg.groups.input_file_group, 'gunw_file_path', []))


def raster_path(cfg):
    return os.path.join(cfg.groups.product_path_group.scratch_path,
                        'mosaic_gunw_coherence.tif')


def raw_raster_path(cfg):
    return os.path.join(cfg.groups.product_path_group.scratch_path,
                        'mosaic_gunw_coherence_raw.tif')


def landcover_mask_options(options):
    """Validate the optional raw-coherence neighborhood rejection method."""
    result = {'method': getattr(options, 'method', 'threshold')}
    if result['method'] not in ('threshold', *RAW_SUPPORT_METHODS):
        raise ValueError('landcover_mask.method must be threshold, raw_support, or raw_support_pooled.')
    if not np.isfinite(options.threshold) or not 0 <= options.threshold <= 1:
        raise ValueError('landcover_mask.threshold must be within [0,1].')
    if result['method'] in RAW_SUPPORT_METHODS:
        result.update(window_size=getattr(options, 'window_size', 3),
                      support_fraction=getattr(options, 'support_fraction', .7),
                      min_valid_fraction=getattr(options, 'min_valid_fraction', .5),
                      save_diagnostics=getattr(options, 'save_diagnostics', True))
        size = result['window_size']
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 3 or size % 2 != 1:
            raise ValueError('landcover_mask.window_size must be an odd integer >= 3.')
        for key in ('support_fraction', 'min_valid_fraction'):
            if not np.isfinite(result[key]) or not 0 < result[key] <= 1:
                raise ValueError(f'landcover_mask.{key} must be within (0,1].')
    return result


def _needs_raw_support(cfg):
    return (enabled(cfg, 'landcover_mask') and landcover_mask_options(
        cfg.groups.processing.gunw_coherence.landcover_mask)['method'] in RAW_SUPPORT_METHODS)


def _status_path(cfg):
    return os.path.join(cfg.groups.product_path_group.scratch_path,
                        'gunw_coherence_usage.json')


def _mosaic_options(cfg):
    options = cfg.groups.processing.gunw_coherence
    result = {key: getattr(options, key) for key in
              ('source', 'frequency', 'polarization', 'resampling', 'mosaic_mode')}
    result['filter'] = coherence_filter_options(options)
    result['raw_support'] = _needs_raw_support(cfg)
    return result


def coherence_filter_options(options):
    """Normalize only settings relevant to the selected coherence filter."""
    config = getattr(options, 'filter', None)
    if config is None or not getattr(config, 'enabled', False):
        return {'enabled': False}
    result = {
        'enabled': True,
        'method': getattr(config, 'method', 'bregman'),
        'block_size': getattr(config, 'block_size', 1024),
        'block_pad': getattr(config, 'block_pad', 32),
    }
    if result['method'] not in ('bregman', 'magnitude_mean', 'complex_mean'):
        raise ValueError('Unknown gunw_coherence.filter.method: ' + result['method'])
    if result['method'] == 'bregman':
        bregman = getattr(config, 'bregman', None)
        result.update(lambda_value=getattr(bregman, 'lambda_value', 10.0),
                      max_num_iter=getattr(bregman, 'max_num_iter', 250),
                      eps=getattr(bregman, 'eps', 1e-4))
    else:
        result.update(window_size=getattr(config, 'window_size', 3),
                      min_valid_fraction=getattr(config, 'min_valid_fraction', .5))
        window = result['window_size']
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 3 or window % 2 != 1:
            raise ValueError('Coherence mean filter window_size must be an odd integer >= 3.')
        fraction = result['min_valid_fraction']
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('Coherence min_valid_fraction must be within (0,1].')
    for key in ('lambda_value', 'eps') if result['method'] == 'bregman' else ():
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'gunw_coherence.filter.{key} must be finite and > 0.')
    keys = ('block_size', 'block_pad', 'max_num_iter') if result['method'] == 'bregman' else ('block_size', 'block_pad')
    for key in keys:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'gunw_coherence.filter.{key} must be a positive integer.')
    if result['method'] != 'bregman' and result['block_pad'] < result['window_size'] // 2:
        raise ValueError('Coherence block_pad must cover at least window_size // 2 pixels.')
    return result


def mean_coherence(values, window_size=3, min_valid_fraction=.5, interferogram=None,
                   return_applied=False):
    """Masked scalar mean, or magnitude of mean(gamma * exp(i*phase)).

    Complex averaging is a phase-sensitive score, not an unbiased coherence
    re-estimate. Invalid center phase or insufficient support retains raw
    coherence; the original coherence valid footprint is always preserved.
    """
    from scipy.ndimage import uniform_filter

    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 3 or window_size % 2 != 1:
        raise ValueError('window_size must be an odd integer >= 3.')
    if not np.isfinite(min_valid_fraction) or not 0 < min_valid_fraction <= 1:
        raise ValueError('min_valid_fraction must be within (0,1].')
    original = np.array(values, dtype=np.float32, copy=True)
    if original.ndim != 2:
        raise ValueError('Coherence filtering requires a two-dimensional array.')
    valid = np.isfinite(original) & (original >= 0) & (original <= 1)
    original[~valid] = np.nan
    samples = valid.copy()
    if interferogram is not None:
        if not np.iscomplexobj(interferogram) or interferogram.shape != original.shape:
            raise ValueError('Interferogram must be complex and on the coherence grid.')
        samples &= np.isfinite(interferogram) & (np.abs(interferogram) > 0)
    denominator = uniform_filter(samples.astype(np.float64), size=window_size,
                                 mode='constant', cval=0)
    apply = samples & (denominator + 1e-12 >= min_valid_fraction)
    if interferogram is None:
        total = uniform_filter(np.where(samples, original, 0).astype(np.float64),
                               size=window_size, mode='constant', cval=0)
    else:
        z = np.zeros(original.shape, dtype=np.complex128)
        z[samples] = original[samples] * np.exp(1j * np.angle(interferogram[samples]))
        real = uniform_filter(z.real, size=window_size, mode='constant', cval=0)
        imag = uniform_filter(z.imag, size=window_size, mode='constant', cval=0)
        total = np.hypot(real, imag)
    result = original.copy()
    result[apply] = np.clip(total[apply] / denominator[apply], 0, 1)
    return (result, apply) if return_applied else result


def filter_coherence(values, lambda_value=10.0, max_num_iter=250, eps=1e-4):
    """Bregman TV directly on [0,1] coherence; smaller lambda smooths more.

    Unlike the intensity filter, zero is valid and no logarithm is taken.
    Missing samples are temporarily nearest-valid extended, then restored.
    This is an extension approximation, not a masked TV objective.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.restoration import denoise_tv_bregman

    for name, value in (('lambda_value', lambda_value), ('eps', eps)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and > 0.')
    if isinstance(max_num_iter, bool) or not isinstance(max_num_iter, (int, np.integer)) or max_num_iter < 1:
        raise ValueError('max_num_iter must be a positive integer.')
    original = np.array(values, dtype=np.float32, copy=True)
    if original.ndim != 2:
        raise ValueError('Coherence filtering requires a two-dimensional array.')
    valid = np.isfinite(original) & (original >= 0) & (original <= 1)
    original[~valid] = np.nan
    if not valid.any():
        return original
    if valid.all():
        extended = original.copy()
    else:
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        extended = original[tuple(nearest)]
    # Protect the numerical solver against tiny edge tiles/native rasters.
    height, width = original.shape
    if min(height, width) < 3:
        extended = np.pad(extended, ((0, max(0, 3-height)), (0, max(0, 3-width))), mode='edge')
    filtered = denoise_tv_bregman(
        extended, weight=float(lambda_value), max_num_iter=max_num_iter,
        eps=eps, isotropic=True, channel_axis=None)[:height, :width]
    if not np.isfinite(filtered[valid]).all():
        raise RuntimeError('Bregman coherence filtering produced nonfinite valid samples.')
    filtered = np.clip(filtered, 0, 1).astype(np.float32)
    filtered[~valid] = np.nan
    return filtered


def _read_coherence_window(data, r0, r1, c0, c1):
    values = np.array(data[r0:r1, c0:c1], dtype=np.float32, copy=True)
    valid = np.isfinite(values) & (values >= 0) & (values <= 1)
    if '_FillValue' in data.attrs:
        valid &= values != np.asarray(data.attrs['_FillValue'], dtype=values.dtype)
    values[~valid] = np.nan
    return values


def _write_filtered_coherence(data, dst, settings, interferogram=None):
    """Filter overlapping 2-D tiles and write only their unpadded interiors."""
    size, pad = settings['block_size'], settings['block_pad']
    height, width = data.shape
    applied = False
    for row in range(0, height, size):
        bottom = min(row + size, height)
        r0, r1 = max(0, row-pad), min(height, bottom+pad)
        for col in range(0, width, size):
            right = min(col + size, width)
            c0, c1 = max(0, col-pad), min(width, right+pad)
            values = _read_coherence_window(data, r0, r1, c0, c1)
            interior = (slice(row-r0, bottom-r0), slice(col-c0, right-c0))
            if np.isfinite(values[interior]).any():
                if settings['method'] == 'bregman':
                    filtered = filter_coherence(
                        values, settings['lambda_value'], settings['max_num_iter'], settings['eps'])
                    applied = True
                else:
                    phase = None
                    if settings['method'] == 'complex_mean':
                        phase = np.array(interferogram[r0:r1, c0:c1], copy=True)
                        if '_FillValue' in interferogram.attrs:
                            fill = np.asarray(interferogram.attrs['_FillValue'], dtype=phase.dtype)
                            phase[phase == fill] = complex(np.nan, np.nan)
                    filtered, evaluated = mean_coherence(
                        values, settings['window_size'], settings['min_valid_fraction'],
                        phase, return_applied=True)
                    applied |= bool(evaluated[interior].any())
                output = filtered[interior]
            else:
                output = values[interior]
            dst.write(output, 1, window=Window(col, row, right-col, bottom-row))
    return applied


def record_usage(cfg, stage, used):
    with open(_status_path(cfg)) as stream:
        status = json.load(stream)
    pol_key = '_'.join(cfg.groups.processing.polarizations)
    status.setdefault(stage, {})[pol_key] = bool(used)
    with open(_status_path(cfg), 'w') as stream:
        json.dump(status, stream, indent=2)


def _dataset_grid(group, pol_group):
    # Product revisions have coordinates either alongside pol datasets or
    # at the parent interferogram grid. Never infer a nominal 20/80 m grid.
    def get(key):
        return pol_group[key] if key in pol_group else group[key]
    x, y = get('xCoordinates'), get('yCoordinates')
    dx, dy = float(get('xCoordinateSpacing')[()]), float(get('yCoordinateSpacing')[()])
    if dx <= 0 or dy >= 0:
        raise ValueError('GUNW coherence requires a north-up, west-left grid.')
    transform = Affine(dx, 0, float(x[0]) - dx / 2,
                       0, dy, float(y[0]) - dy / 2)
    return transform, rasterio.crs.CRS.from_epsg(int(get('projection')[()])), (len(y), len(x))


def extract_coherence(path, output, options, gcov_paths, lines_per_block,
                      require_native_20m=False):
    """Export a native coherence grid with explicit NaN nodata."""
    gcov_pols = set()
    for gcov in gcov_paths:
        with open_h5(gcov) as src:
            key = f'/science/LSAR/GCOV/grids/frequency{options.frequency}/listOfPolarizations'
            if key in src:
                gcov_pols.update(_text(pol)[:2] for pol in src[key][()])
    with open_h5(path) as src:
        group_path = f'{GRIDS}/frequency{options.frequency}/{options.source}'
        if group_path not in src:
            raise ValueError(f'Missing selected coherence grid {group_path}: {path}')
        group = src[group_path]
        available = sorted(pol for pol in gcov_pols if
                           f'{pol}/coherenceMagnitude' in group)
        if options.polarization == 'auto':
            # Prefer co-pol while allowing every supported polarization.
            available.sort(key=lambda pol: (pol[0] != pol[1], pol))
            pol = available[0] if available else None
        else:
            pol = options.polarization if options.polarization in available else None
        if pol is None:
            raise ValueError(f'No selected GUNW coherence polarization shared with GCOV: {path}')
        pol_group = group[pol]
        data = pol_group['coherenceMagnitude']
        transform, crs, shape = _dataset_grid(group, pol_group)
        if data.shape != shape:
            raise ValueError(f'GUNW coherence coordinates/shape disagree: {path}')
        filtering = coherence_filter_options(options)
        interferogram = None
        if filtering['enabled'] or require_native_20m:
            if not (np.isclose(transform.a, 20, rtol=0, atol=.1) and
                    np.isclose(-transform.e, 20, rtol=0, atol=.1)):
                raise ValueError(
                    'GUNW coherence filtering/raw support requires a native 20 m coherence grid; '
                    'select source: wrappedInterferogram for native 20 m processing. '
                    f'Found spacing ({transform.a}, {transform.e}): {path}')
        if filtering['enabled']:
            if filtering['method'] == 'complex_mean':
                if options.source != 'wrappedInterferogram' or 'wrappedInterferogram' not in pol_group:
                    raise ValueError('complex_mean requires the wrappedInterferogram dataset on the native coherence grid: ' + path)
                interferogram = pol_group['wrappedInterferogram']
                if interferogram.shape != shape or interferogram.dtype.kind != 'c':
                    raise ValueError('complex_mean requires a complex interferogram with the same shape as coherence: ' + path)
            logger.info('Filtering native 20 m GUNW coherence with %s: %s: %s',
                        filtering['method'], filtering, path)
        profile = dict(driver='GTiff', height=shape[0], width=shape[1], count=1,
                       dtype='float32', crs=crs, transform=transform, nodata=np.nan,
                       tiled=True, compress='deflate', BIGTIFF='IF_SAFER')
        with rasterio.open(output, 'w', **profile) as dst:
            applied = False
            if filtering['enabled']:
                applied = _write_filtered_coherence(data, dst, filtering, interferogram)
            else:
                for row in range(0, shape[0], lines_per_block):
                    end = min(row + lines_per_block, shape[0])
                    # Default HDF5 fillvalue=0 is not declared nodata: preserve
                    # zero unless the dataset explicitly marks it _FillValue.
                    values = _read_coherence_window(data, row, end, 0, shape[1])
                    dst.write(values, 1, window=Window(0, row, shape[1], end-row))
            dst.update_tags(GUNW_COHERENCE_FILTER_APPLIED=str(applied))
    return pol


def _mosaic_native_coherence(native_paths, reference_path, output_path, options,
                             filtered=False, raw=False):
    """Align and merge either the configured layer or the separate raw layer."""
    with ExitStack() as stack:
        ref = stack.enter_context(rasterio.open(reference_path))
        if raw and (abs(ref.transform.a) < 19.9 or abs(ref.transform.e) < 19.9):
            raise ValueError('Raw coherence support requires a processing grid of 20 m or coarser; finer grids duplicate native samples.')
        profile = ref.profile.copy()
        profile.update(count=1, dtype='float32', nodata=np.nan,
                       compress='deflate', BIGTIFF='IF_SAFER')
        views = []
        for native in native_paths:
            src = stack.enter_context(rasterio.open(native))
            views.append(stack.enter_context(WarpedVRT(
                src, crs=ref.crs, transform=ref.transform, width=ref.width,
                height=ref.height, nodata=np.nan,
                resampling=Resampling.nearest if raw else Resampling[options.resampling])))
        dst = stack.enter_context(rasterio.open(output_path, 'w', **profile))
        dst.update_tags(GUNW_COHERENCE_FILTER_APPLIED=str(filtered),
                        GUNW_COHERENCE_RAW_INPUT=str(raw),
                        GUNW_COHERENCE_FILTER_SETTINGS=json.dumps(
                            {'enabled': False} if raw else coherence_filter_options(options)))
        has_data = False
        for _, window in dst.block_windows(1):
            shape = (int(window.height), int(window.width))
            total = np.zeros(shape, dtype=np.float32)
            count = np.zeros(shape, dtype=np.uint32)
            for view in views:
                values = view.read(1, window=window)
                valid = np.isfinite(values) & (values >= 0) & (values <= 1)
                if raw or options.mosaic_mode == 'first':
                    valid &= count == 0
                total[valid] += values[valid]
                count[valid] += 1
            valid = count > 0
            has_data |= bool(valid.any())
            output = np.full(shape, np.nan, dtype=np.float32)
            np.divide(total, count, out=output, where=valid)
            dst.write(output, 1, window=window)
        if not has_data:
            raise ValueError('Supplied GUNW coherence has no valid overlap with the GCOV mosaic.')


def mosaic(cfg, reference_path):
    """Resample each GUNW directly to the exact GCOV mosaic grid, then merge."""
    if not (enabled(cfg, 'fuzzy') or enabled(cfg, 'landcover_mask')):
        return
    options = cfg.groups.processing.gunw_coherence
    scratch = cfg.groups.product_path_group.scratch_path
    inputs = cfg.groups.input_file_group
    _, _, matches = split_inputs(inputs.input_file_path + inputs.gunw_file_path)
    # An interrupted/failed rerun must not leave previous usage marked valid.
    Path(_status_path(cfg)).unlink(missing_ok=True)
    records = []
    use_raw = _needs_raw_support(cfg)
    with TemporaryDirectory(prefix='gunw_', dir=scratch) as temporary:
        native_paths = []
        raw_paths = []
        for index, path in enumerate(inputs.gunw_file_path):
            native = os.path.join(temporary, f'{index}.tif')
            pol = extract_coherence(path, native, options, matches[path],
                                    cfg.groups.processing.mosaic.read_row_blk_size,
                                    require_native_20m=use_raw)
            native_paths.append(native)
            with rasterio.open(native) as src:
                filtered = src.tags().get('GUNW_COHERENCE_FILTER_APPLIED') == 'True'
            records.append({'path': path, 'polarization': pol,
                            'gcov': matches[path], 'filtered': filtered})
            if use_raw:
                raw_native = native
                if coherence_filter_options(options)['enabled']:
                    raw_native = os.path.join(temporary, f'{index}_raw.tif')
                    raw_options = copy.deepcopy(options)
                    raw_options.filter = SimpleNamespace(enabled=False)
                    extract_coherence(path, raw_native, raw_options, matches[path],
                                      cfg.groups.processing.mosaic.read_row_blk_size,
                                      require_native_20m=True)
                raw_paths.append(raw_native)
        _mosaic_native_coherence(native_paths, reference_path, raster_path(cfg), options,
                                 filtered=any(item['filtered'] for item in records))
        if use_raw:
            _mosaic_native_coherence(raw_paths, reference_path, raw_raster_path(cfg), options, raw=True)
    with open(_status_path(cfg), 'w') as stream:
        json.dump({'sources': records, 'options': _mosaic_options(cfg),
                   'fuzzy': {}, 'landcover_mask': {}}, stream, indent=2)
    logger.info('GUNW coherence aligned to GCOV: %s', raster_path(cfg))


def check_grid(cfg, reference_path, raw=False):
    """Catch missing/stale or misaligned intermediate rasters before use."""
    path = raw_raster_path(cfg) if raw else raster_path(cfg)
    with rasterio.open(path) as coherence, rasterio.open(reference_path) as ref:
        if (coherence.shape != ref.shape or coherence.crs != ref.crs or
                coherence.transform != ref.transform):
            raise ValueError('GUNW coherence and processing raster grids differ; rerun mosaic.')
        if raw and coherence.tags().get('GUNW_COHERENCE_RAW_INPUT') != 'True':
            raise ValueError('Raw support requires the separate unsmoothed coherence mosaic; rerun mosaic.')
    with open(_status_path(cfg)) as stream:
        status = json.load(stream)
    current = set(cfg.groups.input_file_group.gunw_file_path)
    if {item['path'] for item in status['sources']} != current:
        raise ValueError('GUNW coherence inputs changed; rerun mosaic.')
    previous_options = dict(status.get('options', {}))
    # Old unfiltered intermediates remain valid while filtering is disabled.
    previous_options.setdefault('filter', {'enabled': False})
    previous_options.setdefault('raw_support', False)
    if previous_options != _mosaic_options(cfg):
        raise ValueError('GUNW coherence mosaic options changed; rerun mosaic.')


def combine_fuzzy(original, intensity_membership, coherence, options, zmf):
    """Add low-coherence water membership; missing coherence is neutral.

    The existing intensity contribution remains 0.5. Coherence is added as
    a fourth equally weighted ancillary membership (HAND, slope, reference
    water, coherence). This helper receives the existing mean result.
    """
    valid = np.isfinite(coherence) & (coherence >= 0) & (coherence <= 1)
    result = original.copy()
    membership = zmf(coherence, options.member_min, options.member_max)
    ancillary = (original - intensity_membership * 0.5) * 2
    updated = intensity_membership * 0.5 + (ancillary * 3 + membership) / 4 * 0.5
    result[valid] = updated[valid]
    return result


def high_coherence_mask(coherence, landcover, options):
    return (np.isfinite(coherence) & (coherence >= options.threshold) &
            (coherence <= 1) & np.isin(landcover, options.worldcover_classes))


def coherence_support_mask(coherence, landcover, options):
    """Require a high raw center plus high-coherence support in target classes.

    raw_support counts the exact center WorldCover class; raw_support_pooled
    combines all selected classes into one group. The center is included.
    Support counts high / valid group samples, while coverage counts valid
    group samples / full window area (including off-image/missing pixels).
    """
    from scipy.ndimage import convolve

    settings = landcover_mask_options(options)
    if settings['method'] not in RAW_SUPPORT_METHODS:
        raise ValueError('coherence_support_mask requires a raw support method.')
    if coherence.ndim != 2 or coherence.shape != landcover.shape:
        raise ValueError('Raw coherence and landcover must have matching 2-D shapes.')
    size = settings['window_size']
    kernel = np.ones((size, size), dtype=np.int32)
    valid = np.isfinite(coherence) & (coherence >= 0) & (coherence <= 1)
    reject = np.zeros(coherence.shape, dtype=bool)
    support = np.full(coherence.shape, np.nan, dtype=np.float32)
    coverage = np.full(coherence.shape, np.nan, dtype=np.float32)
    if settings['method'] == 'raw_support_pooled':
        groups = (np.isin(landcover, options.worldcover_classes),)
    else:
        groups = (landcover == code for code in set(options.worldcover_classes))
    for group in groups:
        center = valid & group
        if not center.any():
            continue
        count = convolve(center.astype(np.int32), kernel, mode='constant', cval=0)
        high = center & (coherence >= options.threshold)
        high_count = convolve(high.astype(np.int32), kernel, mode='constant', cval=0)
        fraction = np.zeros(coherence.shape, dtype=np.float64)
        np.divide(high_count, count, out=fraction, where=count > 0)
        support[center] = fraction[center]
        coverage[center] = count[center] / size**2
        reject |= (high & (count >= settings['min_valid_fraction'] * size**2) &
                   (fraction + 1e-12 >= settings['support_fraction']))
    return reject, support, coverage


def mask_water(cfg, water_path, landcover_path, record=True):
    """Remove selected coherent land pixels from the intermediate water map."""
    options = cfg.groups.processing.gunw_coherence.landcover_mask
    settings = landcover_mask_options(options)
    use_raw = settings['method'] in RAW_SUPPORT_METHODS
    check_grid(cfg, water_path, raw=use_raw)
    used = False
    removed_count = 0
    pad = settings['window_size'] // 2 if use_raw else 0
    pol_key = '_'.join(cfg.groups.processing.polarizations)
    diagnostic_paths = {
        'support': os.path.join(cfg.groups.product_path_group.scratch_path,
                               f'gunw_coherence_support_{pol_key}.tif'),
        'coverage': os.path.join(cfg.groups.product_path_group.scratch_path,
                                f'gunw_coherence_support_coverage_{pol_key}.tif'),
        'removed': os.path.join(cfg.groups.product_path_group.scratch_path,
                               f'gunw_coherence_removed_water_{pol_key}.tif'),
    }
    with TemporaryDirectory(prefix='gunw_mask_', dir=os.path.dirname(water_path)) as temporary, \
            ExitStack() as stack:
        water = stack.enter_context(rasterio.open(water_path))
        landcover = stack.enter_context(rasterio.open(landcover_path))
        coherence = stack.enter_context(rasterio.open(raw_raster_path(cfg) if use_raw else raster_path(cfg)))
        if (landcover.shape != water.shape or landcover.transform != water.transform or
                landcover.crs != water.crs):
            raise ValueError('Landcover and intermediate water grids differ.')
        output = os.path.join(temporary, 'water.tif')
        profile = water.profile.copy()
        profile.update(driver='GTiff', BIGTIFF='IF_SAFER')
        dst = stack.enter_context(rasterio.open(output, 'w', **profile))
        dst.update_tags(**water.tags())
        diagnostics = {}
        previous_removed = None
        if use_raw and settings['save_diagnostics']:
            if not record and os.path.exists(diagnostic_paths['removed']):
                previous_removed = stack.enter_context(rasterio.open(diagnostic_paths['removed']))
            for key in diagnostic_paths:
                diagnostic_profile = profile.copy()
                diagnostic_profile.update(count=1, dtype='uint8' if key == 'removed' else 'float32',
                                          nodata=255 if key == 'removed' else np.nan)
                diagnostics[key] = stack.enter_context(rasterio.open(
                    os.path.join(temporary, key + '.tif'), 'w', **diagnostic_profile))
        for _, window in water.block_windows(1):
            values = water.read(1, window=window)
            row, col = int(window.row_off), int(window.col_off)
            bottom, right = row + int(window.height), col + int(window.width)
            r0, c0 = max(0, row-pad), max(0, col-pad)
            r1, c1 = min(water.height, bottom+pad), min(water.width, right+pad)
            read_window = Window(c0, r0, c1-c0, r1-r0)
            inner = (slice(row-r0, bottom-r0), slice(col-c0, right-c0))
            coh = coherence.read(1, window=read_window)
            lc = landcover.read(1, window=read_window)
            if use_raw:
                mask, support, coverage = coherence_support_mask(coh, lc, options)
            else:
                mask = high_coherence_mask(coh, lc, options)
            valid = (np.isfinite(coh[inner]) & (coh[inner] >= 0) & (coh[inner] <= 1) &
                     np.isin(lc[inner], options.worldcover_classes) & (values == 1))
            used |= bool(valid.any())
            removed = (values == 1) & mask[inner]
            removed_count += int(removed.sum())
            if diagnostics:
                diagnostics['support'].write(support[inner], 1, window=window)
                diagnostics['coverage'].write(coverage[inner], 1, window=window)
                changed = removed.astype(np.uint8)
                if previous_removed is not None:
                    changed |= (previous_removed.read(1, window=window) == 1)
                changed[(values != 0) & (values != 1)] = 255
                diagnostics['removed'].write(changed, 1, window=window)
            values[removed] = 0
            dst.write(values, 1, window=window)
        # A separate output also supports COG inputs without breaking their
        # layout through in-place updates.
        stack.close()
        os.replace(output, water_path)
        for key in diagnostics:
            os.replace(os.path.join(temporary, key + '.tif'), diagnostic_paths[key])
    if record:
        record_usage(cfg, 'landcover_mask', used)
    logger.info('Coherence landcover rejection (%s): removed %d water pixels', settings['method'], removed_count)


def metadata(cfg, gcov_paths):
    """Record actual stage usage, separately from presence of supplied files."""
    result = {'GUNW_COHERENCE_USED': False,
              'GUNW_COHERENCE_FUZZY_USED': False,
              'GUNW_COHERENCE_LANDCOVER_MASK_USED': False,
              'GUNW_COHERENCE_FILTER_APPLIED': False,
              'GUNW_COHERENCE_RAW_SUPPORT_USED': False}
    if not (enabled(cfg, 'fuzzy') or enabled(cfg, 'landcover_mask')):
        return result
    with open(_status_path(cfg)) as stream:
        status = json.load(stream)
    sources = [item for item in status['sources']
               if set(item['gcov']) & set(gcov_paths)]
    fuzzy = bool(enabled(cfg, 'fuzzy') and sources and any(status['fuzzy'].values()))
    masking = bool(enabled(cfg, 'landcover_mask') and sources and
                   any(status['landcover_mask'].values()))
    options = cfg.groups.processing.gunw_coherence
    filtering = coherence_filter_options(options)
    mask_settings = landcover_mask_options(options.landcover_mask)
    raw_support = bool(masking and mask_settings['method'] in RAW_SUPPORT_METHODS)
    filter_applied = bool((fuzzy or (masking and not raw_support)) and
                          any(item.get('filtered', False) for item in sources))
    result.update({
        'GUNW_COHERENCE_USED': fuzzy or masking,
        'GUNW_COHERENCE_FUZZY_USED': fuzzy,
        'GUNW_COHERENCE_LANDCOVER_MASK_USED': masking,
        'GUNW_COHERENCE_SOURCE': options.source,
        'GUNW_COHERENCE_FREQUENCY': options.frequency,
        'GUNW_COHERENCE_INPUTS': ','.join(Path(item['path']).name for item in sources),
        'GUNW_COHERENCE_POLARIZATIONS': ','.join(sorted({item['polarization'] for item in sources})),
        'GUNW_COHERENCE_RESAMPLING': options.resampling,
        'GUNW_COHERENCE_MOSAIC_MODE': options.mosaic_mode,
        'GUNW_COHERENCE_FUZZY_MEMBER_MIN': options.fuzzy.member_min,
        'GUNW_COHERENCE_FUZZY_MEMBER_MAX': options.fuzzy.member_max,
        'GUNW_COHERENCE_HIGH_THRESHOLD': options.landcover_mask.threshold,
        'GUNW_COHERENCE_WORLDCOVER_CLASSES': ','.join(map(str, options.landcover_mask.worldcover_classes)),
        'GUNW_COHERENCE_FILTER_APPLIED': filter_applied,
        'GUNW_COHERENCE_FILTER_SETTINGS': json.dumps(filtering),
        'GUNW_COHERENCE_RAW_SUPPORT_USED': raw_support,
        'GUNW_COHERENCE_LANDCOVER_SETTINGS': json.dumps(mask_settings),
        'GUNW_COHERENCE_LANDCOVER_INPUT': 'raw_nearest_first' if raw_support else 'configured_mosaic',
    })
    return result


def compare_filters(cfg, reference_path):
    """Build six isolated 20 m variants and their paired histogram report.

    This is an opt-in coherence comparison only: no fuzzy/classification
    stages are executed and the normal workflow mosaic is not overwritten.
    Every variant reads its GUNW inputs independently.
    """
    if not getattr(cfg.groups.input_file_group, 'gunw_file_path', []):
        raise ValueError('--compare-filters requires GUNW inputs.')
    root = Path(cfg.groups.product_path_group.scratch_path) / 'coherence_filter_comparison'
    reference_path = os.path.abspath(reference_path)
    with rasterio.open(reference_path):
        pass
    variants = [('raw_20m', None, 3), ('bregman', 'bregman', 3)]
    variants += [(f'{method}_{size}x{size}', method, size)
                 for method in ('magnitude_mean', 'complex_mean') for size in (3, 5)]
    paths = {}
    for name, method, window in variants:
        case = copy.deepcopy(cfg)
        output = root / name
        output.mkdir(parents=True, exist_ok=True)
        case.groups.product_path_group.scratch_path = str(output)
        options = case.groups.processing.gunw_coherence
        options.source = 'wrappedInterferogram'
        options.fuzzy.enabled = True  # enable mosaic production, not fuzzy execution
        options.landcover_mask.enabled = False  # comparisons are coherence-only
        if not hasattr(options, 'filter'):
            options.filter = SimpleNamespace()
        options.filter.enabled = method is not None
        if method is not None:
            options.filter.method = method
            options.filter.window_size = window
            if method != 'bregman':
                options.filter.block_pad = max(getattr(options.filter, 'block_pad', 32), window // 2)
        logger.info('Building coherence comparison variant: %s', name)
        mosaic(case, reference_path)
        paths[name] = raster_path(case)

    bins = np.linspace(0, 1, 101)
    delta_bins = np.linspace(-1, 1, 201)
    stats = {name: dict(n=0, total=0., squares=0., delta=0., below=0, up=0, down=0,
                        histogram=np.zeros(100, dtype=np.int64),
                        delta_histogram=np.zeros(200, dtype=np.int64)) for name in paths}
    with ExitStack() as stack:
        rasters = {name: stack.enter_context(rasterio.open(path)) for name, path in paths.items()}
        for _, window in rasters['raw_20m'].block_windows(1):
            arrays = {name: src.read(1, window=window) for name, src in rasters.items()}
            common = np.logical_and.reduce([np.isfinite(array) for array in arrays.values()])
            raw = arrays['raw_20m'][common].astype(np.float64)
            for name, array in arrays.items():
                values = array[common].astype(np.float64)
                delta = values - raw
                item = stats[name]
                item['n'] += values.size
                item['total'] += float(values.sum())
                item['squares'] += float(np.square(values).sum())
                item['delta'] += float(delta.sum())
                item['below'] += int((values < .1).sum())
                item['up'] += int(((raw < .1) & (values >= .1)).sum())
                item['down'] += int(((raw >= .1) & (values < .1)).sum())
                item['histogram'] += np.histogram(values, bins=bins)[0]
                item['delta_histogram'] += np.histogram(delta, bins=delta_bins)[0]
    rows = []
    for name, item in stats.items():
        n = item['n']
        if not n:
            raise ValueError('No common valid pixels for filter comparison.')
        mean = item['total'] / n
        rows.append(dict(method=name, n=n, mean=mean,
                         std=np.sqrt(max(0., item['squares']/n - mean**2)),
                         mean_delta=item['delta']/n, below_0_1_percent=100*item['below']/n,
                         up_cross_percent=100*item['up']/n, down_cross_percent=100*item['down']/n))
    with open(root / 'metrics.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    histograms = {
        'scope': 'entire GCOV mosaic, same common valid pixels for all variants',
        'value_bins': bins.tolist(), 'delta_bins': delta_bins.tolist(),
        'variants': {name: dict(path=paths[name], n=item['n'],
                               histogram=item['histogram'].tolist(),
                               delta_histogram=item['delta_histogram'].tolist())
                     for name, item in stats.items()},
    }
    with open(root / 'histograms.json', 'w') as stream:
        json.dump(histograms, stream, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for index, (name, item) in enumerate(stats.items()):
        color = plt.colormaps['tab10'](index)
        density = item['histogram'] / item['n'] / np.diff(bins)
        axes[0].stairs(density, bins, label=name, color=color)
        axes[1].plot(bins, np.r_[0., np.cumsum(item['histogram'])/item['n']], label=name, color=color)
        if name != 'raw_20m':
            axes[2].stairs(item['delta_histogram']/item['n']/np.diff(delta_bins),
                           delta_bins, label=name, color=color)
    axes[0].set(xlabel='Coherence / complex score', ylabel='Density', title='Value histogram')
    axes[1].set(xlabel='Coherence / complex score', ylabel='Cumulative fraction', title='CDF')
    axes[2].set(xlabel='Processed - raw', ylabel='Density', title='Paired change histogram')
    axes[2].axvline(0, color='black', linestyle='--')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'Entire mosaic: {rows[0]["n"]:,} common valid pixels')
    fig.savefig(root / 'histogram_comparison.png', dpi=160)
    plt.close(fig)
    logger.info('Coherence comparison outputs: %s', root)
    return paths


if __name__ == '__main__':
    # Rebuild coherence using an existing GCOV mosaic, without repeating GCOV
    # extraction/mosaicking. This also serves standalone stage workflows.
    from dswx_sar.nisar.dswx_ni_runconfig import RunConfig, _get_parser
    from dswx_sar.common import _generate_log

    parser = _get_parser()
    parser.add_argument('--compare-filters', action='store_true',
                        help='Build raw/Bregman/scalar/complex 3x3 and 5x5 coherence variants and histograms.')
    args = parser.parse_args()
    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)
    _generate_log.configure_log_file(cfg.groups.log_file)
    reference = os.path.join(
        cfg.groups.product_path_group.scratch_path,
        f'{cfg.groups.processing.mosaic.mosaic_prefix}_{cfg.groups.processing.polarizations[0]}.tif')
    if args.compare_filters:
        compare_filters(cfg, reference)
    else:
        mosaic(cfg, reference)
