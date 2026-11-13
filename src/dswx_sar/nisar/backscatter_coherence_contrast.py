"""Independent IV detector using filtered co-pol power and GUNW coherence.

No RGB product, dual-pol ratio or fuzzy refinement is used. Fixed normalization
limits reproduce the previously inspected display contrast across all dates.
"""
from contextlib import ExitStack
import json
import os

import numpy as np
import rasterio
from rasterio.windows import Window

METHOD = 'backscatter_coherence_contrast'


def requested(cfg):
    iv = cfg.groups.processing.inundated_vegetation
    return (iv.enabled is not False and getattr(iv, 'method', 'dual_pol_ratio') == METHOD)


def validate(options):
    detector = getattr(options, 'detector', 'contrast')
    if detector not in ('contrast', 'rectangle'):
        raise ValueError('detector must be contrast or rectangle.')
    if getattr(options, 'input_units', 'power') not in ('power', 'amplitude'):
        raise ValueError('input_units must be power or amplitude for filtered_image input.')
    if detector == 'rectangle':
        for name in ('db_bounds', 'coherence_bounds'):
            bounds = np.asarray(getattr(options, name), dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f'{name} must have two finite increasing bounds.')
        if not 0 <= options.coherence_bounds[0] < options.coherence_bounds[1] <= 1:
            raise ValueError('coherence_bounds must be within [0, 1].')
        return
    for name in ('raw_coherence_limits', 'filtered_coherence_limits', 'backscatter_db_limits'):
        limits = np.asarray(getattr(options, name), dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f'{name} must contain two finite increasing values.')
    for name in ('contrast_threshold', 'backscatter_min', 'raw_coherence_max'):
        value = getattr(options, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be within [0, 1].')


def classify(power, raw, filtered, target, options):
    """Return candidate, contrast, and data-valid mask; power is linear power."""
    validate(options)
    if not (power.shape == raw.shape == filtered.shape == target.shape):
        raise ValueError('BCC inputs must share a grid and shape.')
    rectangle = getattr(options, 'detector', 'contrast') == 'rectangle'
    valid = (np.isfinite(power) & (power > 0) & np.isfinite(raw)
             & (raw >= 0) & (raw <= 1))
    if not rectangle:
        valid &= np.isfinite(filtered) & (filtered >= 0) & (filtered <= 1)
    db = np.full(power.shape, np.nan, dtype=float)
    factor = 20 if getattr(options, 'input_units', 'power') == 'amplitude' else 10
    db[valid] = factor * np.log10(power[valid])
    if rectangle:
        dlo, dhi = options.db_bounds
        clo, chi = options.coherence_bounds
        candidate = (valid & target.astype(bool) & (db > dlo) & (db < dhi)
                     & (raw > clo) & (raw < chi))
        # No scalar contrast score is used in rectangle mode. Keep the existing
        # output contract, but explicitly mark its score raster as unavailable.
        return candidate, np.full(db.shape, np.nan, dtype=np.float32), valid

    def normalize(values, limits):
        return np.clip((values - limits[0]) / (limits[1] - limits[0]), 0, 1)

    amplitude = normalize(db, options.backscatter_db_limits)
    r = normalize(raw, options.raw_coherence_limits)
    f = normalize(filtered, options.filtered_coherence_limits)
    contrast = np.where(valid, np.minimum(amplitude - r, amplitude - f), np.nan)
    candidate = (valid & target.astype(bool)
                 & (amplitude > options.backscatter_min)
                 & (contrast > options.contrast_threshold)
                 & (raw < options.raw_coherence_max))
    return candidate, contrast, valid


def _same_grid(dataset, reference):
    if (dataset.shape != reference.shape or dataset.crs != reference.crs
            or dataset.transform != reference.transform):
        raise ValueError(f'Grid mismatch: {dataset.name}; rerun preprocessing/mosaic.')


def run(cfg):
    # Lazy imports keep the numerical classifier independently testable.
    from dswx_sar.nisar import gunw_coherence
    from dswx_sar.common._masking_with_ancillary import get_label_landcover_esa_10
    from dswx_sar.common._detect_inundated_vegetation import parse_ranges

    proc = cfg.groups.processing
    iv = proc.inundated_vegetation
    options = iv.backscatter_coherence_contrast
    validate(options)
    scratch = cfg.groups.product_path_group.scratch_path
    pols = list(proc.polarizations) + list(proc.polarimetric_option or [])
    key = '_'.join(pols)
    pol = options.polarization
    if pol == 'auto':
        pol = next((p for p in pols if p in ('HH', 'VV')), None)
    if pol not in pols or pol not in ('HH', 'VV'):
        raise ValueError('BCC requires the selected HH or VV filtered power band.')
    power_path = os.path.join(scratch, f'filtered_image_{key}.tif')
    gunw_coherence.check_grid(cfg, power_path)
    gunw_coherence.check_grid(cfg, power_path, raw=True)
    if not proc.gunw_coherence.filter.enabled:
        raise ValueError('BCC requires gunw_coherence.filter.enabled: true; rerun mosaic.')
    with open(gunw_coherence._status_path(cfg)) as stream:
        status = json.load(stream)
    if any(item['polarization'] != pol for item in status['sources']):
        raise ValueError('BCC power and GUNW polarizations differ; set gunw_coherence.polarization explicitly.')

    world_path = os.path.join(scratch, 'interpolated_landcover.tif')
    glad_path = os.path.join(scratch, 'interpolated_glad.tif')
    source = iv.target_area_file_type
    if source == 'auto':
        source = 'GLAD' if os.path.isfile(glad_path) else 'WorldCover'
    if source not in ('GLAD', 'WorldCover'):
        raise ValueError(f'Unknown IV target source: {source}')
    wc_codes = [get_label_landcover_esa_10()[name] for name in iv.target_worldcover_class]
    glad_codes = parse_ranges(iv.target_glad_class) if source == 'GLAD' else []
    with ExitStack() as stack:
        power = stack.enter_context(rasterio.open(power_path))
        raw = stack.enter_context(rasterio.open(gunw_coherence.raw_raster_path(cfg)))
        filtered = stack.enter_context(rasterio.open(gunw_coherence.raster_path(cfg)))
        world = stack.enter_context(rasterio.open(world_path))
        glad = stack.enter_context(rasterio.open(glad_path)) if source == 'GLAD' else None
        aoi_path = getattr(options, 'aoi_mask_path', '')
        aoi = stack.enter_context(rasterio.open(aoi_path)) if aoi_path else None
        for dataset in (raw, filtered, world, glad, aoi):
            if dataset is not None:
                _same_grid(dataset, power)
        profile = power.profile.copy()
        profile.update(count=1, dtype='uint8', nodata=None)
        # Preserve downstream filename contracts. high_dualpol_ratio is a
        # compatibility alias for the BCC candidate, NOT a ratio measurement.
        outputs = {}
        for name in ('temp_inundated_vegetation', 'temp_target_area',
                     'temp_high_dualpol_ratio', 'temp_bcc_candidate', 'temp_bcc_valid'):
            dst = stack.enter_context(rasterio.open(os.path.join(scratch, f'{name}_{key}.tif'), 'w', **profile))
            dst.update_tags(IV_METHOD=METHOD, IV_DETECTOR=getattr(options, 'detector', 'contrast'), IV_PARAMETERS=json.dumps(vars(options)))
            outputs[name] = dst
        score_profile = dict(profile, dtype='float32', nodata=np.nan)
        score = stack.enter_context(rasterio.open(os.path.join(scratch, f'temp_bcc_score_{key}.tif'), 'w', **score_profile))
        score.update_tags(IV_METHOD=METHOD, IV_DETECTOR=getattr(options, 'detector', 'contrast'), SCORE_USED=str(getattr(options, 'detector', 'contrast') == 'contrast'), IV_PARAMETERS=json.dumps(vars(options)))
        used = False
        for row in range(0, power.height, int(iv.line_per_block)):
            window = Window(0, row, power.width, min(int(iv.line_per_block), power.height-row))
            def read(ds, band=1):
                return ds.read(band, window=window, masked=True).astype('float64').filled(np.nan)
            wc_target = np.isin(read(world), wc_codes)
            target = wc_target.astype('uint8')
            if glad is not None:
                g = read(glad)
                target = np.isin(g, glad_codes).astype('uint8')
                target[((g == 255) | ~np.isfinite(g)) & wc_target] = 2
            if aoi is not None:
                a = read(aoi)
                target[~(np.isfinite(a) & (a > 0))] = 0
            candidate, contrast, valid = classify(read(power, pols.index(pol)+1), read(raw), read(filtered), target > 0, options)
            used |= bool((valid & (target > 0)).any())
            target[~valid] = 0
            outputs['temp_inundated_vegetation'].write(candidate.astype('uint8') * 2, 1, window=window)
            outputs['temp_target_area'].write(target, 1, window=window)
            for name in ('temp_high_dualpol_ratio', 'temp_bcc_candidate'):
                outputs[name].write(candidate.astype('uint8'), 1, window=window)
            outputs['temp_bcc_valid'].write(valid.astype('uint8'), 1, window=window)
            score.write(contrast.astype('float32'), 1, window=window)
    gunw_coherence.record_usage(cfg, 'bcc_iv', used)
