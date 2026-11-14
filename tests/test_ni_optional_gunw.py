"""Regression checks using tiny real HDF5/GeoTIFF inputs (no mission data).

Run: PYTHONPATH=src pytest -q tests/test_ni_optional_gunw.py
GDAL-independent functions are loaded by AST where their containing module
imports the full GDAL workflow; their numerical dependencies are not mocked.
"""
import ast
import copy
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import h5py
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from ruamel.yaml import YAML
from scipy import ndimage
from skimage.segmentation import watershed

from dswx_sar.nisar import gunw_coherence as gc
from dswx_sar.nisar import backscatter_coherence_contrast as bcc
from dswx_sar.nisar.dswx_ni_runconfig import (
    check_polarizations, DSWX_NI_SINGLE_FRAME_POL_DICT, load_validate_yaml,
    wrap_namespace,
)

ROOT = Path(__file__).resolve().parents[1]


def function_from_file(relative, name, extra=None):
    tree = ast.parse((ROOT / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(np=np, cv2=cv2, ndimage=ndimage, watershed=watershed)
    namespace.update(extra or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, 'exec'), namespace)
    return namespace[name]


def config(tmp_path, pols=('VV', 'VH'), method=bcc.METHOD):
    defaults = YAML(typ='safe').load(ROOT / 'src/dswx_sar/defaults/algorithm_parameter_ni.yaml')
    proc = wrap_namespace(defaults['runconfig']['processing'])
    proc.polarizations = list(pols)
    proc.polarimetric_option = []
    proc.inundated_vegetation.method = method
    proc.gunw_coherence.source = 'wrappedInterferogram'
    proc.gunw_coherence.filter.enabled = False
    proc.inundated_vegetation.backscatter_coherence_contrast.detector = 'rectangle'
    return NS(groups=NS(processing=proc,
                        input_file_group=NS(input_file_path=[], gunw_file_path=[]),
                        product_path_group=NS(scratch_path=str(tmp_path))))


def h5_input(path, product, pols=('VV', 'VH'), frame=1, offset=0):
    with h5py.File(path, 'w') as f:
        ident = f.create_group(gc.IDENTIFICATION)
        for k, v in dict(productType=product, trackNumber=1, frameNumber=frame,
                         orbitPassDirection='Descending').items():
            ident[k] = v
        prefixes = ('',) if product == 'GCOV' else ('reference', 'secondary')
        for prefix in prefixes:
            for suffix, time in [('Start', '2026-01-01T00:00:00'), ('End', '2026-01-01T00:01:00')]:
                key = prefix + 'ZeroDoppler' + suffix + 'Time' if prefix else 'zeroDoppler' + suffix + 'Time'
                ident[key] = time
        freq = f.create_group(f'/science/LSAR/{product}/grids/frequencyA')
        freq['listOfPolarizations'] = np.array(pols, dtype='S2')
        if product == 'GUNW':
            grid = freq.create_group('wrappedInterferogram')
            grid['xCoordinates'] = 500010 + offset + np.arange(8) * 20
            grid['yCoordinates'] = 4099990 - np.arange(8) * 20
            grid['xCoordinateSpacing'] = 20.
            grid['yCoordinateSpacing'] = -20.
            grid['projection'] = 32611
            for pol in pols:
                group = grid.create_group(pol)
                group['coherenceMagnitude'] = np.full((8, 8), .62 if pol == 'VV' else .4, np.float32)
                group['wrappedInterferogram'] = np.ones((8, 8), np.complex64)
    return str(path)


def raster(path, values, offset=0):
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[None]
    with rasterio.open(path, 'w', driver='GTiff', height=values.shape[1], width=values.shape[2],
                       count=values.shape[0], dtype=values.dtype, crs='EPSG:32611',
                       transform=from_origin(500000 + offset, 4100000, 20, 20), nodata=None) as out:
        out.write(values)
    return str(path)


@pytest.mark.parametrize('pols', [('HH', 'HV'), ('VV', 'VH'), ('HH',), ('VV',), ('HH', 'VV', 'HV', 'VH')])
@pytest.mark.parametrize('selection', [['auto'], 'auto', None])
def test_gcov_auto(tmp_path, pols, selection):
    path = h5_input(tmp_path / 'gcov.h5', 'GCOV', pols)
    result = check_polarizations(selection, [dict(input_path=path, valid_freqs=['A'])],
                                DSWX_NI_SINGLE_FRAME_POL_DICT)
    assert set(result[2]) == set(pols)
    assert result[0] == sorted(set(pols) & {'HH', 'VV'})


@pytest.mark.parametrize('pols,expected', [(('HH', 'HV'), True), (('VV', 'VH'), True),
                                          (('HH',), False), (('VV',), False)])
@pytest.mark.parametrize('iv_switch', ['auto', True])
def test_no_gunw_fallback(tmp_path, pols, expected, iv_switch):
    cfg = config(tmp_path, pols)
    cfg.groups.processing.inundated_vegetation.enabled = iv_switch
    cfg.groups.processing.gunw_coherence.fuzzy.enabled = True
    cfg.groups.processing.gunw_coherence.landcover_mask.enabled = True
    assert not bcc.requested(cfg)
    assert bcc.iv_enabled(cfg) is expected
    # Must return before opening a reference raster, HDF5 or stale status.
    gc.mosaic(cfg, 'nonexistent.tif')
    assert not gc.enabled(cfg, 'fuzzy')
    assert not gc.metadata(cfg, [])['GUNW_COHERENCE_USED']


def test_gunw_matching_rejects_wrong_frame(tmp_path):
    a = h5_input(tmp_path / 'a.h5', 'GCOV')
    b = h5_input(tmp_path / 'b.h5', 'GUNW', frame=2)
    with pytest.raises(ValueError, match='no matching GCOV'):
        gc.split_inputs([a, b])


@pytest.mark.parametrize('pols', [('HH', 'HV'), ('VV', 'VH'), ('HH', 'VV', 'HV', 'VH')])
def test_gunw_auto_is_copol(tmp_path, pols):
    cfg = config(tmp_path, pols)
    a = h5_input(tmp_path / 'a.h5', 'GCOV', pols)
    b = h5_input(tmp_path / 'b.h5', 'GUNW', pols)
    pol = gc.extract_coherence(b, str(tmp_path / 'coh.tif'), cfg.groups.processing.gunw_coherence,
                              [a], 4)
    assert pol == ('HH' if 'HH' in pols else 'VV')
    cfg.groups.processing.gunw_coherence.polarization = 'HH' if 'HH' not in pols else 'VH'
    if cfg.groups.processing.gunw_coherence.polarization not in pols:
        with pytest.raises(ValueError, match='No selected'):
            gc.extract_coherence(b, str(tmp_path / 'bad.tif'), cfg.groups.processing.gunw_coherence, [a], 4)


def test_auto_does_not_silently_use_crosspol(tmp_path):
    cfg = config(tmp_path, ('HV',))
    a = h5_input(tmp_path / 'a.h5', 'GCOV', ('HV',))
    b = h5_input(tmp_path / 'b.h5', 'GUNW', ('HV',))
    with pytest.raises(ValueError, match='No selected'):
        gc.extract_coherence(b, str(tmp_path / 'out.tif'), cfg.groups.processing.gunw_coherence, [a], 4)


@pytest.mark.parametrize('pol', ['HH', 'VV'])
@pytest.mark.parametrize('filter_method', [None, 'adaptive_guided', 'magnitude_mean'])
def test_real_mosaic_and_grid(tmp_path, pol, filter_method):
    cfg = config(tmp_path, (pol,))
    a = h5_input(tmp_path / 'a.h5', 'GCOV', (pol,))
    b = h5_input(tmp_path / 'b.h5', 'GUNW', (pol,))
    cfg.groups.input_file_group.input_file_path = [a]
    cfg.groups.input_file_group.gunw_file_path = [b]
    filtering = cfg.groups.processing.gunw_coherence.filter
    filtering.enabled = filter_method is not None
    if filter_method:
        filtering.method = filter_method
    reference = raster(tmp_path / 'reference.tif', np.full((8, 8), .04, np.float32))
    prefix = cfg.groups.processing.mosaic.mosaic_prefix
    raster(tmp_path / f'{prefix}_{pol}.tif', np.full((8, 8), .04, np.float32))
    gc.mosaic(cfg, reference)
    gc.check_grid(cfg, reference)
    gc.check_grid(cfg, reference, raw=True)
    with rasterio.open(gc.raster_path(cfg)) as src:
        np.testing.assert_allclose(src.read(1), .62 if pol == 'VV' else .4, atol=1e-5)
    assert gc.resolve_copol(cfg) == pol
    with pytest.raises(ValueError, match='No selected'):
        gc.resolve_copol(cfg, 'VV' if pol == 'HH' else 'HH')


def test_mixed_coherence_separate_polarizations(tmp_path):
    cfg = config(tmp_path, ('HH', 'VV', 'HV', 'VH'))
    inputs = cfg.groups.input_file_group
    for pol in ('HH', 'VV'):
        inputs.input_file_path.append(h5_input(tmp_path / f'g{pol}.h5', 'GCOV', (pol,)))
        inputs.gunw_file_path.append(h5_input(tmp_path / f'i{pol}.h5', 'GUNW', (pol,)))
    ref = raster(tmp_path / 'ref.tif', np.ones((8, 8), np.float32))
    cfg.groups.processing.gunw_coherence.filter.enabled = True
    cfg.groups.processing.gunw_coherence.filter.method = 'adaptive_guided'
    prefix = cfg.groups.processing.mosaic.mosaic_prefix
    for pol in ('HH', 'VV'):
        raster(tmp_path / f'{prefix}_{pol}.tif', np.full((8, 8), .04, np.float32))
    gc.mosaic(cfg, ref)
    for pol, value in [('HH', .4), ('VV', .62)]:
        gc.check_grid(cfg, ref, polarization=pol)
        gc.check_grid(cfg, ref, polarization=pol, raw=True)
        with rasterio.open(gc.raster_path(cfg, pol)) as src:
            np.testing.assert_allclose(src.read(1), value, atol=1e-5)
            assert src.tags()['GUNW_COHERENCE_GUIDE'] == f'{prefix}_{pol}.tif'


def test_default_and_auto_schema(tmp_path):
    p = tmp_path / 'algorithm.yaml'
    data = YAML(typ='safe').load(ROOT / 'src/dswx_sar/defaults/algorithm_parameter_ni.yaml')
    data['runconfig']['processing']['polarizations'] = 'auto'
    with p.open('w') as stream:
        YAML().dump(data, stream)
    cfg = load_validate_yaml(str(p), 'algorithm_parameter_ni')
    assert cfg['runconfig']['processing']['polarizations'] == 'auto'


@pytest.mark.parametrize('seed', range(5))
def test_watershed_preserves_water_and_component_stats(seed):
    label = function_from_file('src/dswx_sar/common/_refine_with_bimodality.py',
                               'label_water_with_narrow_connections_split')
    rng = np.random.default_rng(seed)
    mask = ndimage.binary_dilation(rng.random((80, 100)) > .985, iterations=3)
    mask[10:25, 10:25] = True
    mask[30:45, 30:45] = True
    mask[24:32, 24] = True
    mask[31, 24:31] = True
    n, labels, stats, _ = label(mask)
    assert np.array_equal(labels > 0, mask)
    assert n == labels.max() + 1
    np.testing.assert_array_equal(stats[:, 4], np.bincount(labels.ravel(), minlength=n))
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        np.testing.assert_array_equal(stats[i, :4], [xs.min(), ys.min(), np.ptp(xs)+1, np.ptp(ys)+1])


def test_missing_coherence_mask_never_marks_all_wetland():
    # None is ignored by the general mask-combiner; callers must guard it.
    tree = ast.parse((ROOT / 'src/dswx_sar/nisar/save_mgrs_tiles_ni.py').read_text())
    assign = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'wetland_coherence_mask' for t in n.targets)
                  and isinstance(n.value, ast.IfExp))
    fake = NS(_make_combined_mask=lambda *args: (_ for _ in ()).throw(AssertionError('should not combine')))
    assert eval(compile(ast.Expression(assign.value), '<mask-guard>', 'eval'),
                dict(coherence_removed=None, wetland=np.ones((2, 2), bool), _dswx_sar_util=fake)) is None


@pytest.mark.parametrize('pols,expected_iv', [(('VV', 'VH'), True), (('VV',), False)])
def test_workflow_no_gunw_reaches_product_writer(tmp_path, pols, expected_iv):
    import time
    cfg = config(tmp_path, pols)
    proc = cfg.groups.processing
    proc.polarization_mode = 'DV_POL' if len(pols) == 2 else 'SV_POL'
    proc.nisar_uni_mode = True
    calls = []
    names = ['mosaic_gcov_frame', 'pre_processing', 'initial_threshold',
             'fuzzy_value_computation', 'region_growing', 'masking_with_ancillary',
             'refine_with_bimodality', 'detect_inundated_vegetation', 'save_mgrs_tiles_ni']
    stages = {name: NS(run=lambda cfg, name=name: calls.append(name)) for name in names}
    fn = function_from_file('src/dswx_sar/dswx_ni.py', 'dswx_ni_workflow',
                            dict(time=time, logger=logging.getLogger(__name__), **stages))
    fn(cfg)
    assert calls[-1] == 'save_mgrs_tiles_ni'
    assert ('detect_inundated_vegetation' in calls) is expected_iv
    assert proc.polarizations == list(pols)


def test_matching_preserves_gcov_only_metadata_requirements(tmp_path, monkeypatch):
    a = h5_input(tmp_path / 'a.h5', 'GCOV')
    b = h5_input(tmp_path / 'b.h5', 'GUNW')
    real_open = gc.open_h5
    opened = []
    def count(path):
        opened.append(path)
        return real_open(path)
    monkeypatch.setattr(gc, 'open_h5', count)
    assert gc.split_inputs([a, b])[2] == {b: [a]}
    assert sorted(opened) == sorted([a, a, b, b])
    opened.clear()
    with h5py.File(a, 'a') as src:
        del src[gc.IDENTIFICATION + '/trackNumber']
    assert gc.split_inputs([a])[0] == [a]
    assert opened == [a]


def test_bcc_uses_each_matching_power_band(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    # Import only these two pure helpers rather than their GDAL-dependent
    # modules. BCC, HDF5, raster reads/writes and grid checks remain real.
    for module, name, relative in [
        ('dswx_sar.common._masking_with_ancillary', 'get_label_landcover_esa_10',
         'src/dswx_sar/common/_masking_with_ancillary.py'),
        ('dswx_sar.common._detect_inundated_vegetation', 'parse_ranges',
         'src/dswx_sar/common/_detect_inundated_vegetation.py'),
    ]:
        stub = ModuleType(module)
        setattr(stub, name, function_from_file(relative, name))
        monkeypatch.setitem(sys.modules, module, stub)
    cfg = config(tmp_path, ('HH', 'VV', 'HV', 'VH'))
    iv = cfg.groups.processing.inundated_vegetation
    iv.target_area_file_type = 'WorldCover'
    iv.target_worldcover_class = ['Herbaceous wetland']
    iv.backscatter_coherence_contrast.db_bounds = [-15, -9]
    iv.backscatter_coherence_contrast.coherence_bounds = [.5, .7]
    inputs = cfg.groups.input_file_group
    for pol in ('HH', 'VV'):
        inputs.input_file_path.append(h5_input(tmp_path / f'g{pol}.h5', 'GCOV', (pol,)))
        inputs.gunw_file_path.append(h5_input(tmp_path / f'i{pol}.h5', 'GUNW', (pol,)))
    # Only VV coherence is in range; VV power is valid only on the right.
    power = np.full((4, 8, 8), .04, np.float32)
    power[1, :, :4] = np.nan
    key = '_'.join(cfg.groups.processing.polarizations)
    ref = raster(tmp_path / f'filtered_image_{key}.tif', power)
    raster(tmp_path / 'interpolated_landcover.tif', np.full((8, 8), 90, np.uint8))
    gc.mosaic(cfg, ref)
    bcc.run(cfg)
    with rasterio.open(tmp_path / f'temp_inundated_vegetation_{key}.tif') as src:
        expected = np.zeros((8, 8), np.uint8)
        expected[:, 4:] = 2
        np.testing.assert_array_equal(src.read(1), expected)


@pytest.mark.parametrize('pol', ['HH', 'VV'])
def test_db_weight_auto_uses_available_copol(tmp_path, pol):
    cfg = config(tmp_path, (pol,), method='dual_pol_ratio')
    options = cfg.groups.processing.gunw_coherence
    options.landcover_mask.enabled = True
    options.landcover_mask.method = 'db_weight'
    inputs = cfg.groups.input_file_group
    inputs.input_file_path = [h5_input(tmp_path / 'a.h5', 'GCOV', (pol,))]
    inputs.gunw_file_path = [h5_input(tmp_path / 'b.h5', 'GUNW', (pol,))]
    water = raster(tmp_path / 'water.tif', np.ones((8, 8), np.uint8))
    lc = raster(tmp_path / 'landcover.tif', np.full((8, 8), 60, np.uint8))
    prefix = cfg.groups.processing.mosaic.mosaic_prefix
    raster(tmp_path / f'{prefix}_{pol}.tif', np.full((8, 8), .04, np.float32))
    gc.mosaic(cfg, water)
    gc.mask_water(cfg, water, lc)
    with rasterio.open(water) as src:
        assert set(np.unique(src.read(1))) <= {0, 1}
    assert (tmp_path / f'gunw_coherence_removed_water_{pol}.tif').is_file()


@pytest.mark.parametrize('mode,raw', [('first', False), ('mean', False), ('mean', True)])
def test_mosaic_first_and_mean_numerical_contract(tmp_path, mode, raw):
    cfg = config(tmp_path)
    options = cfg.groups.processing.gunw_coherence
    options.mosaic_mode = mode
    first = np.full((8, 8), .2, np.float32)
    first[:, :2] = np.nan
    p1 = raster(tmp_path / 'first.tif', first)
    p2 = raster(tmp_path / 'second.tif', np.full((8, 8), .8, np.float32))
    ref = raster(tmp_path / 'ref.tif', np.zeros((8, 8), np.float32))
    out = str(tmp_path / 'result.tif')
    gc._mosaic_native_coherence([p1, p2], ref, out, options, raw=raw)
    expected = np.full((8, 8), .2 if mode == 'first' or raw else .5, np.float32)
    expected[:, :2] = .8
    with rasterio.open(out) as src:
        np.testing.assert_allclose(src.read(1), expected, atol=1e-6)
