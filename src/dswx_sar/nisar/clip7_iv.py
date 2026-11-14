"""Add IV to final dry WTR pixels using native secondary A/HH GeoTIFFs."""
from pathlib import Path
import logging
import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from osgeo import gdal


def classify(amplitude, coherence, db_bounds, coherence_bounds):
    """Inputs include a three-pixel halo. Return 0, 1, or 255 (no decision)."""
    aw = sliding_window_view(amplitude, (7, 7))
    cw = sliding_window_view(coherence, (7, 7))
    valid = sliding_window_view(
        np.isfinite(amplitude) & (amplitude > 0) & np.isfinite(coherence)
        & (coherence >= 0) & (coherence <= 1), (7, 7)).all(axis=(-1, -2))
    result = np.where(valid, 0, 255).astype('uint8')
    center_db = np.full(valid.shape, np.nan)
    center_db[valid] = 20 * np.log10(aw[..., 3, 3][valid])
    test = valid & (center_db >= db_bounds[0]) & (center_db < db_bounds[1])
    if test.any():
        samples = cw[test].reshape(-1, 49).astype('float64')
        median = np.median(samples, axis=1)
        std = samples.std(axis=1)
        clipped = np.clip(samples[:, 24], median - std, median + std)
        result[test] = (clipped >= coherence_bounds[0]) & (clipped < coherence_bounds[1])
    return result


def read_physical(band, window):
    a = band.ReadAsArray(*window).astype('float64')
    a[band.GetMaskBand().ReadAsArray(*window) == 0] = np.nan
    scale, offset = band.GetScale(), band.GetOffset()
    return a * (1 if scale is None else scale) + (0 if offset is None else offset)


def write_candidate(amp_path, coh_path, output, options):
    amp, coh = (gdal.Open(str(p)) for p in (amp_path, coh_path))
    if (not amp or not coh or not amp.GetSpatialRef() or not coh.GetSpatialRef()
            or not amp.GetSpatialRef().IsSame(coh.GetSpatialRef())
            or amp.GetGeoTransform() != coh.GetGeoTransform()
            or (amp.RasterXSize, amp.RasterYSize) != (coh.RasterXSize, coh.RasterYSize)):
        raise ValueError(f'Clip7 requires aligned native rasters: {amp_path}, {coh_path}')
    width, height = amp.RasterXSize, amp.RasterYSize
    dst = gdal.GetDriverByName('GTiff').Create(str(output), width, height, 1,
        gdal.GDT_Byte, options=['TILED=YES', 'COMPRESS=DEFLATE', 'BIGTIFF=IF_SAFER'])
    dst.SetProjection(amp.GetProjection())
    dst.SetGeoTransform(amp.GetGeoTransform())
    band = dst.GetRasterBand(1)
    band.SetNoDataValue(255)
    for y in range(0, height, 256):
        for x in range(0, width, 256):
            w, h = min(256, width-x), min(256, height-y)
            x0, y0 = max(0, x-3), max(0, y-3)
            x1, y1 = min(width, x+w+3), min(height, y+h+3)
            window = (x0, y0, x1-x0, y1-y0)
            padding = ((y0-y+3, y+h+3-y1), (x0-x+3, x+w+3-x1))
            arrays = [np.pad(read_physical(ds.GetRasterBand(1), window),
                             padding, constant_values=np.nan) for ds in (amp, coh)]
            band.WriteArray(classify(*arrays, options.db_bounds, options.coherence_bounds), x, y)
    dst.FlushCache()


def acquisition_key(path, secondary=False):
    name = Path(path).name
    prefix = re.match(r'NISAR_L2_PR_(GCOV|GUNW)_\d+_(\d+)_([AD])_(\d+)_', name)
    times = re.findall(r'(?<!\d)\d{8}T\d{6}(?!\d)', name)
    if not prefix or len(times) != (4 if secondary else 2) or prefix[1] != (
            'GUNW' if secondary else 'GCOV'):
        raise ValueError(f'Unrecognized NISAR acquisition name: {name}')
    return (*prefix.groups()[1:], *times[-2:])


def prepare(options, gcov_files, scratch):
    for limits in (options.db_bounds, options.coherence_bounds):
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f'Invalid clip7_iv bounds: {limits}')
    root = Path(options.geotiff_dir)
    if not options.geotiff_dir or not root.is_dir():
        raise ValueError('Set processing.clip7_iv.geotiff_dir to the native GeoTIFF directory.')
    keys = {acquisition_key(p) for p in gcov_files}
    inputs = {}
    for kind in ('amplitude', 'coherence'):
        for path in sorted(root.rglob(f'*_A_HH_{kind}.tif')):
            key = acquisition_key(path, secondary=kind == 'coherence')
            if key in keys:
                if (key, kind) in inputs:
                    raise ValueError(f'Ambiguous {kind} inputs for {key}')
                inputs[key, kind] = path
    outputs = {}
    for p in gcov_files:
        key = acquisition_key(p)
        if (key, 'coherence') not in inputs:
            logging.getLogger('dswx_sar').warning('No secondary clip7 coverage for %s', p)
            continue
        if (key, 'amplitude') not in inputs:
            raise ValueError(f'Missing secondary A/HH GeoTIFF pair for {p} in {root}')
        output = Path(scratch) / ('clip7_iv_' + '_'.join(key) + '.tif')
        write_candidate(inputs[key, 'amplitude'], inputs[key, 'coherence'], output, options)
        outputs[p] = output
    if not outputs:
        raise ValueError(f'No secondary A/HH acquisition matched any GCOV input in {root}')
    return outputs


class TileAddition:
    def __init__(self, paths):
        self.paths = paths
        self.add = None

    def apply(self, layer, ds):
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        if layer == 'WTR':
            yes = np.zeros(data.shape, dtype=bool)
            no = np.zeros_like(yes)
            for path in self.paths:
                src = gdal.Open(str(path))
                mem = gdal.GetDriverByName('MEM').Create('', ds.RasterXSize,
                    ds.RasterYSize, 1, gdal.GDT_Byte)
                mem.SetProjection(ds.GetProjection())
                mem.SetGeoTransform(ds.GetGeoTransform())
                target = mem.GetRasterBand(1)
                target.SetNoDataValue(255)
                target.Fill(255)
                err = gdal.ReprojectImage(src, mem, None, None, gdal.GRA_NearestNeighbour)
                if err != 0:
                    raise RuntimeError(f'Could not align clip7 candidate: {path}')
                candidate = target.ReadAsArray()
                yes |= candidate == 1
                no |= candidate == 0
            self.add = (data == 0) & yes & ~no
            data[self.add] = 3
        elif layer == 'BWTR':
            data[self.add & (data < 250)] = 1
        elif layer == 'CONF':
            eligible = self.add & (data < 250)
            wetland = (data >= 30) & (data < 40)
            data[eligible & wetland] = 35
            data[eligible & ~wetland] = 3
        band.WriteArray(data)
