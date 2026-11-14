"""Add IV to final dry WTR pixels using workflow GCOV power and raw GUNW coherence mosaics."""
from pathlib import Path
import logging
import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from osgeo import gdal


def classify(power, coherence, db_bounds, coherence_bounds):
    """Inputs include a three-pixel halo. Return 0, 1, or 255 (no decision)."""
    cw = sliding_window_view(coherence, (7, 7))
    valid = sliding_window_view(
        np.isfinite(power) & (power > 0) & np.isfinite(coherence)
        & (coherence >= 0) & (coherence <= 1), (7, 7)).all(axis=(-1, -2))
    result = np.where(valid, 0, 255).astype('uint8')
    center_db = np.full(valid.shape, np.nan)
    center_db[valid] = 10 * np.log10(power[3:-3, 3:-3][valid])
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


def write_candidate(power_path, coh_path, output, options):
    power, coh = (gdal.Open(str(p)) for p in (power_path, coh_path))
    if (not power or not coh or not power.GetSpatialRef() or not coh.GetSpatialRef()
            or not power.GetSpatialRef().IsSame(coh.GetSpatialRef())
            or power.GetGeoTransform() != coh.GetGeoTransform()
            or (power.RasterXSize, power.RasterYSize) != (coh.RasterXSize, coh.RasterYSize)):
        raise ValueError(f'Clip7 requires aligned workflow mosaics: {power_path}, {coh_path}')
    width, height = power.RasterXSize, power.RasterYSize
    dst = gdal.GetDriverByName('GTiff').Create(str(output), width, height, 1,
        gdal.GDT_Byte, options=['TILED=YES', 'COMPRESS=DEFLATE', 'BIGTIFF=IF_SAFER'])
    dst.SetProjection(power.GetProjection())
    dst.SetGeoTransform(power.GetGeoTransform())
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
                             padding, constant_values=np.nan) for ds in (power, coh)]
            band.WriteArray(classify(*arrays, options.db_bounds, options.coherence_bounds), x, y)
    dst.FlushCache()


def prepare(cfg):
    """Build one candidate on the processing grid from this run's raw mosaics."""
    from dswx_sar.nisar import gunw_coherence as gunw

    proc = cfg.groups.processing
    options = proc.clip7_iv
    for limits in (options.db_bounds, options.coherence_bounds):
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f'Invalid clip7_iv bounds: {limits}')
    if not gunw.clip7_requested(cfg):
        return None
    scratch = Path(cfg.groups.product_path_group.scratch_path)
    power = scratch / f'{proc.mosaic.mosaic_prefix}_HH.tif'
    with open(gunw._status_path(cfg)) as stream:
        sources = json.load(stream)['sources']
    if {s['path'] for s in sources} != set(cfg.groups.input_file_group.gunw_file_path):
        raise ValueError('Clip7 GUNW inputs changed; rerun mosaic_gcov_frame.')
    hh = [s for s in sources if s['polarization'] == 'HH']
    if not hh:
        logging.getLogger('dswx_sar').info('Skipping Clip7: no A/HH coherence.')
        return None
    # The notebook paired GCOV with the secondary epoch. General GUNW
    # matching also accepts reference GCOV; check dates before using a mosaic.
    current = set(cfg.groups.input_file_group.input_file_path)
    roles = set()
    for source in hh:
        with gunw.open_h5(source['path']) as dataset:
            secondary = gunw._identity(dataset, 'GUNW')['intervals'][1]
        for path in source['gcov']:
            if path not in current:
                raise ValueError('Clip7 GCOV inputs changed; rerun mosaic_gcov_frame.')
            with gunw.open_h5(path) as dataset:
                start, end = gunw._identity(dataset, 'GCOV')['intervals'][0]
            roles.add(start <= secondary[1] and secondary[0] <= end)
    if roles == {False}:
        logging.getLogger('dswx_sar').info('Skipping Clip7: reference-epoch GCOV.')
        return None
    if roles != {True}:
        raise ValueError('Mixed reference/secondary HH mosaic; run the secondary acquisition separately.')
    suffix = 'HH' if len({s['polarization'] for s in sources}) > 1 else None
    raw = Path(gunw.raw_raster_path(cfg, suffix))
    if not raw.is_file() or not power.is_file():
        raise FileNotFoundError(f'Missing Clip7 mosaics ({power}, {raw}); rerun mosaic_gcov_frame.')
    gunw.check_grid(cfg, power, raw=True, polarization=suffix)
    output = scratch / 'clip7_iv_HH.tif'
    write_candidate(power, raw, output, options)
    return output


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
