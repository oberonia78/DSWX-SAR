import datetime
import logging
import os
import glob 
from datetime import datetime, timedelta

from pathlib import Path
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from pyproj import Transformer
from sklearn.metrics import cohen_kappa_score
from shapely.geometry import LinearRing, Point, Polygon, box
import rasterio

from dswx_sar import (dswx_sar_util,
                      generate_log)
from dswx_sar.dswx_runconfig import _get_parser, RunConfig


logger = logging.getLogger('dswx_s1')

def extract_metadata(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        metadata = src.meta  # Extract basic metadata

        # If you need more detailed metadata:
        tags = src.tags()
        profile = src.profile

    return metadata, tags, profile

def get_projection_proj4(projection):
    srs = osr.SpatialReference()
    if projection.upper() == 'WGS84':
        srs.SetWellKnownGeogCS(projection)
    else:
        srs.ImportFromProj4(projection)
    projection = srs.ExportToProj4()
    projection = projection.strip()
    return projection


def crop_image(input_file, bbox,
              resample_algorithm='nearest',
              relocated_file=None):

    print('intput', input_file)
    print('output', relocated_file)
    x0, xf, y0, yf = bbox

    if relocated_file is None:
        relocated_file = tempfile.NamedTemporaryFile(
                    dir='.', suffix='.tif').name

    gdal.Warp(relocated_file, input_file, format='GTiff',
              outputBounds=[x0, yf, xf, y0], multithread=True,
              resampleAlg=resample_algorithm,
              errorThreshold=0)

    ds = gdal.Open(relocated_file)
    relocated_array = ds.ReadAsArray()

    return relocated_array


def relocate2(input_file, geotransform, projection,
              length, width,
              resample_algorithm='nearest',
              relocated_file=None):

    print('intput', input_file)
    print('output', relocated_file)
    dy = geotransform[5]
    dx = geotransform[1]
    y0 = geotransform[3]
    x0 = geotransform[0]

    xf = x0 + width * dx
    yf = y0 + length * dy
    print(x0, xf, y0, yf)
    dstSRS = get_projection_proj4(projection)
    print(dstSRS)
    if relocated_file is None:
        relocated_file = tempfile.NamedTemporaryFile(
                    dir='.', suffix='.tif').name

    opt = gdal.WarpOptions(dstSRS=dstSRS,
                     xRes=dx,
                     yRes=abs(dy),
                     outputBounds=[x0, yf, xf, y0],
                     resampleAlg=resample_algorithm,
                     format='GTiff')
    intput_gdal = gdal.Open(input_file)
    print(relocated_file, type(relocated_file))
    print(input_file, type(input_file))

    # output_gdal = gdal.Open(relocated_file.name)
    outdir_path = os.path.dirname(relocated_file)
    print('output dir', outdir_path)
    os.makedirs(outdir_path, exist_ok=True)
    print(f'gdalwarp {relocated_file.resolve()}')
    print(f'gdalwarp {str(relocated_file.resolve())}')
    ds2 = gdal.Warp(str(relocated_file.resolve()), intput_gdal, options=opt)
    ds2 = None
    input_gdal = None
    output_gdal = None
    ds = gdal.Open(str(relocated_file.resolve()))
    relocated_array = ds.ReadAsArray()
    ds = None
    return relocated_array


class StatisticWater:

    def __init__(self, int_str):

        band_set = dswx_sar_util.read_geotiff(int_str)
        self.int_str = int_str
        if len(band_set.shape) == 3:
            band_avg = np.squeeze(np.nanmean(band_set, axis=0))

        elif len(band_set.shape) == 2:
            band_avg = band_set

        mask_zero = band_avg <= 0
        mask_nan = np.isnan(band_avg)

        self.mask = np.logical_or(mask_zero, mask_nan)
        self.band_avg = band_avg


    def compute_accuracy(self,
                         classified,
                         class_value,
                         reference,
                         outputdir='.'):

        reference[self.mask] = 0
        classified[self.mask] = 0

        cloud_mask = (reference == 255) | ((reference>0) & (reference<100)) | np.isnan(reference) | np.isnan(classified) | (classified == 255)
        reference[cloud_mask] = 0
        classified[cloud_mask] = 0
        self.ref_msk = reference == 100
        self.cloud = cloud_mask
        self.cls_msk = classified == class_value
        self.overlap = np.logical_and(self.ref_msk, self.cls_msk)
        self.True_positive = np.count_nonzero(self.overlap)
        self.True_negative = np.count_nonzero(np.logical_and(np.invert(self.ref_msk), np.invert(self.cls_msk)))
        self.False_positive = np.count_nonzero(np.logical_and((self.ref_msk), np.invert(self.cls_msk)))
        self.False_negative = np.count_nonzero(np.logical_and( np.invert(self.ref_msk),(self.cls_msk)))
        mkappa1 = 2* (self.True_positive*self.True_negative - self.False_negative*self.False_positive)
        mkappa2 = ( (self.True_positive + self.False_positive)*(self.True_negative + self.False_positive) + (self.True_positive + self.False_negative)*(self.True_negative + self.False_negative))
        print('manual-kappa', mkappa1/mkappa2)
        print('manual PA', self.True_positive/(self.True_positive+self.False_positive))
        num_ref = np.count_nonzero(self.ref_msk)
        num_cls = np.count_nonzero(self.cls_msk)
        print(num_ref)

        reference2 = reference[np.invert(self.mask | cloud_mask)] == 100
        classified2 = classified[np.invert(self.mask | cloud_mask)] == class_value
        # reference3 = reference2[np.invert(cloud_mask)]
        # classified3 = classified2[np.invert(cloud_mask)]

        self.kappa = cohen_kappa_score(reference2, classified2)
        _, ax = plt.subplots(1,1,figsize=(30, 30))
        ax.imshow(10*np.log10(self.band_avg),
                cmap = plt.get_cmap('gray'),
                vmin=-25,
                vmax=-5)
        colors = ["red"]  # use hex colors here, if desired.
        cmap = ListedColormap(colors)
        plt.imshow(self.ref_msk, alpha=0.8, interpolation='nearest')
        plt.savefig(os.path.join(f'{outputdir}', 'DSWX_S1_GLAD') )
        plt.close()

        correct_class = np.count_nonzero(self.overlap)
        self.producer_acc = correct_class / num_ref * 100
        self.user_acc = correct_class /num_cls * 100
        outputlog = f'{outputdir}/acc_log_glad'
        with open(outputlog, 'a') as file_en:

            print('num_ref', num_ref)
            print('num_cls', num_cls)
            print(correct_class)
            print('User accuracy :', self.user_acc)
            print('Producer accuracy :', self.producer_acc)
            print('kappa :', self.kappa)
            log_str = f'num_ref, {num_ref} \n num_cls, {num_cls} \n num_true_positive, {correct_class} \n user acc, {self.user_acc} \n prod acc, {self.producer_acc}\n'

            file_en.write(log_str)
            log_str = f'kappa value, {self.kappa}\n'
            file_en.write(log_str)
            if self.user_acc >= 90 and self.producer_acc>=90:
                file_en.write('\n excellent')
            elif (self.user_acc < 90 and self.user_acc > 80) or (self.producer_acc <90 and self.producer_acc>80):
                file_en.write('\n good')
            else:
                file_en.write('\n bad')



    def create_image(self, outputdir):

        index_map = np.zeros(self.cls_msk.shape,dtype='int8')

        only_ref = np.logical_and(self.ref_msk, np.invert(self.cls_msk))
        only_cls = np.logical_and(np.invert(self.ref_msk), self.cls_msk)

        index_map[self.overlap] = 1
        index_map[only_ref] = 2
        index_map[only_cls] = 3
        index_map[self.cloud] = 5

        # overlapped, reference, dswx
        colors = ["blue" , "red", "green", "k", "yellow"]  # use hex colors here, if desired.
        cmap = ListedColormap(colors)
        print(cmap)
        fig, ax = plt.subplots(1,1,figsize=(30, 30))
        im = ax.imshow(10*np.log10(self.band_avg),
                      cmap = plt.get_cmap('gray'),
                      vmin=-25,
                      vmax=-5)

        mask_layer = np.ma.masked_where(index_map == 0, index_map)
        plt.imshow(mask_layer, alpha=0.8, cmap=cmap, interpolation='nearest')
        # plt.imshow(self.overlap, alpha=0.9, cmap =blue_cmap)
        # plt.imshow(only_ref, alpha=0.9, cmap = plt.get_cmap('Reds'))
        # ax.imshow(only_cls, alpha=0.9, cmap = plt.get_cmap('Greens'))

        rows, cols = self.ref_msk.shape
        yposition = int(rows / 10)
        xposition = int(cols / 50)
        plt.title('dswx s1 stat.')
        steps = 200
        plt.text(xposition, yposition,
                f"user acc {self.user_acc:.2f} %" ,fontsize=20)
        plt.text(xposition, yposition + steps * 1,
                f"producer acc {self.producer_acc:.2f} %",fontsize=20)

        plt.text(cols - 10*xposition, yposition,
                f"DSWX and reference ", fontsize=20,
                 backgroundcolor='blue',
                 weight='bold',
                 color='white')
        plt.text(cols - 10*xposition, yposition + steps * 1,
                f"DSWX only " ,fontsize=20, backgroundcolor='green', weight='bold',
                 color='white')
        plt.text(cols - 10* xposition, yposition + steps * 2,
                f"Reference only",fontsize=20, backgroundcolor='red', weight='bold',
                 color='white')

        plt.savefig(os.path.join(outputdir, 'DSWX_S1_GLAD_stat2') )
        plt.close()
        water_meta = dswx_sar_util.get_meta_from_tif(self.int_str)

        dswx_sar_util.save_dswx_product(index_map,
                    os.path.join(outputdir, 'DSWX_S1_GLAD_stat2.tif'),
                    geotransform=water_meta['geotransform'],
                    projection=water_meta['projection'],
                    description='Water classification (WTR)',
                    scratch_dir=outputdir)#,
                    # no_data=self.cls_msk)


def check_dateline(poly):
    """Split `poly` if it crosses the dateline.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Input polygon.

    Returns
    -------
    polys : list of shapely.geometry.Polygon
         A list containing: the input polygon if it didn't cross
        the dateline, or two polygons otherwise (one on either
        side of the dateline).
    """

    xmin, _, xmax, _ = poly.bounds
    # Check dateline crossing
    if (xmax - xmin) > 180.0:
        dateline = shapely.wkt.loads('LINESTRING( 180.0 -90.0, 180.0 90.0)')

        # build new polygon with all longitudes between 0 and 360
        x, y = poly.exterior.coords.xy
        new_x = (k + (k <= 0.) * 360 for k in x)
        new_ring = LinearRing(zip(new_x, y))

        # Split input polygon
        # (https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango_)
        merged_lines = shapely.ops.linemerge([dateline, new_ring])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomp = shapely.ops.polygonize(border_lines)

        polys = list(decomp)

        # The Copernicus DEM used for NISAR processing has a longitude
        # range [-180, +180]. The current version of gdal.Translate
        # does not allow to perform dateline wrapping. Therefore, coordinates
        # above 180 need to be wrapped down to -180 to match the Copernicus
        # DEM longitude range
        for polygon_count in range(2):
            x, y = polys[polygon_count].exterior.coords.xy
            if not any([k > 180 for k in x]):
                continue

            # Otherwise, wrap longitude values down to 360 deg
            x_wrapped_minus_360 = np.asarray(x) - 360
            polys[polygon_count] = Polygon(zip(x_wrapped_minus_360, y))

        assert (len(polys) == 2)
    else:
        # If dateline is not crossed, treat input poly as list
        polys = [poly]

    return polys

def latlon_to_utm(lat, lon, output_espg, input_epsg=4326):
    # Determine the UTM zone number and hemisphere
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    # utm_crs = f"EPSG:326{utm_zone}" if hemisphere == "north" else f"EPSG:327{utm_zone}"
    utm_crs = f"EPSG:{output_espg}" if hemisphere == "north" else f"EPSG:327{utm_zone}"

    # Initialize the transformer
    transformer = Transformer.from_crs(input_epsg, utm_crs)

    # Convert the latitude and longitude to UTM coordinates
    easting, northing = transformer.transform(lat, lon)

    return utm_zone, hemisphere, easting, northing


def read_tif_latlon(intput_tif_str):
    #  Initialize the Image Size
    ds = gdal.Open(intput_tif_str)
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    epsg_input = proj.GetAttrValue('AUTHORITY',1)

    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]

    ds = None
    del ds  # close the dataset (Python object and pointers)
    if epsg_input != 4326:
        xcoords = [minx, maxx, maxx, minx]
        ycoords = [miny, miny, maxy, maxy]

        poly_wkt = []  # Initialize as a list

        for xcoord, ycoord in zip(xcoords, ycoords):
            lon, lat = get_lonlat(xcoord, ycoord, int(epsg_input))
            poly_wkt.append((lon, lat))

        poly = Polygon(poly_wkt)
    else:
        poly = box(minx, miny, maxx, maxy)

    return poly

def determine_polygon(intput_tif_str, bbox=None):
    """Determine bounding polygon using RSLC radar grid/orbit
    or user-defined bounding box

    Parameters:
    ----------
    ref_slc: str
        Filepath to reference RSLC product
    bbox: list, float
        Bounding box with lat/lon coordinates (decimal degrees)
        in the form of [West, South, East, North]

    Returns:
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to RSLC perimeter
        or bbox shape on the ground
    """
    if bbox is not None:
        print('Determine polygon from bounding box')
        poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    else:
        print('Determine polygon from Geotiff')
        poly = read_tif_latlon(intput_tif_str)

    return poly


def get_lonlat(xcoord, ycoord, epsg):

    from osgeo import ogr
    from osgeo import osr

    InSR = osr.SpatialReference()
    InSR.ImportFromEPSG(epsg)       # WGS84/Geographic
    OutSR = osr.SpatialReference()
    OutSR.ImportFromEPSG(4326)     # WGS84 UTM Zone 56 South

    Point = ogr.Geometry(ogr.wkbPoint)
    Point.AddPoint(xcoord, ycoord) # use your coordinates here
    Point.AssignSpatialReference(InSR)    # tell the point what coordinates it's in
    Point.TransformTo(OutSR)              # project it to the out spatial reference
    return Point.GetX(), Point.GetY()

import requests
from pprint import pprint

def download_dswx_hls(polys, datetime_start_str, datetime_end_str):
    CMR_OPS = 'https://cmr.earthdata.nasa.gov/search'
    url = f'{CMR_OPS}/{"granules"}'
    boundind_box = polys.bounds
    provider = 'POCLOUD'
    parameters = {'temporal': f'{datetime_start_str},{datetime_end_str}',
                    'concept_id': 'C2617126679-POCLOUD',
                    'provider': provider,
                    'bounding_box': f'{boundind_box[1]},{boundind_box[0]},{boundind_box[3]},{boundind_box[2]}',
                    'page_size': 200,}
    print(parameters)
    response = requests.get(url,
                            params=parameters,
                            headers={
                                'Accept': 'application/json'
                            }
                        )
    print(response.status_code)
    print(response.headers['CMR-Hits'])
    number_hls_data = response.headers['CMR-Hits']
    if number_hls_data:
        collections = response.json()['feed']['entry']

        for collection in collections:
            print(f'{collection["links"][4]["href"]}')
    return number_hls_data
# def run(args):

#     logger.info(f'start computing statistics')
#     t_all = time.time()

#     outputdir = args.scratch_dir
#     pol_list = args.pols
#     pol_str = '_'.join(pol_list)
#     filename = args.input_list[0]


#     interp_wbd_str = os.path.join(outputdir, 'interpolated_wbd')
#     interp_wbd = dswx_sar_util.read_geotiff(interp_wbd_str) /100

#     stat = StatisticWater(args)

#     if args.dswx_workflow == 'opera_dswx_s1':
#         water_map_tif_str = os.path.join(outputdir, 'bimodality_output_binary_{}.tif'.format(pol_str))
#         water_map = dswx_sar_util.read_geotiff(water_map_tif_str)

#         stat.compute_accuracy(classified=water_map,
#                             class_value=1,
#                             reference=interp_wbd,
#                             reference_threshold=0.8,
#                             mask=interp_wbd>1)
#         stat.create_comparision_image('bimodal_step')

#         water_map_tif_str = os.path.join(outputdir, 'refine_landcover_binary_{}.tif'.format(pol_str))
#         water_map = dswx_sar_util.read_geotiff(water_map_tif_str)

#         stat.compute_accuracy(classified=water_map,
#                             class_value=1,
#                             reference=interp_wbd,
#                             reference_threshold=0.8,
#                             mask=interp_wbd>1)
#         stat.create_comparision_image('landcover_step')

#     water_map_tif_str = os.path.join(outputdir, 'region_growing_output_binary_{}.tif'.format(pol_str))
#     water_map = dswx_sar_util.read_geotiff(water_map_tif_str)

#     stat.compute_accuracy(classified=water_map,
#                          class_value=1,
#                          reference=interp_wbd,
#                          reference_threshold=0.8,
#                          mask=interp_wbd>1 )
#     stat.create_comparision_image('region_growing_step')

#     t_time_end = time.time()

#     t_all_elapsed = t_time_end - t_all
#     logger.info(f"successfully ran computing statistics in {t_all_elapsed:.3f} seconds")


def run(cfg):

    input_list = cfg.groups.input_file_group.input_file_path
    outputdir = cfg.groups.product_path_group.scratch_path
    pol_list = cfg.groups.processing.polarizations

    os.makedirs(outputdir, exist_ok=True)
    
    water_map_tif_str = \
        os.path.join(outputdir, 'full_water_binary_BWTR_set.tif')
    
    water_map = dswx_sar_util.read_geotiff(water_map_tif_str)
    rows, cols = water_map.shape

    poly = determine_polygon(water_map_tif_str, bbox=None)
    print(poly)
    reftif = gdal.Open(water_map_tif_str)
    proj = osr.SpatialReference(wkt=reftif.GetProjection())
    epsg_output = proj.GetAttrValue('AUTHORITY',1)
    del reftif
    
    polys = check_dateline(poly)
    input_dir = input_list[0]
    pol = pol_list[0]

    rtc_path_input = glob.glob(f'{input_dir}/*_{pol}.tif')[0]
    metadata, tags, profile = extract_metadata(rtc_path_input)

    acquisition_time = tags['ZERO_DOPPLER_START_TIME']

    formats_input = '%Y-%m-%dT%H:%M:%S.%fZ'
    formats_output = '%Y-%m-%dT%H:%M:%SZ'
    date_obj = datetime.strptime(acquisition_time, formats_input)
    
    # Compute 10 days before and 10 days after
    search_start_date = date_obj - timedelta(days=10)
    search_end_date = date_obj + timedelta(days=10)
    datetime_start_str = search_start_date.strftime(formats_output)
    datetime_end_str = search_end_date.strftime(formats_output)
    for poly_cand in polys:

        num_data = \
            download_dswx_hls(poly_cand, datetime_start_str, datetime_end_str)


    if len(area_coord_list) >=2:
        cmdline = f'gdalbuildvrt {outputdir}/glad_img.vrt {outputdir}/glad*tif'
        os.system(cmdline)
        output_glad = f'{outputdir}/glad_img.vrt'

    # hls_map = dswx_sar_util.read_geotiff(output_glad)
    # if len(hls_map.shape) == 2:
    #     hls_rows, hls_cols = hls_map.shape
    # elif len(hls_map.shape) == 3:
    #     hls_band, hls_rows, hls_cols = hls_map.shape
    # print('hls size:', hls_map.shape)
    cropping_flag = False
    if args.lat_arr and args.lon_arr:
        cropping_flag = True
        east_set = []
        north_set = []
        for i in range(len(args.lat_arr)):
            _, _, east_arr, north_arr = latlon_to_utm(
                                        args.lat_arr[i],
                                        args.lon_arr[i],
                                        output_espg=epsg_output,
                                        input_epsg=4326)
            east_set.append(east_arr)
            north_set.append(north_arr)
    print(east_set)
    print(north_set)

    print('opera', rows, cols)

    layer_gdal_dataset = gdal.Open(water_map_tif_str)
    hls_gdal_dataset = gdal.Open(output_glad)

    water_interpolated_path = Path(os.path.join(outputdir, 'resample_water.tif'))
    print('water intermediate, ', water_interpolated_path)
    if not water_interpolated_path.is_file():

        geotransform = layer_gdal_dataset.GetGeoTransform()
        projection = layer_gdal_dataset.GetProjection()

        relocate2(input_file=output_glad,
                 geotransform=geotransform,
                 projection=projection,
                 length=rows,
                 width=cols,
                 resample_algorithm='near',
                 relocated_file=water_interpolated_path)
    interp_dswx_glad = dswx_sar_util.read_geotiff(str(water_interpolated_path.absolute()))

    if cropping_flag:

        crop_image(water_map_tif_str, bbox=[np.min(east_set), np.max(east_set), np.min(north_set), np.max(north_set)],
                    relocated_file=f'{water_map_tif_str}_sub.tif')
        water_map = dswx_sar_util.read_geotiff(f'{water_map_tif_str}_sub.tif')
        crop_image(str(water_interpolated_path.absolute()), bbox=[np.min(east_set), np.max(east_set), np.min(north_set), np.max(north_set)],
                    relocated_file=f'{str(water_interpolated_path.absolute())}_sub.tif')
        interp_dswx_glad = dswx_sar_util.read_geotiff(f'{str(water_interpolated_path.absolute())}_sub.tif')
        crop_image(args.int_tif, bbox=[np.min(east_set), np.max(east_set), np.min(north_set), np.max(north_set)],
                    relocated_file=f'{args.int_tif}_sub.tif')
        args.int_tif = f'{str(water_interpolated_path.absolute())}_sub.tif'



    stat = StatisticWater(str(args.int_tif))

    outputlog = f'{outputdir}/acc_log'
    print('log_files', outputlog)
    with open(outputlog, 'w') as file_en:
        file_en.write(f'input_dswx {args.dswx_s1_tif} \n')
        file_en.write(f'input_glad {output_glad} \n')

    stat.compute_accuracy(classified=water_map,
                         class_value=1,
                         reference=interp_dswx_glad,
                         outputdir=outputdir)
    stat.create_image(outputdir)
    layer_gdal_dataset = None
    hls_gdal_dataset = None


def main():

    parser = _get_parser()

    args = parser.parse_args()

    generate_log.configure_log_file(args.log_file)

    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)

    run(cfg)


if __name__ == '__main__':
    main()
