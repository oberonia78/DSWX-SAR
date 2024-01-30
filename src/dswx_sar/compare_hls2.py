import datetime
import os
import glob 
from datetime import datetime, timedelta
import re
import argparse

from osgeo import gdal, osr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import shapely
from shapely.geometry import LinearRing, Point, Polygon, box
import rasterio
import requests


def createParser():
    parser = argparse.ArgumentParser(
        description='Preparing the directory structure and config files')

    parser.add_argument('-i', '--inputdir', dest='input_dir', type=str, required=True,
                        help='initial input file')
    parser.add_argument('-o', '--outputdir', dest='output_dir', type=str, required=True,
                        help='initial input file')
    return parser

def extract_metadata(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        metadata = src.meta  # Extract basic metadata

        # If you need more detailed metadata:
        tags = src.tags()
        profile = src.profile

    return metadata, tags, profile


class StatisticWater:

    def __init__(self):
        pass

    def set_intensity_background(self,
                                 image_path):
        with rasterio.open(image_path) as src:
            self.intensity = src.read(1)
    
    def set_classified_data(self,
                            image_path):
        with rasterio.open(image_path) as src:
            self.cls_img = src.read(1)
            self.cls_no_data_value = src.nodatavals

    def set_reference_data(self,
                           image_path):
        with rasterio.open(image_path) as src:
            self.ref_img = src.read(1)
            self.ref_no_data_value = src.nodatavals

    def compute_accuracy(self,
                         cls_positive_values,
                         cls_negative_values,
                         cls_ignore_values,
                         ref_positive_values,
                         ref_negative_values,
                         ref_ignore_values):
        
        invalid_area = (self.cls_img == self.cls_no_data_value) | (self.ref_img == self.ref_no_data_value)
        ignore_area = np.isin(self.cls_img, cls_ignore_values) | np.isin(self.ref_img, ref_ignore_values)
        entire_ignore = np.logical_or(invalid_area, ignore_area)
        self.entire_ignore = entire_ignore

        self.ref_msk = np.isin(self.ref_img, ref_positive_values)
        self.cls_msk = np.isin(self.cls_img, cls_positive_values)

        self.ref_neg = np.isin(self.ref_img, ref_negative_values)
        self.cls_neg = np.isin(self.cls_img, cls_negative_values)

        self.positive_overlap = np.logical_and(self.ref_msk, self.cls_msk)
        self.positive_overlap = np.logical_and(self.positive_overlap, ~entire_ignore)
        self.negative_overlap = np.logical_and(self.ref_neg, self.cls_neg)
        self.negative_overlap = np.logical_and(self.negative_overlap, ~entire_ignore)

        self.false_positive = np.logical_and(self.cls_msk, self.ref_neg)
        self.false_positive = np.logical_and(self.false_positive, ~entire_ignore)

        self.false_negative = np.logical_and(self.cls_neg, self.ref_msk)
        self.false_negative = np.logical_and(self.false_negative, ~entire_ignore)

        self.true_positive_num = np.count_nonzero(self.positive_overlap)
        self.true_negative_num = np.count_nonzero(self.negative_overlap)
        self.false_negative_num = np.count_nonzero(self.false_negative)
        self.false_positive_num = np.count_nonzero(self.false_positive)

        print('True Positive', self.true_positive_num,)
        print('True Negative', self.true_negative_num)
        print('False Negative', self.false_negative_num)
        print('False Positive', self.false_positive_num)
    
        mkappa1 = 2* (self.true_positive_num*self.true_negative_num - self.false_negative_num*self.false_positive_num)
        mkappa2 = ( (self.true_positive_num + self.false_positive_num)*(self.true_negative_num + self.false_positive_num) \
                    + (self.true_positive_num + self.false_negative_num)*(self.true_negative_num + self.false_negative_num))
        print('manual-kappa', mkappa1/mkappa2)
        self.kappa = mkappa1/mkappa2
        print('manual PA', self.true_positive_num/(self.true_positive_num+self.false_positive_num))
        self.num_ref = np.count_nonzero(self.ref_msk)
        self.num_cls = np.count_nonzero(self.cls_msk)

        # reference2 = reference[np.invert(self.mask | cloud_mask)] == 100
        # classified2 = classified[np.invert(self.mask | cloud_mask)] == class_value
        # # reference3 = reference2[np.invert(cloud_mask)]
        # # classified3 = classified2[np.invert(cloud_mask)]

        # self.kappa = cohen_kappa_score(reference2, classified2)

    def create_comparison_log(self,
                            output_filename):

        correct_class = self.true_positive_num
        if self.true_positive_num + self.false_negative_num > 0:
            self.producer_acc = correct_class / (self.true_positive_num + self.false_negative_num) * 100 #self.num_ref * 100
        else:
            self.producer_acc =0
        if self.true_positive_num + self.false_positive_num > 0:
            self.user_acc = correct_class / (self.true_positive_num + self.false_positive_num) * 100 #self.num_cls * 100
        else:
            self.user_acc = 0
        outputlog = output_filename 
        with open(outputlog, 'a') as file_en:

            print('num_ref', self.num_ref)
            print('num_cls', self.num_cls)
            print(correct_class)
            print('User accuracy :', self.user_acc)
            print('Producer accuracy :', self.producer_acc)
            print('kappa :', self.kappa)
            log_str = f'num_ref, {self.num_ref} \n' \
                      f'num_cls, {self.num_cls} \n' \
                      f'num_true_positive, {self.true_positive_num} \n' \
                      f'num_true_negative, {self.true_negative_num} \n' \
                      f'num_false_positive, {self.false_positive_num} \n' \
                      f'num_false_negative, {self.false_negative_num} \n' \
                      f'user acc, {self.user_acc} \n'\
                      f'prod acc, {self.producer_acc} \n' \
                      f'kappa, {self.kappa}\n'

            file_en.write(log_str)

            file_en.write(log_str)
            if self.user_acc >= 90 and self.producer_acc>=90:
                file_en.write('excellent')
            elif (self.user_acc < 90 and self.user_acc > 80) or (self.producer_acc <90 and self.producer_acc>80):
                file_en.write('good')
            else:
                file_en.write('bad')


    def create_comparison_image(self,
                                output_filename):

        index_map = np.zeros(self.cls_msk.shape,dtype='int8')

        index_map[self.positive_overlap] = 1
        index_map[self.false_negative] = 2
        index_map[self.false_positive] = 3
        index_map[self.negative_overlap] = 4
        index_map[self.entire_ignore] = 5

        # overlapped, reference, dswx
        colors = ["blue" , "red", "green", "white", "gray"]  # use hex colors here, if desired.
        cmap = ListedColormap(colors)
        print(cmap)
        plt.subplots(1, 1, figsize=(30, 30))

        mask_layer = np.ma.masked_where(index_map == 0, index_map)
        plt.imshow(mask_layer, alpha=0.8, cmap=cmap, interpolation='nearest')

        rows, cols = self.ref_msk.shape
        yposition = int(rows / 10)
        xposition = int(cols / 50)
        plt.title('dswx s1 stat.')
        steps = 200
        plt.text(xposition, yposition,
                f"user acc {self.user_acc:.2f} %" ,fontsize=20)
        plt.text(xposition, yposition + steps * 1,
                f"producer acc {self.producer_acc:.2f} %",fontsize=20)
        plt.text(xposition, yposition + steps * 2,
                f"kappa acc {self.kappa:.2f} %",fontsize=20)
        
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

        plt.savefig(output_filename)
        plt.close()

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



def download_dswx_hls(polys, 
                      datetime_start_str, 
                      datetime_end_str, 
                      MGRS_tile,
                      download_folder,
                      download_flag=True):
    CMR_OPS = 'https://cmr.earthdata.nasa.gov/search'
    url = f'{CMR_OPS}/{"granules"}'
    boundind_box = polys.bounds
    provider = 'POCLOUD'
    parameters = {'temporal': f'{datetime_start_str},{datetime_end_str}',
                    'concept_id': 'C2617126679-POCLOUD',
                    'provider': provider,
                    'bounding_box': f'{boundind_box[1]+0.2},{boundind_box[0]+0.2},{boundind_box[3]-0.2},{boundind_box[2]-0.2}',
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
    downloaded_list = []
    num_search_data = response.headers['CMR-Hits']
    number_hls_data = 0
    if num_search_data:
        collections = response.json()['feed']['entry']
        # print('collection', collections)
        for collection in collections:
            dswx_hls_file_id = collection['producer_granule_id']
            if MGRS_tile in dswx_hls_file_id:
                print('url', f'{collection["links"][4]["href"]}')
                print('s3', f'{collection["links"][3]["href"]}')
                if download_flag:
                    dswx_hls_url = collection["links"][4]["href"]
                    dswx_hls_filename = os.path.basename(dswx_hls_url)
                    download_file = f'{download_folder}/{dswx_hls_filename}'

                    if not os.path.isfile(download_file):
                        response = requests.get(dswx_hls_url, stream=True)

                    # Check if the request was successful
                        if response.status_code == 200:
                            # Open a local file with wb (write binary) permission.
                            with open(f'{download_file}', 'wb') as file:
                                print('downloading')
                                for chunk in response.iter_content(chunk_size=128):
                                    file.write(chunk)
                else:
                    print('under dev.')
                downloaded_list.append(download_file)
                number_hls_data += 1

            else:
                print('MGRS tile id does not match')

    return number_hls_data, downloaded_list


def run(cfg):

    input_dir = cfg.input_dir
    sas_outputdir = cfg.output_dir

    os.makedirs(sas_outputdir, exist_ok=True)
    
    for water_s1_path in glob.iglob(f'{input_dir}/OPERA*BWTR*'):
        print(water_s1_path)
    
        poly = determine_polygon(water_s1_path, bbox=None)
    
        polys = check_dateline(poly)

        metadata, tags, profile = extract_metadata(water_s1_path)

        try:
            acquisition_time = tags['SENSING_START']
        except: 
            acquisition_time = tags['RTC_SENSING_START_TIME']
        pattern = re.compile(r"_([A-Z0-9]+)_\d{8}T\d{6}Z")
        # Search for the pattern in the filename
        match = pattern.search(os.path.basename(water_s1_path))
        MGRS_tile_id = match.group(1)
        formats_input = '%Y-%m-%dT%H:%M:%SZ'
        date_obj = datetime.strptime(acquisition_time, formats_input)
    
        # Compute 10 days before and 10 days after
        search_start_date = date_obj - timedelta(days=10)
        search_end_date = date_obj + timedelta(days=10)
        datetime_start_str = search_start_date.strftime(formats_input)
        datetime_end_str = search_end_date.strftime(formats_input)

        stat = StatisticWater()

        stat.set_classified_data(water_s1_path)

        for poly_cand in polys:

            num_data, dswx_hls_list = \
                download_dswx_hls(poly_cand,
                                  datetime_start_str,
                                  datetime_end_str,
                                  MGRS_tile_id,
                                  download_folder='DSWx-HLS',
                                  download_flag=True)
            if num_data:
                for dswx_hls_file in dswx_hls_list:
                    stat.set_reference_data(dswx_hls_file)
                    refname_base = os.path.splitext(os.path.basename(dswx_hls_file))[0]
                    clsname_base = os.path.splitext(os.path.basename(water_s1_path))[0]

                    # open water vs no-water (DSWX-S1)
                    # open water vs no-water (DSWx-HLS)
                    stat.compute_accuracy(
                         cls_positive_values=[1],
                         cls_negative_values=[0, 3, 4],
                         cls_ignore_values=[5, 6],
                         ref_positive_values=[1],
                         ref_negative_values=[0],
                         ref_ignore_values=[9, 253])

                    output_filename = f'{sas_outputdir}/acc_log_openwater_{clsname_base}_{refname_base}.txt'
                    stat.create_comparison_log(output_filename)
                    
                    output_filename = \
                        os.path.join(sas_outputdir,
                                     f'DSWX_S1_open_water_{clsname_base}_{refname_base}.png')
                    stat.create_comparison_image(output_filename)

                    # inundated vs no-water (DSWX-S1)
                    # partial vs no-water (DSWx-HLS)
                    stat.compute_accuracy(
                         cls_positive_values=[7],
                         cls_negative_values=[0, 3, 4],
                         cls_ignore_values=[5, 6],
                         ref_positive_values=[2],
                         ref_negative_values=[0],
                         ref_ignore_values=[9, 253])

                    output_filename = f'{sas_outputdir}/acc_log_inundated_{clsname_base}_{refname_base}.txt'
                    stat.create_comparison_log(output_filename)
                    
                    output_filename = \
                        os.path.join(sas_outputdir,
                                     f'DSWX_S1_inundated_{clsname_base}_{refname_base}.png')
                    stat.create_comparison_image(output_filename)


                    # all water vs no-water (DSWX-S1)
                    # all water vs no-water (DSWx-HLS)
                    stat.compute_accuracy(
                         cls_positive_values=[1, 7],
                         cls_negative_values=[0, 3, 4],
                         cls_ignore_values=[5, 6],
                         ref_positive_values=[1, 2],
                         ref_negative_values=[0],
                         ref_ignore_values=[9, 253])

                    output_filename = f'{sas_outputdir}/acc_log_all_water_{clsname_base}_{refname_base}.txt'
                    stat.create_comparison_log(output_filename)
                    
                    output_filename = \
                        os.path.join(sas_outputdir,
                                     f'DSWX_S1_all_water_{clsname_base}_{refname_base}.png')
                    stat.create_comparison_image(output_filename)
def main():
    parser = createParser()
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
