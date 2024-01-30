#!/usr/bin/env python

'''download RTC products from ASF'''
import ast
import argparse
import os
import glob
import requests
import shapely
import re
from datetime import datetime, timedelta

import numpy as np 
import geopandas as gpd
from osgeo import ogr, osr, gdal
from shapely.geometry import LinearRing, Polygon, box


def createParser():
    parser = argparse.ArgumentParser(
        description='Preparing the directory structure and config files')

    parser.add_argument('-b', '--bbox', dest='bbox', type=str, nargs='+', required=True,
                        help='initial input file')
    parser.add_argument('-s', '--start_time', dest='start_time', type=str, required=True,
                        help='start aquisition time ')
    parser.add_argument('-e', '--end_time', dest='end_time', type=str, required=True,
                        help='end aquisition time ')
    parser.add_argument('-m', '--mgrs_tile_db', dest='mgrs_tile_db', type=str,
                        required=True,
                        help='end aquisition time ')
    parser.add_argument('-o', '--outputdir', dest='output_dir', type=str, required=True,
                        help='initial input file')
    return parser


def read_burst_id_from_filename(filename):
    parts = filename.split('_')
    return parts[3]

def get_bounding_box_from_mgrs_tile_db(
        mgrs_tile_name,
        mgrs_db_path):
    """Get UTM bounding box from a given MGRS tile name
    from MGRS database

    Parameters
    ----------
    mgrs_tile_name: str
        Name of the MGRS tile (ex. 18LWQ)
    mgrs_db_path : str
        Path to the MGRS database file

    Returns
    -------
    minx: float
        Minimum x cooridate (UTM) for the given MGRS tile
    maxx: float
        Maximum x cooridate (UTM) for the given MGRS tile
    miny: float
        Minimum y cooridate (UTM) for the given MGRS tile
    maxy: float
        Maximum y cooridate (UTM) for the given MGRS tile
    epsg: int
        EPSG code
    """
    # Load the database from the MGRS db file.
    vector_gdf = gpd.read_file(mgrs_db_path)
    # Filter the MGRS database using the provided "mgrs_tile_name"
    filtered_gdf = vector_gdf[vector_gdf['mgrs_tile'] ==
                              mgrs_tile_name[0]]

    # Get the bounding box coordinates and
    # EPSG code from the filtered data
    minx = filtered_gdf['xmin'].values[0]
    maxx = filtered_gdf['xmax'].values[0]
    miny = filtered_gdf['ymin'].values[0]
    maxy = filtered_gdf['ymax'].values[0]
    epsg = filtered_gdf['epsg'].values[0]

    return minx, maxx, miny, maxy, epsg



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


    InSR = osr.SpatialReference()
    InSR.ImportFromEPSG(epsg)       # WGS84/Geographic
    OutSR = osr.SpatialReference()
    OutSR.ImportFromEPSG(4326)     # WGS84 UTM Zone 56 South

    Point = ogr.Geometry(ogr.wkbPoint)
    Point.AddPoint(xcoord, ycoord) # use your coordinates here
    Point.AssignSpatialReference(InSR)    # tell the point what coordinates it's in
    Point.TransformTo(OutSR)              # project it to the out spatial reference
    return Point.GetY(), Point.GetX()


def parse_date_from_filename(filename):
    # Extracts date part from the filename
    date_str = filename.split('_')[5]
    return datetime.strptime(date_str, '%Y%m%dT%H%M%S')

def calculate_overlap(polygon1, polygon2):
    # Returns the area of overlap between two polygons
    return polygon1.intersection(polygon2).area

def download_s1_asf(polys,
                    datetime_start_str,
                    datetime_end_str,
                    download_folder,
                    download_flag=True):
    CMR_OPS = 'https://cmr.earthdata.nasa.gov/search'
    url = f'{CMR_OPS}/{"granules"}'
    boundind_box = polys.bounds
    provider = 'ASF'
    parameters = {'temporal': f'{datetime_start_str}/{datetime_end_str}',
                    'concept_id': ['C1214470488-ASF','C1327985661-ASF'],
                    'provider': provider,
                    'bounding_box': f'{boundind_box[0]},{boundind_box[1]-1},{boundind_box[2]},{boundind_box[3]+1}',
                    'page_size': 800,}
    print(parameters)
    response = requests.get(url,
                            params=parameters,
                            headers={
                                'Accept': 'application/json'
                            }
                        )
    print(response, response.headers['CMR-Hits'],'found from ASF')
    downloaded_list = []
    num_search_data = response.headers['CMR-Hits']

    number_s1_data = 0
    found_s1_url_dict = {}
    if num_search_data:
        collections = response.json()['feed']['entry']
        for collection in collections:
            s1_file_id = collection['producer_granule_id']
            # print(s1_file_id)
            if download_flag:
                polygon_str = collection['polygons']
                coords = list(map(float, polygon_str[0][0].split()))
                coord_pairs = [(coords[i+1], coords[i]) for i in range(0, len(coords), 2)]
                polygon = shapely.Polygon(coord_pairs)

                for link_ind in range(0, len(collection['links'])):
                    asf_s1_url = collection["links"][link_ind]["href"]
                    iw_check = False
                    if 'IW' in asf_s1_url:
                        iw_check = True
                        number_s1_data += 1

                    if iw_check and asf_s1_url.endswith('zip') and asf_s1_url.startswith('http'):
                        s1_filename = os.path.basename(asf_s1_url)
                        download_file = f'{download_folder}/{s1_filename}'
                        check_file = glob.glob(f'{download_folder}/*/{s1_filename}')

                        if (not os.path.isfile(download_file)) and (len(check_file)==0):
                            here_download = True
                        else:
                            here_download = False
                        found_s1_url_dict[s1_filename] = (asf_s1_url, download_file, polygon, here_download)
                        downloaded_list.append(download_file)

            else:
                print('under dev.')

    # print(found_s1_url_dict)
    if len(datetime_end_str) == 8:
        input_date_format = '%Y%m%d'
    else:
        input_date_format = '%Y-%m-%dT%H:%M:%SZ'

    reference_start_date = datetime.strptime(datetime_start_str, input_date_format)
    reference_end_date = datetime.strptime(datetime_end_str, input_date_format)
    reference_date = (reference_end_date - reference_start_date)/2 + reference_start_date

    best_match = None
    best_match_date = None
    max_overlap = 0
    min_time_diff = float('inf')
    for key in found_s1_url_dict.keys():
        s1_filename = key
        # print(s1_filename)
        s1_url, s1_download_file, s1_polygon, here_download = found_s1_url_dict[key]
        s1_file_date = parse_date_from_filename(s1_filename)
        time_diff = abs((s1_file_date - reference_date).total_seconds())
        overlap = calculate_overlap(s1_polygon, polys)
        print(polys)
        print(s1_polygon)
        print(s1_filename, overlap)
        if overlap > max_overlap or (overlap == max_overlap and time_diff < min_time_diff):
            best_match = s1_filename
            max_overlap = overlap
            min_time_diff = time_diff
            best_match_date = s1_file_date
    print(best_match_date, 'is the best matched.')

    for key in found_s1_url_dict.keys():
        s1_filename = key
        s1_url, s1_download_file, s1_polygon, here_download = found_s1_url_dict[key]
        s1_file_date = parse_date_from_filename(s1_filename)
        time_diff = abs((s1_file_date - best_match_date).total_seconds())

        if time_diff < 3600 * 2:        # 2 hours

            response = requests.get(s1_url, stream=True)
            print(response)

            # Check if the request was successful
            if response.status_code == 200:
                # Open a local file with wb (write binary) permission.
                with open(f'{s1_download_file}', 'wb') as file:
                    print('downloading')
                    for chunk in response.iter_content(chunk_size=128):
                        file.write(chunk)

    return number_s1_data, downloaded_list


def run(cfg):

    sas_outputdir = cfg.output_dir
    area_indicator = cfg.bbox

    os.makedirs(sas_outputdir, exist_ok=True)

    # if bbox represent the lat/lon coordinate
    if len(area_indicator) > 1:
        poly = box(area_indicator[0], area_indicator[1],
                   area_indicator[2], area_indicator[3])

    #if input is MGRS tile ID. 
    else:
        (minx, maxx, miny, maxy, epsg) = \
            get_bounding_box_from_mgrs_tile_db(
                area_indicator, 
                cfg.mgrs_tile_db)
        lon_list = []
        lat_list = []
        for xvalue in [minx, maxx]:
            for yvalue in [miny, maxy]:
                lon, lat = get_lonlat(float(xvalue), float(yvalue), int(epsg))
                lon_list.append(lon)
                lat_list.append(lat)
        poly = box(np.min(lon_list), np.min(lat_list),
                   np.max(lon_list), np.max(lat_list))

    polys = check_dateline(poly)
    
    formats_output = '%Y-%m-%dT%H:%M:%SZ'
    formats_input = '%Y%m%d'

    search_start_date = datetime.strptime(cfg.start_time, formats_input)
    search_end_date = datetime.strptime(cfg.end_time, formats_input)

    # Compute 10 days before and 10 days after
    # search_start_date = date_obj - timedelta(days=10)
    # search_end_date = date_obj + timedelta(days=10)
    datetime_start_str = search_start_date.strftime(formats_output)
    datetime_end_str = search_end_date.strftime(formats_output)

    num_data = 0
    index = 0
    for poly_cand in polys:
        num_data, dswx_hls_list = \
            download_s1_asf(poly_cand,
                            datetime_start_str,
                            datetime_end_str,
                            download_folder=cfg.output_dir,
                            download_flag=True)
        num_data += num_data

def main():
    parser = createParser()
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
