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

import geopandas as gpd
from osgeo import ogr, osr, gdal
from shapely.geometry import LinearRing, Polygon, box


def createParser():
    parser = argparse.ArgumentParser(
        description='Preparing the directory structure and config files')

    parser.add_argument('-b', '--bbox', dest='bbox', type=str, nargs=4, required=True,
                        help='initial input file')
    parser.add_argument('-s', '--start_time', dest='start_time', type=str, required=True,
                        help='start aquisition time ')
    parser.add_argument('-e', '--end_time', dest='end_time', type=str, required=True,
                        help='end aquisition time ')
    parser.add_argument('-m', '--mgrs_collection_tile_db', dest='mgrs_collection_tile_db', type=str,
                        required=True,
                        help='end aquisition time ')
    parser.add_argument('-o', '--outputdir', dest='output_dir', type=str, required=True,
                        help='initial input file')
    return parser


def read_burst_id_from_filename(filename):
    parts = filename.split('_')
    return parts[3]

def get_intersecting_mgrs_tiles_list_from_db(
        bbox,
        mgrs_collection_file,
        track_number=None):
    """Find and return a list of MGRS tiles
    that intersect a reference GeoTIFF file
    By searching in database

    Parameters
    ----------
    image_tif: str
        Path to the input GeoTIFF file.
    mgrs_collection_file : str
        Path to the MGRS tile collection.
    track_number : int, optional
        Track number (or relative orbit number) to specify
        MGRS tile collection

    Returns
    ----------
    mgrs_list: list
        List of intersecting MGRS tiles.
    most_overlapped : GeoSeries
        The record of the MGRS tile with the maximum overlap area.
    """
    # Load the raster data

    left, bottom, right, top = map(float, bbox)

    epsg_code = 4326

    antimeridian_crossing_flag = False
    if left > 0  and right < 0:
        antimeridian_crossing_flag = True
    # Create a GeoDataFrame from the raster polygon
    if antimeridian_crossing_flag:
        # Create a Polygon from the bounds
        raster_polygon_left = Polygon(
            [(left, bottom),
            (left, top),
            (180, top),
            (180, bottom)])
        raster_polygon_right = Polygon(
            [(-180, bottom),
            (-180, top),
            (right, top),
            (right, bottom)])
        raster_gdf = gpd.GeoDataFrame([1, 2],
                                      geometry=[raster_polygon_left,
                                                raster_polygon_right],
                                      crs=4326)
    else:
        # Create a Polygon from the bounds
        raster_polygon = Polygon(
            [(left, bottom),
            (left, top),
            (right, top),
            (right, bottom)])
        raster_gdf = gpd.GeoDataFrame([1],
                                      geometry=[raster_polygon],
                                      crs=4326)

    # Load the vector data
    vector_gdf = gpd.read_file(mgrs_collection_file)

    # If track number is given, then search MGRS tile collection with track number
    if track_number is not None:
        vector_gdf = vector_gdf[
            vector_gdf['relative_orbit_number'] == track_number].to_crs("EPSG:4326")
    else:
        vector_gdf = vector_gdf.to_crs("EPSG:4326")

    # Calculate the intersection
    intersection = gpd.overlay(raster_gdf,
                               vector_gdf,
                               how='intersection')

    # Add a new column with the intersection area
    intersection['Area'] = intersection.to_crs(epsg=epsg_code).geometry.area

    # Find the polygon with the maximum intersection area
    # most_overlapped = intersection.loc[intersection['Area'].idxmax()]
    top_5_overlapped = intersection.nlargest(5, 'Area')
    print(top_5_overlapped)
    burst_list_set = []
    # Now, iterate over these rows to get the 'bursts' for each
    for index, row in top_5_overlapped.iterrows():
        burst_list = ast.literal_eval(row['bursts'])
        burst_list_set.append(burst_list)
    # burst_list = ast.literal_eval(most_overlapped['bursts'])

    return burst_list_set, top_5_overlapped


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
    return Point.GetX(), Point.GetY()


def download_opera_rtc(polys,
                      datetime_start_str,
                      datetime_end_str,
                      burst_list,
                      download_folder,
                      download_flag=True):
    CMR_OPS = 'https://cmr.earthdata.nasa.gov/search'
    url = f'{CMR_OPS}/{"granules"}'
    boundind_box = polys.bounds
    provider = 'ASF'
    parameters = {'temporal': f'{datetime_start_str}/{datetime_end_str}',
                    'concept_id': 'C2777436413-ASF',
                    'provider': provider,
                    'bounding_box': f'{boundind_box[0]-3},{boundind_box[1]-1.5},{boundind_box[2]+3},{boundind_box[3]+1.5}',
                    'page_size': 800,}

    print(parameters)
    response = requests.get(url,
                            params=parameters,
                            headers={
                                'Accept': 'application/json'
                            }
                        )
    print(response.headers['CMR-Hits'],'found from ASF')
    downloaded_list = []
    num_search_data = response.headers['CMR-Hits']

    number_hls_data = 0
    if num_search_data:
        collections = response.json()['feed']['entry']
        # print('collection', collections)
        for collection in collections:
            rtc_file_id = collection['producer_granule_id']
            rtc_id = read_burst_id_from_filename(rtc_file_id)
            print(rtc_file_id, rtc_id.lower())
            rtc_id = rtc_id.replace('-' ,'_')
            if rtc_id.lower() in burst_list:
                # print('url', f'{collection["links"][4]["href"]}')
                # print('s3', f'{collection["links"][3]["href"]}')
                if download_flag:
                    
                    for link_ind in range(0, len(collection['links'])):
                        dswx_hls_url = collection["links"][link_ind]["href"]
                        polarization_check = False
                        for pol_id in ['VV', 'VH', 'mask', 'HH', 'HV']:
                            if pol_id in dswx_hls_url:
                                polarization_check = True

                        if polarization_check and dswx_hls_url.endswith('tif') and dswx_hls_url.startswith('http'):

                            dswx_hls_filename = os.path.basename(dswx_hls_url)
                            download_file = f'{download_folder}/{dswx_hls_filename}'
                            check_file = glob.glob(f'{download_folder}/*/{dswx_hls_filename}')
                            print(dswx_hls_url, check_file)

                            if (not os.path.isfile(download_file)) and (len(check_file)==0):
                                response = requests.get(dswx_hls_url, stream=True)
                                print(response)
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
                print('burst id does not match')

    return number_hls_data, downloaded_list


def run(cfg):

    sas_outputdir = cfg.output_dir
    bbox = cfg.bbox

    os.makedirs(sas_outputdir, exist_ok=True)

    poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    polys = check_dateline(poly)
    print(polys)
    formats_output = '%Y-%m-%dT%H:%M:%SZ'
    formats_input = '%Y%m%d'

    search_start_date = datetime.strptime(cfg.start_time, formats_input)
    search_end_date = datetime.strptime(cfg.end_time, formats_input)

    # Compute 10 days before and 10 days after
    # search_start_date = date_obj - timedelta(days=10)
    # search_end_date = date_obj + timedelta(days=10)
    datetime_start_str = search_start_date.strftime(formats_output)
    datetime_end_str = search_end_date.strftime(formats_output)

    burst_list_set, mgrs_col_id = get_intersecting_mgrs_tiles_list_from_db(
        bbox,
        cfg.mgrs_collection_tile_db
    )

    num_data = 0
    index = 0
    while num_data == 0:
        for poly_cand in polys:
            print(index)
            burst_list = burst_list_set[index]
            print(burst_list)
            num_data, dswx_hls_list = \
                download_opera_rtc(poly_cand,
                                    datetime_start_str,
                                    datetime_end_str,
                                    burst_list,
                                    download_folder=cfg.output_dir,
                                    download_flag=True)
            num_data += num_data
        index += 1

def main():
    parser = createParser()
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
