#!/usr/bin/env python

import argparse
import os
import glob
import re
import shutil
import zipfile
import numpy as np
from lxml import etree
from datetime import datetime

from batch_dswx_util import create_runconfig


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


def sort_rtc(input_dir):

    # The directory where your files are located (current directory in this example)
    file_directory = input_dir  # Use '.' for current directory or replace with your directory path
    # Regex pattern to extract the unique pattern from the filenames (Txxx-xxxxxx here refers to the pattern in your example)
    pattern_regex = r'(T\d{3}-\d{6})'
    pattern_regex = r'(T\d{3}-\d{6}-IW[1-3])'

    # List all files in the specified directory
    # all_files = os.listdir(file_directory)
    all_files = glob.glob(f'{input_dir}/*.tif')

    # Process files and sort them into directories
    for filename in all_files:
        filename2 = filename.split('/')[-1]
        # Search for the pattern in the filename
        match = re.search(pattern_regex, filename2)
        if match:
            # Extract the directory name from the pattern
            directory_name = match.group(1)
            full_dir_name = f'{file_directory}/{directory_name}'
            # Create a directory for this pattern if it doesn't already exist
            if not os.path.exists(full_dir_name):
                os.makedirs(full_dir_name)
            # Move the file into the new directory
            shutil.move(os.path.join(file_directory, filename2), os.path.join(full_dir_name, filename2))
            print(f"Moved {filename} to {full_dir_name}/")




def parse_csv_file(file_path):
    import csv

    x_coord_set = []
    y_coord_set = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Each row is a list of strings
            # You can convert each string to a float or leave it as is, depending on your requirement
            float_values = [float(value) for value in row]
            x_coord_set.append(float_values[-2])
            y_coord_set.append(float_values[-1])
    return x_coord_set, y_coord_set


def read_kmz_file(kmz_file_path):
    extracted_folder_path = 'test_kml'

    with zipfile.ZipFile(kmz_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)

    kml_file_path = os.path.join(extracted_folder_path, 'doc.kml') 

    # Parse the KML file
    with open(kml_file_path) as file:
        kml = etree.parse(file)


    # Parse the KML file
    with open(kml_file_path) as file:
        kml = etree.parse(file)

    # Iterate through Placemarks
    kmz_dict = {}
    for placemark in kml.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
        name = placemark.find('.//{http://www.opengis.net/kml/2.2}name')
        if name is not None:
            print('Title:', name.text.replace(' ', '-'))
            title = name.text.replace(' ', '-')

        # Extract coordinates for the polygon
        polygon = placemark.find('.//{http://www.opengis.net/kml/2.2}Polygon')
        if polygon is not None:
            # Extract coordinates from the polygon
            coordinates = polygon.find('.//{http://www.opengis.net/kml/2.2}coordinates')
            x_set = []
            y_set = []
            if coordinates is not None:
                # coordinates.text will contain the coordinates as a string
                coords_text = coordinates.text.strip()
                # Splitting the text into individual coordinates
                coords_list = coords_text.split()
                # Each part of coords_list is a string of "longitude,latitude,altitude"
                # Process as needed, for example:
                for coord_str in coords_list:
                    lon, lat, _ = map(float, coord_str.split(','))
                    x_set.append(lon)
                    y_set.append(lat)
        kmz_dict[title] = [np.min(x_set), np.min(y_set), np.max(x_set), np.max(y_set)]
    return kmz_dict


def run():

    mgrs_tile_collection_db = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-gamma/sample_data/input_dir/ancillary_data/MGRS_tile_collection_v0.2.sqlite'
    sample_yaml = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-calval/sample_products_calval/dswx_sample.yaml'
    output_dir = 'batch_dswx'

    # read coordinates
    kmz_file_path = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-calval/sample_products_calval/DSWx_S1_Sample_Sites.kmz'
    coord_dict = read_kmz_file(kmz_file_path)

    now = datetime.now()

    current_time_str = now.strftime("%Y%m%dT%H%M%S")
    dswx_script_shell = f'dswx_sar_batch_kmz_{current_time_str}'

    # downlaod rtc
    for area_name, bbox in coord_dict.items():
        # if area_name in ['Jones_Center', 'franklinbluffs', 'carpinteria', 'bonanza']:
        # if area_name in [ 'carpinteria']:
        if area_name in [ 'W-2', 'Sycan-Marsh']:            
        # if area_name in ['pad']:
            output_batch_dir = f'{output_dir}-{area_name}'
            rtc_dir = f'{output_batch_dir}/input_dir/rtc'
            x_start = bbox[0] - 1.5
            x_end = bbox[2] + 1.5
            y_start = bbox[1] - 1.5
            y_end = bbox[3] + 1.5
            cmdline = f'python3 /mnt/aurora-r0/jungkyo/tool/DSWX-SAR_series/DSWX-SAR/src/dswx_sar/download_rtc_asf.py '\
                    f'-b {x_start} {y_start} {x_end} {y_end} -s 20231101 -e 20231113 '\
                    f'-m {mgrs_tile_collection_db}  -o {rtc_dir}'

            print(cmdline)
            os.system(cmdline)

            tif_file = glob.glob(f'{rtc_dir}/*.tif')
            if len(tif_file) > 0:
                sort_rtc(rtc_dir)

            # # download hand
            os.makedirs(f'{output_batch_dir}/input_dir/anc_data' ,exist_ok=True)
            hand_cmdline = f'python3 /mnt/aurora-r0/jungkyo/tool/DSWX-SAR_series/DSWX-SAR/src/dswx_sar/stage_hand.py '\
                        f'-b {x_start-1} {y_start-1} {x_end+1} {y_end+1}  -o {output_batch_dir}/input_dir/anc_data/hand.vrt'
            os.system(hand_cmdline)

            # create runconfig
            output_check = glob.glob(f'{output_batch_dir}/output_dir/*.tif')
            if len(output_check) == 0:
                create_runconfig(output_batch_dir, sample_yaml, dswx_script_shell)

    # run dswx_s1
    os.system(f'sh {dswx_script_shell}')

def main():
    # parser = createParser()
    # args = parser.parse_args()

    run()


if __name__ == '__main__':
    main()
