#!/usr/bin/env python

import argparse
import os
import glob
import re
import shutil
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
    parser.add_argument('--overwrite', dest='overwrite', type=bool, default=False,
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


def run():

    mgrs_tile_collection_db = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-gamma/sample_data/input_dir/ancillary_data/MGRS_tile_collection_v0.2.sqlite'
    sample_yaml = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-calval/sample_products_calval/dswx_sample.yaml'
    output_dir = 'batch_dswx'

    # read coordinates
    csv_file_path = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-calval/sample_products_calval/OPERA_FEOW_Cntrd_Calibration_Set1.csv'
    x_coord_set, y_coord_set = parse_csv_file(csv_file_path)

    now = datetime.now()
    current_time_str = now.strftime("%Y%m%dT%H%M%S")
    dswx_script_shell = f'dswx_cmd_script_{current_time_str}'

    # downlaod rtc
    for batch_ind, (x_center, y_center) in enumerate(zip(x_coord_set, y_coord_set)):
        print(batch_ind)
        if batch_ind in [1]:
            output_batch_dir = f'{output_dir}-{batch_ind}'
            output_check = glob.glob(f'{output_batch_dir}/output_dir/*.tif')
            if len(output_check) == 0:

                rtc_dir = f'{output_batch_dir}/input_dir/rtc'
                x_start = x_center - 1.5
                x_end = x_center + 1.5
                y_start = y_center - 1.5
                y_end = y_center + 1.5
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
                create_runconfig(output_batch_dir, sample_yaml, dswx_script_shell)

    # run dswx_s1
    # os.system(f'sh {dswx_script_shell}')

def main():
    # parser = createParser()
    # args = parser.parse_args()

    run()


if __name__ == '__main__':
    main()
