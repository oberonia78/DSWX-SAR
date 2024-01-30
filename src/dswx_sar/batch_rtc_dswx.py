#!/usr/bin/env python
import ast
import argparse
import os
import glob
import re
import shutil

import numpy as np
from datetime import datetime, timedelta
from download_s1_mgrs import get_bounding_box_from_mgrs_tile_db, get_lonlat
import geopandas as gpd
import sqlite3


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


def create_runconfig(main_input_dir,
                     sample_yaml,
                     burst_list,
                     dswx_cmd_file):

    rtc_path = os.path.join(f'{main_input_dir}/rtc_outputdir', 't*')
    rtc_list = glob.glob(rtc_path)
    # print(rtc_list)
    input_str = ''
    print(type(burst_list))

    for filename in rtc_list:
        rtc_dir_name = filename.split('/')[-1]
        # print(rtc_dir_name)
        # print(type(rtc_dir_name))
        if rtc_dir_name == 't139_297145_iw3':
            print(rtc_dir_name)
            print(burst_list)
            print(rtc_dir_name in burst_list)
        if rtc_dir_name in burst_list:
            # input_str += f'        - input_dir/rtc/{rtc_dir_name}\n'
            input_str += f'        - {filename}\n'

    print(input_str)
    with open(dswx_cmd_file, 'a') as file_en:
    
        new_yaml = f"{main_input_dir}/dswx_{main_input_dir}.yaml"

        with open(sample_yaml, 'r') as file :
            filedata = file.read()
        outputdir = f'{main_input_dir}'

        # Replace the target string:
        filedata = filedata.replace('test_input_here', input_str)
        filedata = filedata.replace('testhere', outputdir)
        # filedata = filedata.replace('output.h5', f"RTC_{datestr}.h5")

        # Write the file out again

        with open(new_yaml, 'w') as file:
            file.write(filedata)

        cmdline = f"python3  /mnt/aurora-r0/jungkyo/tool/DSWX-SAR_series/DSWX-SAR/src/dswx_sar/dswx_s1.py  {new_yaml}  --debug_mode \n"
        file_en.write(cmdline)


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


def parse_csv_file_mgrs(file_path):
    import csv
    site_name = []
    mgrs_list = []
    acq_time_list = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Each row is a list of strings
            # You can convert each string to a float or leave it as is, depending on your requirement
            values = [value for value in row]
            site_name.append(values[0])
            filename = values[2]
            split_filename = filename.split('_')
            mgrstile = split_filename[3]
            mgrs_list.append(mgrstile[1:])
            acq_time_list.append(split_filename[4])
    return site_name, mgrs_list, acq_time_list


def create_rtc_runconfig(main_input_dir,
                     orbit_dir,
                     sample_yaml,
                     rtc_cmd_file):

    rtc_path = os.path.join(f'{main_input_dir}/s1_data/', '*zip')
    # print(rtc_path)
    rtc_list = glob.glob(rtc_path)
    input_str = ''
    for filename in rtc_list:
        print(filename)
        rtc_dir_name = filename.split('/')[-1]
        # input_str += f'        - input_dir/rtc/{rtc_dir_name}\n'
        input_str += f'             - {filename}\n'
    # print(input_str)
    orbit_dir_path = os.path.join(f'{orbit_dir}', '*EOF')
    orbit_list = glob.glob(orbit_dir_path)
    orbit_str = ''
    for filename in orbit_list:
        # print(filename)
        rtc_dir_name = filename.split('/')[-1]
        # input_str += f'        - input_dir/rtc/{rtc_dir_name}\n'
        orbit_str += f'             - {filename}\n'
    # print(orbit_str)
    main_input_dir_str = main_input_dir.replace('/', '_')
    with open(rtc_cmd_file, 'a') as file_en:
    
        new_yaml = f"{main_input_dir}/rtc_runconfig_{main_input_dir_str}.yaml"

        with open(sample_yaml, 'r') as file :
            filedata = file.read()
        outputdir = f'{main_input_dir}'

        # Replace the target string:
        filedata = filedata.replace('test_input_here', input_str)
        filedata = filedata.replace('test_orbit_here', orbit_str)
        filedata = filedata.replace('testhere', outputdir)
        # filedata = filedata.replace('output.h5', f"RTC_{datestr}.h5")

        # Write the file out again
        print(new_yaml)
        with open(new_yaml, 'w') as file:
            file.write(filedata)

        cmdline = f"python3 /mnt/aurora-r0/jungkyo/tool/OPERA_RTC/RTC/app/rtc_s1.py {new_yaml} \n"
        file_en.write(cmdline)



def get_intersecting_mgrs_tiles_list_from_db(
        mgrs,
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

    # Connect to your SQLite database
    conn = sqlite3.connect(mgrs_collection_file)

    # Cursor to execute queries
    cursor = conn.cursor()
    # SQL PRAGMA query to get table info
    cursor.execute(f'PRAGMA table_info(mgrs_burst_db)')

    # Fetch the results
    table_info = cursor.fetchall()

    field_names = [info[1] for info in table_info]  # Column name is in the second position
    mgrs_list_id = field_names.index('mgrs_tiles')
    burst_id = field_names.index('bursts')
    # SQL query
    sql_query = f"""
    SELECT * FROM mgrs_burst_db 
    WHERE relative_orbit_number = {track_number}
    """

    # Execute the query
    cursor.execute(sql_query)

    # Fetch the results
    results = cursor.fetchall()

    # Process the results
    target_burst_id = []
    for row in results:
        mgrs_id_list = ast.literal_eval(row[mgrs_list_id])
        if mgrs in mgrs_id_list:
            print(mgrs_id_list, '--', mgrs)
            print(row[burst_id])
            target_burst_id = row[burst_id]

    # Close the connection
    conn.close()
    return target_burst_id


def run():

    mgrs_tile_db = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-gamma/sample_data/input_dir/ancillary_data/MGRS_tile.sqlite'
    mgrs_tile_collection_db = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-gamma/sample_data/input_dir/ancillary_data/MGRS_tile_collection_v0.2.sqlite'
    sample_yaml = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-calval/sample_product_calval_historical/dswx_sample.yaml'
    rtcsample_yaml = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-calval/sample_product_calval_historical/rtc_runconfig.yaml'
    output_dir = 'batch_dswx'

    # read coordinates
    csv_file_path = '/mnt/aurora-r0/jungkyo/OPERA/DSWx-S1-calval/sample_product_calval_historical/validation_table.csv'
    site_name, mgrs_list, acq_time_list = parse_csv_file_mgrs(csv_file_path)
    
    now = datetime.now()
    current_time_str = now.strftime("%Y%m%dT%H%M%S")
    rtc_script_shell = f'rtc_cmd_script_{current_time_str}'
    dswx_script_shell = f'dswx_cmd_script_{current_time_str}'

    # download s1
    for batch_ind, (site, mgrs, acq) in enumerate(zip(site_name, mgrs_list, acq_time_list)):
        print(batch_ind, '---', site, mgrs, acq)
        if batch_ind not in [17]:
        # if batch_ind in [7]:
            input_date = datetime.strptime(acq, '%Y%m%dT%H%M%SZ')

            # Calculate dates 12 days before and after
            date_before = input_date - timedelta(days=12)
            date_after = input_date + timedelta(days=12)

            # Convert these dates back to the desired string format
            date_before_str = date_before.strftime('%Y%m%d')
            date_after_str = date_after.strftime('%Y%m%d')

            output_batch_dir = f'{output_dir}-{site}'
            os.makedirs(output_batch_dir, exist_ok=True)
            s1_dir = f'{output_batch_dir}/s1_data'
            output_check = glob.glob(f'{s1_dir}/*.zip')
            if len(output_check) == 0:

                cmdline = f'python3 /mnt/aurora-r0/jungkyo/tool/DSWX-SAR_series/DSWX-SAR/src/dswx_sar/download_s1_mgrs.py '\
                    f'-b {mgrs} -s {date_before_str} -e {date_after_str} '\
                    f'-m {mgrs_tile_db}  -o {s1_dir}'

                print(cmdline)
                # os.system(cmdline)

            rtc_out_dir = f'{output_batch_dir}/rtc_outputdir'
            output_check = glob.glob(f'{rtc_out_dir}')            
            zip_file = glob.glob(f'{s1_dir}/*zip')

            if len(zip_file) > 0 and len(output_check)==0:
                for zipff in zip_file:
                    # print(zipff, 'orbit')
                    cmdline = f'python3 /mnt/aurora-r0/jungkyo/tool/fetchOrbit_new.py -i {zipff} -o orbit -u oberonia78@gmail.com'
                    # os.system(cmdline)

            # if len(zip_file) > 0 and len(output_check)==0:
            #     # # create runconfig
            #     create_rtc_runconfig(output_batch_dir, 'orbit', rtcsample_yaml, rtc_script_shell)

    # run rtc_s1
    # os.system(f'sh {rtc_script_shell}')
    wrong_path = 'wrong_items'

    #----------------DSWX-S1----------------------
    for batch_ind, (site, mgrs, acq) in enumerate(zip(site_name, mgrs_list, acq_time_list)):
        print(batch_ind, '---', site, mgrs, acq)
        output_batch_dir = f'{output_dir}-{site}'
        if batch_ind not in [17]:


            output_check = glob.glob(f'{output_batch_dir}/output_dir/*.tif')
            print(output_check)

            file_exist_okay = False
            if len(output_check) == 0:
                rtc_dir = f'{output_batch_dir}/rtc_outputdir/*'
                rtc_dir_list = glob.glob(rtc_dir)
                track_id_list = [part.split('/')[-1].split('_')[0] for part in rtc_dir_list]
                unique_track_id = list(set(track_id_list))
                track_number_list = [track_id[1:]   for track_id in unique_track_id]
                print(track_number_list, mgrs)

                burst_list = get_intersecting_mgrs_tiles_list_from_db(
                    mgrs,
                    mgrs_tile_collection_db,
                    track_number=track_number_list[0])
                
                if len(burst_list)>0:
                    burst_list = ast.literal_eval(burst_list)
                # create runconfig
                    create_runconfig(output_batch_dir,
                                    sample_yaml,
                                    burst_list,
                                    dswx_script_shell)
                    file_exist_okay = True
                else:
                    print(f'{site} {mgrs} is wrong')
                    with open(wrong_path, 'a') as file:
                        file.write(f'{site} {mgrs} is wrong \n')

            if (not glob.glob(f'{output_batch_dir}/anc_data/hand.vrt')) and (file_exist_okay):

                (minx, maxx, miny, maxy, epsg) = \
                    get_bounding_box_from_mgrs_tile_db(
                        [mgrs], 
                        mgrs_tile_db)
                
                lon_list = []
                lat_list = []
                for xvalue in [minx, maxx]:
                    for yvalue in [miny, maxy]:
                        lon, lat = get_lonlat(float(xvalue), float(yvalue), int(epsg))
                        lon_list.append(lon)
                        lat_list.append(lat)

                lat_center = (np.nanmax(lat_list) + np.nanmin(lat_list))/2
                lon_center = (np.nanmax(lon_list) + np.nanmin(lon_list))/2

                rtc_dir = f'{output_batch_dir}/rtc_outputdir/'
                x_start = lon_center - 2.5
                x_end = lon_center + 2.5
                y_start = lat_center - 2.5
                y_end = lat_center + 2.5

                os.makedirs(f'{output_batch_dir}/anc_data' ,exist_ok=True)

                hand_cmdline = f'python3 /mnt/aurora-r0/jungkyo/tool/DSWX-SAR_series/DSWX-SAR/src/dswx_sar/stage_hand.py '\
                                f'-b {x_start-1} {y_start-1} {x_end+1} {y_end+1}  -o {output_batch_dir}/anc_data/hand.vrt'

                os.system(hand_cmdline)


    # os.system(f'sh {dswx_script_shell}')


def main():
    # parser = createParser()
    # args = parser.parse_args()

    run()


if __name__ == '__main__':
    main()
