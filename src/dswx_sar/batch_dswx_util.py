import os
import glob

def create_runconfig(main_input_dir,
                     sample_yaml,
                     dswx_cmd_file):

    rtc_path = os.path.join(f'{main_input_dir}/input_dir/rtc', 'T*')
    rtc_list = glob.glob(rtc_path)
    input_str = ''
    for filename in rtc_list:
        print(filename)
        rtc_dir_name = filename.split('/')[-1]
        # input_str += f'        - input_dir/rtc/{rtc_dir_name}\n'
        input_str += f'        - {filename}\n'


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
