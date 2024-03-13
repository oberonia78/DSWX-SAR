import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from pathlib import Path
import mimetypes
import logging
import time

from dswx_sar import dswx_sar_util
from dswx_sar import generate_log
from dswx_sar.dswx_runconfig import _get_parser, RunConfig

logger = logging.getLogger('dswx_s1')

class StatisticWater:
    def __init__(self, scratch_dir, pol_list):
        self.outputdir = scratch_dir
        self.pol_str = '_'.join(pol_list)

        filt_im_str = os.path.join(self.outputdir, f"filtered_image_{self.pol_str}.tif")
        band_set = dswx_sar_util.read_geotiff(filt_im_str)
        
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
                         reference_threshold,
                         mask=None):
        if mask is not None:
            self.mask = (self.mask) | (mask)
 
        reference[self.mask] = 0
        classified[self.mask] = 0
        
        self.ref_msk = (reference > reference_threshold) & (reference <= 1)
        self.cls_msk = classified == class_value
        self.overlap = np.logical_and(self.ref_msk, self.cls_msk)
        num_ref = np.count_nonzero(self.ref_msk)
        num_cls = np.count_nonzero(self.cls_msk)
        
        correct_class = np.count_nonzero(self.overlap)
        self.producer_acc = correct_class / num_ref * 100
        self.user_acc = correct_class /num_cls * 100

        logger.info(f'Number_of_Reference: {num_ref}')
        logger.info(f'Number_of_classified: {num_cls}')
        logger.info(f'Number_of_postivie_true: {correct_class}')

        logger.info(f'User accuracy: {self.user_acc}')
        logger.info(f'Producer accuracy: {self.producer_acc}')
    
    def create_comparision_image(self, png_name=None):
        
        index_map = np.zeros(self.cls_msk.shape,dtype='int8')

        only_ref = np.logical_and(self.ref_msk, np.invert(self.cls_msk))
        only_cls = np.logical_and(np.invert(self.ref_msk), self.cls_msk)
        
        index_map[self.overlap] = 1
        index_map[only_ref] = 2
        index_map[only_cls] = 3
        # overlapped, reference, dswx
        colors = ["blue" , "red", "green"]  # use hex colors here, if desired.
        cmap = ListedColormap(colors) 

        fig, ax = plt.subplots(1,1,figsize=(30, 30))
        im = ax.imshow(10*np.log10(self.band_avg), 
                      cmap = plt.get_cmap('gray'),
                      vmin=-25,
                      vmax=-5)

        mask_layer = np.ma.masked_where(index_map == 0, index_map)
        plt.imshow(mask_layer, alpha=0.8, cmap=cmap, interpolation='nearest')
        # plt.imshow(self.overlap, alpha=0.9, cmap =blue_cmap)
        # # plt.imshow(only_ref, alpha=0.9, cmap = plt.get_cmap('Reds'))
        # # ax.imshow(only_cls, alpha=0.9, cmap = plt.get_cmap('Greens'))

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
        if png_name == None:
            plt.savefig(os.path.join(self.outputdir, 'DSWX_S1_stat_{}'.format(self.pol_str)) )
        else:
            plt.savefig(os.path.join(self.outputdir, f'{png_name}_{self.pol_str}') )
        plt.close()



def run(cfg):
    '''Remove the false water which have low backscattering based on the 
    occurrence map and landcover map.    
    '''
    logger.info(f'start computing statistics')
    t_all = time.time()

    outputdir = cfg.groups.product_path_group.scratch_path
    sas_outputdir = cfg.groups.product_path_group.sas_output_path
    processing_cfg = cfg.groups.processing
    pol_list = processing_cfg.polarizations
    pol_str = '_'.join(pol_list)
    dswx_workflow = processing_cfg.dswx_workflow

    water_cfg = processing_cfg.reference_water
    ref_water_max = water_cfg.max_value
    ref_no_data = water_cfg.no_data_value

    interp_wbd_str = os.path.join(outputdir, 'interpolated_wbd')
    interp_wbd = dswx_sar_util.read_geotiff(interp_wbd_str) / ref_water_max
    
    stat = StatisticWater(outputdir, pol_list)

    if dswx_workflow == 'opera_dswx_s1':
        water_map_tif_str = os.path.join(outputdir, f'bimodality_output_binary_{pol_str}.tif')
        water_map = dswx_sar_util.read_geotiff(water_map_tif_str)
        
        stat.compute_accuracy(classified=water_map, 
                            class_value=1,
                            reference=interp_wbd, 
                            reference_threshold=0.8,
                            mask=interp_wbd>1)
        stat.create_comparision_image('bimodal_step')
        
        water_map_tif_str = os.path.join(outputdir, f'refine_landcover_binary_{pol_str}.tif')
        water_map = dswx_sar_util.read_geotiff(water_map_tif_str)

        stat.compute_accuracy(classified=water_map, 
                            class_value=1,
                            reference=interp_wbd, 
                            reference_threshold=0.8,
                            mask=interp_wbd>1)
        stat.create_comparision_image('landcover_step')

    water_map_tif_str = os.path.join(outputdir, f'region_growing_output_binary_{pol_str}.tif')
    water_map = dswx_sar_util.read_geotiff(water_map_tif_str)

    stat.compute_accuracy(classified=water_map, 
                         class_value=1,
                         reference=interp_wbd, 
                         reference_threshold=0.8,
                         mask=interp_wbd>1 )
    stat.create_comparision_image('region_growing_step')
    
    t_time_end =time.time()

    t_all_elapsed = t_time_end - t_all
    logger.info(f"successfully ran computing statistics in {t_all_elapsed:.3f} seconds")
                    
def main():

    parser = _get_parser()

    args = parser.parse_args()


    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return
 
    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_s1', args)    
    generate_log.configure_log_file(cfg.groups.log_file)

    run(cfg)

if __name__ == '__main__':
    main()

