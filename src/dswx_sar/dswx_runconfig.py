from dataclasses import dataclass
import os
import sys
import logging
from functools import singledispatch
from types import SimpleNamespace

import yamale
import argparse
from ruamel.yaml import YAML

import dswx_sar

logger = logging.getLogger('dswx-s1')

WORKFLOW_SCRIPTS_DIR = os.path.dirname(dswx_sar.__file__)

def _get_parser():
    parser = argparse.ArgumentParser(description='',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input
    parser.add_argument('input_yaml',
                        type=str,
                        nargs='+',
                        help='Input YAML run configuration file')

    parser.add_argument('--debug_mode', action='store_true', default=False,
                        help='Print figures. Off/False by default.')

    parser.add_argument('--log',
                        '--log-file',
                        dest='log_file',
                        type=str,
                        help='Log file')

    return parser

# def _deep_update(original, update):
def _deep_update(original, update):

    """Update default runconfig dict with user-supplied dict.
    Parameters
    ----------
    original : dict
        Dict with default options to be updated
    update: dict
        Dict with user-defined options used to update original/default
    Returns
    -------
    original: dict
        Default dictionary updated with user-defined options
    References
    ----------
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for key, val in update.items():
        if isinstance(val, dict):
            original[key] = _deep_update(original.get(key, {}), val)
        else:
            if val is not None:
                original[key] = val

    # return updated original
    return original

def load_validate_yaml(yaml_path: str, workflow_name: str) -> dict:
    """Initialize RunConfig class with options from given yaml file.
    Parameters
    ----------
    yaml_path : str
        Path to yaml file containing the options to load
    workflow_name: str
        Name of the workflow for which uploading default options
    """
    try:
        # Load schema corresponding to 'workflow_name' and to validate against
        schema_name = workflow_name
        schema = yamale.make_schema(
            f'{WORKFLOW_SCRIPTS_DIR}/schemas/{schema_name}.yaml',
            parser='ruamel')
    except:
        err_str = f'unable to load schema for workflow {workflow_name}.'
        logger.error(err_str)
        raise ValueError(err_str)

    # load yaml file or string from command line
    if os.path.isfile(yaml_path):
        try:
            data = yamale.make_data(yaml_path, parser='ruamel')
        except yamale.YamaleError as yamale_err:
            err_str = f'Yamale unable to load {workflow_name} ' \
                      'runconfig yaml {yaml_path} for validation.'
            logger.error(err_str)
            raise yamale.YamaleError(err_str) from yamale_err
    else:
        raise FileNotFoundError

    # validate yaml file taken from command line
    try:
        yamale.validate(schema, data)
    except yamale.YamaleError as yamale_err:
        err_str = f'Validation fail for {workflow_name} runconfig yaml {yaml_path}.'
        logger.error(err_str)
        raise yamale.YamaleError(err_str) from yamale_err

    # load default runconfig
    parser = YAML(typ='safe')
    default_cfg_path = f'{WORKFLOW_SCRIPTS_DIR}/defaults/{schema_name}.yaml'
    with open(default_cfg_path, 'r') as f_default:
        default_cfg = parser.load(f_default)
    print('default,', default_cfg)
    with open(yaml_path, 'r') as f_yaml:
        user_cfg = parser.load(f_yaml)

    # Copy user-supplied configuration options into default runconfig
    _deep_update(default_cfg, user_cfg)
    # Validate YAML values under groups dict
    if 'groups' in default_cfg['runconfig'].keys():
        validate_group_dict(default_cfg['runconfig']['groups'])
    print('default2,', default_cfg)

    return default_cfg


def check_write_dir(dst_path: str):
    """Check if given directory is writeable; else raise error.
    Parameters
    ----------
    dst_path : str
        File path to directory for which to check writing permission
    """
    if not dst_path:
        dst_path = '.'

    # check if scratch path exists
    dst_path_ok = os.path.isdir(dst_path)

    if not dst_path_ok:
        try:
            os.makedirs(dst_path, exist_ok=True)
        except OSError:
            err_str = f"Unable to create {dst_path}"
            logger.error(err_str)
            raise OSError(err_str)

    # check if path writeable
    write_ok = os.access(dst_path, os.W_OK)
    if not write_ok:
        err_str = f"{dst_path} scratch directory lacks write permission."
        logger.error(err_str)
        raise PermissionError(err_str)

def check_file_path(file_path: str) -> None:
    """Check if file_path exist else raise an error.
    Parameters
    ----------
    file_path : str
        Path to file to be checked
    """
    if not os.path.exists(file_path):
        err_str = f'{file_path} not found'
        logger.error(err_str)
        raise FileNotFoundError(err_str)


def check_polarizations(pol_list):
    """Sort polarizations so that co-pols are preceded. 
    Parameters
    ----------
    pol_list : list
        List of polarizations.
    """
    co_pol_list = []
    cross_pol_list = []
    def custom_sort(pol):
        if pol in ['VV', 'HH']:
            return (0, pol)  # Sort 'VV' and 'HH' before others
        return (1, pol)

    pol_list = sorted(pol_list, key=custom_sort)

    for pol in pol_list:
        if pol in ['VV', 'HH']:
            co_pol_list.append(pol)
        else:
            cross_pol_list.append(pol)

    return co_pol_list, cross_pol_list, pol_list


def validate_group_dict(group_cfg: dict) -> None:
    """Check and validate runconfig entries.
    Parameters
    ----------
    group_cfg : dict
        Dictionary storing runconfig options to validate
    """
    # Check 'dynamic_ancillary_file_groups' section of runconfig
    # Check that ancillary file exist and is GDAL-compatible
    dem_path = group_cfg['dynamic_ancillary_file_group']['dem_file']
    landcover_path = group_cfg[
                    'dynamic_ancillary_file_group']['worldcover_file']
    ref_water_path = group_cfg[
                    'dynamic_ancillary_file_group']['reference_water_file']
    hand_path = group_cfg[
                    'dynamic_ancillary_file_group']['hand_file']
    ancillary_file_paths = [dem_path, landcover_path,
                            ref_water_path, hand_path]

    for path in ancillary_file_paths:
        check_file_path(path)

    # Check 'product_group' section of runconfig.
    # Check that directories herein have writing permissions
    product_group = group_cfg['product_path_group']
    check_write_dir(product_group['sas_output_path'])
    check_write_dir(product_group['scratch_path'])

    static_ancillary_flag = group_cfg[
        'static_ancillary_file_group']['static_ancillary_inputs_flag']

    if static_ancillary_flag:
        mgrs_database_file = group_cfg[
            'static_ancillary_file_group']['mgrs_database_file']
        mgrs_collection_database_file = group_cfg[
            'static_ancillary_file_group']['mgrs_collection_database_file']

        if mgrs_database_file is None or mgrs_collection_database_file is None:
            err_str = f'Static ancillary data flag is {static_ancillary_flag} ' \
                      f'but mgrs_database_file {mgrs_database_file} and ' \
                      f'mgrs_collection_database_file {mgrs_collection_database_file}'
            logger.error(err_str)
            raise ValueError(err_str)

        check_file_path(mgrs_collection_database_file)
        check_file_path(mgrs_database_file)


@singledispatch
def wrap_namespace(ob):
    return ob

@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{key: wrap_namespace(val)
                              for key, val in ob.items()})

@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(val) for val in ob]

def unwrap_to_dict(sns: SimpleNamespace) -> dict:
    sns_as_dict = {}
    for key, val in sns.__dict__.items():
        if isinstance(val, SimpleNamespace):
            sns_as_dict[key] = unwrap_to_dict(val)
        else:
            sns_as_dict[key] = val

    return sns_as_dict

@dataclass(frozen=True)
class RunConfig:
    '''dataclass containing RTC runconfig'''
    # workflow name
    name: str
    # runconfig options converted from dict
    groups: SimpleNamespace
    # run config path
    run_config_path: str

    @classmethod
    def load_from_yaml(cls, yaml_path: str, workflow_name: str, args):
        """Initialize RunConfig class with options from given yaml file.
        Parameters
        ----------
        yaml_path : str
            Path to yaml file containing the options to load
        workflow_name: str
            Name of the workflow for which uploading default options
        """
        cfg = load_validate_yaml(yaml_path, workflow_name)

        groups_cfg = cfg['runconfig']['groups']

        # Convert runconfig dict to SimpleNamespace
        sns = wrap_namespace(groups_cfg)
        product = sns.primary_executable.product_type
        sensor = product.split('_')[-1]
        ancillary = sns.dynamic_ancillary_file_group
        print('-----')
        print(ancillary)
        print('-----')

        algorithm_cfg = load_validate_yaml(ancillary.algorithm_parameters,
                                           f'algorithm_parameter_{sensor.lower()}')
        print('-----')
        print(algorithm_cfg)
        print('-----')
        co_pol, cross_pol, pol_list = check_polarizations(
            algorithm_cfg['runconfig']['processing']['polarizations'])
        
        algorithm_cfg['runconfig']['processing']['polarizations'] = pol_list
        algorithm_cfg['runconfig']['processing']['copol'] = co_pol[0]
        algorithm_cfg['runconfig']['processing']['crosspol'] = cross_pol[0]

        algorithm_sns = wrap_namespace(algorithm_cfg['runconfig']['processing'])
        
        # copy runconfig parameters from dictionary
        sns.processing = algorithm_sns
        processing_group = sns.processing

        debug_mode = processing_group.debug_mode

        if args.debug_mode and not debug_mode:
            logger.warning(f'command line visualization "{args.debug_mode}"'
                f' has precedence over runconfig visualization "{debug_mode}"')
            sns.processing.debug_mode = args.debug_mode

        log_file = sns.log_file
        if args.log_file is not None:
            logger.warning(f'command line log file "{args.log_file}"'
                f' has precedence over runconfig log file "{log_file}"')
            sns.log_file = args.log_file

        return cls(cfg['runconfig']['name'], sns, yaml_path)

    @property
    def input_file_path(self):
        return self.groups.input_file_group.input_file_path

    @property
    def dem(self) -> str:
        return self.groups.dynamic_ancillary_file_group.dem_file

    @property
    def dem_description(self) -> str:
        return self.groups.dynamic_ancillary_file_group.dem_description

    @property
    def polarizations(self): #-> list[str]:
        return self.groups.processing.polarizations

    @property
    def product_path(self):
        return self.groups.product_group.product_path

    @property
    def product_id(self):
        return self.groups.product_group.product_id

    @property
    def scratch_path(self):
        return self.groups.product_group.scratch_path

    def as_dict(self):
        ''' Convert self to dict for write to YAML/JSON
        Unable to dataclasses.asdict() because isce3 objects can not be pickled
        '''
        self_as_dict = {}
        for key, val in self.__dict__.items():
            if key == 'groups':
                val = unwrap_to_dict(val)

            self_as_dict[key] = val
        return self_as_dict

    def to_yaml(self):
        self_as_dict = self.as_dict()
        yaml = YAML(typ='safe')
        yaml.dump(self_as_dict, sys.stdout, indent=4)