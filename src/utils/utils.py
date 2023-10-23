import torch
import os
import logging
from munch import Munch
from src.models.encoders.clip import CLIP
from src.models.encoders.align import ALIGN
from src.models.encoders.altclip import AltCLIP
from src.models.encoders.bridgetower import BridgeTower
from src.models.encoders.groupvit import GroupViT


def get_device():
    """

    :return:
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_built() else 'cpu'


def is_local():
    import os
    if 'mhendriksen2' in os.getcwd():
        return False
    return True


def get_machine_type() -> str:
    cwd = os.getcwd()
    if 'mhendriksen2' in cwd:
        return 'dsp'
    elif 'mhendriksen' in cwd:
        return 'local'
    else:
        raise NotImplementedError

def get_results_dir():
    machine_type = get_machine_type()
    if machine_type == 'dsp':
        dir = './results'
    elif machine_type == 'local':
        dir = '/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-vl/results'
    else:
        raise NotImplementedError
    return dir

def get_project_path():
    machine_type = get_machine_type()
    if machine_type == 'dsp':
        path = '/home/mhendriksen2/projects/evaluating-cmr-in-vl'
    elif machine_type == 'local':
        path = '/Users/mhendriksen/Desktop/repositories/evaluating-cmr-in-vl'
    else:
        raise NotImplementedError
    return path


def get_config_path(dataset, model):
    machine_type = get_machine_type()
    config_filename = f"development_{machine_type}.yaml"
    path = os.path.join(f"./config/{model}/{dataset}", config_filename)
    print('Config path: ', path)
    return path

def get_config(dataset, model):
    path = get_config_path(dataset, model)
    with open(path, 'rb') as f:
        config = Munch.fromYAML(f)
        print('Loaded config path')
    return config


def get_logger(config):
    root = f'logs/{config.args.model}/{config.args.dataset}'
    os.makedirs(root, exist_ok=True)
    filename = f'{root}/{config.args.task}-{config.args.perturbation}.log'
    logging.basicConfig(format="%(asctime)-15s %(levelname)-8s %(message)s",
                        filename=filename,
                        filemode='a+',
                        encoding='utf-8',
                        level=logging.DEBUG)
    logging.info("New script")
    print('Setup a logger at: ', filename)
    
    return logging


def get_model(config):
    model_name = config.args.model
    # load the model
    # print('model name: ', model)
    if model_name == "clip":
        model = CLIP(config=config)
    elif model_name == "align":
        model = ALIGN(config=config)
    elif model_name == 'altclip':
        model = AltCLIP(config=config)
    elif model_name == 'bridgetower':
        model = BridgeTower(config=config)
    elif model_name == 'groupvit':
        model = GroupViT(config=config)
    elif model_name == "flava":
        model = FLAVA(config=config)
    else:
        print(
            "Uknown model, please choose among the following options: clip, align, altclip, bridgetower, groupvit"
        )
    return model

def get_abs_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
