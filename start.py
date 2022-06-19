import logging
import os
import shutil
import sys
import time
from typing import List

import yaml
from easydict import EasyDict as edict

# view https://github.com/protocolbuffers/protobuf/issues/10051 for detail
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw


def get_merged_config() -> edict:
    """
    merge ymir_config, cfg.param and code_config
    code_config will be overwrited by cfg.param.
    """
    def get_code_config(code_config_file: str) -> dict:
        if code_config_file is None:
            return dict()
        else:
            with open(code_config_file, 'r') as f:
                return yaml.safe_load(f)

    exe_cfg = env.get_executor_config()
    code_config_file = exe_cfg.get('code_config', None)
    code_cfg = get_code_config(code_config_file)
    code_cfg.update(exe_cfg)

    merged_cfg = edict()
    # the hyperparameter information
    merged_cfg.param = code_cfg

    # the ymir path information
    merged_cfg.ymir = env.get_current_env()
    return merged_cfg


def start() -> int:
    cfg = get_merged_config()

    logging.info(f'merged config: {cfg}')
    uid = os.getuid()
    gid = os.getgid()
    logging.info(f'user info: {uid}:{gid}')

    if cfg.ymir.run_training:
        _run_training(cfg)
    elif cfg.ymir.run_mining:
        _run_mining(cfg)
    elif cfg.ymir.run_infer:
        _run_infer(cfg)
    else:
        logging.warning('no task running')

    return 0


def _run_training(cfg: edict) -> None:
    """
    sample function of training, which shows:
    1. how to get config
    2. how to read training and validation datasets
    3. how to write logs
    4. how to write training result
    """
    # get config
    class_names: List[str] = cfg.param['class_names']
    expected_mAP: float = cfg.param.get('map')
    model: str = cfg.param.get('model')

    # read training dataset items
    # note that `dr.item_paths` is a generator
    for asset_path, ann_path in dr.item_paths(env.DatasetType.TRAINING):
        logging.info(f"asset: {asset_path}, annotation: {ann_path}")
        with open(ann_path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                class_id, x1, y1, x2, y2 = [int(s)
                                            for s in line.strip().split(',')]
                name = class_names[class_id]
                logging.info(f"{name} xmin={x1} ymin={y1} xmax={x2} ymax={y2}")
        break

    # write task process percent to monitor.txt
    monitor.write_monitor_logger(percent=0.0)

    # fake training function
    _dummy_work(cfg)

    # suppose we have a long time training, and have saved the final model
    # use `cfg.ymir.output.models_dir` to get model output dir
    models_dir = cfg.ymir.output.models_dir
    os.makedirs(models_dir, exist_ok=True)
    model_weight = os.path.join(models_dir, f'{model}.pt')
    os.system(f'touch {model_weight}')

    # write other information
    with open(os.path.join(models_dir, 'model.yaml'), 'w') as f:
        f.write(f'model: {model}')
    shutil.copy('models/vgg.py',
                os.path.join(models_dir, 'vgg.py'))

    # use `rw.write_training_result` to save training result
    # the files in model_names will be saved and can be download from ymir-web
    rw.write_training_result(model_names=[f'{model}.pt',
                                          'model.yaml',
                                          'vgg.py'],
                             mAP=expected_mAP,
                             classAPs={class_name: expected_mAP
                                       for class_name in class_names})

    # if task done, write 100% percent log
    logging.info('task done')
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict) -> None:
    pass

def _run_infer(cfg: edict) -> None:
    pass


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    sys.exit(start())
