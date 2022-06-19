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
    function for training task
    1. convert dataset
    2. training model
    3. save model weight/hyperparameter/... to design directory
    """
    # 1. convert dataset
    out_dir = cfg.ymir.output.root_dir
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0))
    
    epochs = cfg.param.epochs
    batch_size = cfg.param.batch_size
    arch = cfg.param.arch
    lr = cfg.param.lr
    gpu = cfg.param.gpu_id
    num_workers = cfg.param.num_workers

    # 2. training model
    model_config = cfg.ymir.param.model_config
    models_dir = cfg.ymir.output.models_dir
    
    command = f'python train.py --gpu {gpu} --batch_size {batch_size} --arch {arch} --num_workers {num_workers} --lr {lr}'
    logging.info(f'start training: {command}')

    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.TASK, p=1.0))

    # if task done, write 100% percent log
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
