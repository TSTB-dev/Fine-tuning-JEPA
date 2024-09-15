import argparse

import multiprocessing as mp

import torch.distributed as dist
import pprint
import yaml

from src.utils.distributed import init_distributed
from src.pretrain import main as pretrain_main
from src.downstream import main as downstream_main
from src.evaluation import main as evaluation_main

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname", type=str,
    help="name of config file to load",
    default="configs.yaml"
)
parser.add_argument(
    "--task", type=str, choices=["pretrain", "downstream", "evaluation", "visualization"],
)
parser.add_argument(
    "--devices", type=str, nargs="+", default=["cuda:0"],
)

def process_main(rank, fname, world_size, devices, task):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(":")[-1])
    
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    
    logging.info(f"called-params {fname}")  
    
    # load params
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logging.info("loaded params...")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")
    if task == "pretrain":
        pretrain_main(params)
    elif task == "downstream":
        downstream_main(params)
    elif task == "evaluation":
        evaluation_main(params)
    else:
        raise ValueError(f"Task {task} should be specified")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method("spawn", True)
    
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args.task)
        )
        p.start()
        processes.append(p)
    
    # wait for all processes to finish
    for p in processes:
        p.join()