import os
import argparse
import torch
import random
import numpy as np
from datetime import datetime 

from common.config import Config
from common.logger import setup_logger
from common.registry import registry
from latentmem.runner import LatentMemRunner

def set_seed(random_seed: int, use_gpu: bool):

    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False      

    print(f"set seed: {random_seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Memory Master: Memory System for LLM-based Multi-Agent System")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args

def get_working_dir(config: Config) -> str:
    mode = config.run_cfg.mode
    model_name = config.model_cfg.mas.llm_name_or_path.split("/")[-1]
    dataset_name = config.dataset_cfg.name
    mas_structure = config.model_cfg.mas.structure

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    latents_len = config.model_cfg.memory.weaver.latents_len
    rag_mode = config.model_cfg.memory.rag.mode
    use_weaver = config.model_cfg.memory.use_weaver
    
    # <model_name>/<dataset_name>/<mas_structure>/<rag_mode>_<latent_len>_<use_weaver>_<timestamp>
    working_dir = f"mem={rag_mode}_ll={latents_len}_weaver={use_weaver}_{time}"
    return os.path.join(".cache", mode, model_name, dataset_name, mas_structure, working_dir)


def main():

    # parse configs
    args = parse_args()
    config = Config(args)
    
    set_seed(config.run_cfg.seed, use_gpu=True)

    # manually set up working dir
    working_dir = get_working_dir(config)

    # set up logger
    log_dir = os.path.join(working_dir, "logs")
    setup_logger(log_dir)

    config.pretty_print()

    # get data builder
    config_dict = config.to_dict()
    data_config_dict = config_dict.get("dataset")
    data_builder_cls = registry.get_builder_class(data_config_dict.get("name"))
    data_builder = data_builder_cls(data_config_dict)
    
    # get memory mas
    model_config_dict = config_dict.get("model")
    memory_mas_cls = registry.get_mas_class(model_config_dict.get("mas").get("structure"))
    memory_mas = memory_mas_cls.from_config(
        config=model_config_dict, 
        working_dir=working_dir,
        task_domain=data_config_dict.get("name")
    )

    # build runner
    runner = LatentMemRunner(
        memory_mas=memory_mas, 
        data_builder=data_builder, 
        config=config, 
        working_dir=working_dir
    )
     
    # train or evaluate
    runner.execute(mode=config.run_cfg.mode)

if __name__ == "__main__":
    main()