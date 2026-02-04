import json
import logging
import os
import yaml

from omegaconf import OmegaConf

from common.registry import registry

class Config:
    def __init__(self, args):
        self.config = {}
        
        self.args = args
        
        # Register the config and configuration for setup
        registry.register("configuration", self)

        user_config = self._build_opt_list(self.args.options)   

        config = OmegaConf.load(self.args.cfg_path)             
        runner_config = self.build_runner_config(config, **user_config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config, **user_config)

        self.config = OmegaConf.merge(  
            runner_config, model_config, dataset_config, user_config
        )
    
    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)
    
    @staticmethod
    def build_model_config(config, **kwargs):
        # mas_structure = config.model.mas.structure
        # mas_config_path = os.path.join("configs", "structures.yaml")
        # with open(mas_config_path, "r", encoding="utf-8") as f:
        #     structures = yaml.safe_load(f)
        
        # if mas_structure not in structures:
        #     raise ValueError()
        
        # mas_params = structures[mas_structure]
        # if mas_params is not None:
        #     for k, v in mas_params.items():
        #         config.model.mas[k] = v
            
        return {"model": config.model}

    @staticmethod
    def build_runner_config(config, **kwargs):
        return {"run": config.run}

    @staticmethod
    def build_dataset_config(config, **kwargs):
        return {"dataset": config.dataset}
    
    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]
    
    def get_config(self):
        return self.config

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def dataset_cfg(self):
        return self.config.dataset

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.dataset))

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))
    
    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)  
        return json.dumps(container, indent=4, sort_keys=True)  

    def to_dict(self):
        return OmegaConf.to_container(self.config)