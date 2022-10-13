from omegaconf import OmegaConf
from easydict import EasyDict

def load_cfg(path):
    # TODO(jh): check cfg grammars & values
    cfg=OmegaConf.load(path)
    return cfg

def convert_to_easydict(cfg):
    cfg=OmegaConf.to_container(cfg,resolve=True)
    cfg=EasyDict(cfg)
    return cfg