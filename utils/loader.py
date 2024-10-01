import os
import importlib
import hydra
import shutil
from omegaconf import OmegaConf


class Loader:
    def __init__(self, dir_save=""):
        if dir_save:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.makedirs(dir_save, exist_ok=True)
            if dir_save:
                shutil.copytree(os.path.join(root, "trains"), os.path.join(dir_save, "trains"), dirs_exist_ok=True)
                # shutil.copytree(os.path.join(root, "benchmarks"), os.path.join(dir_save, "benchmarks"), dirs_exist_ok=True)
                shutil.copytree(os.path.join(root, "cfg"), os.path.join(dir_save, "cfg"), dirs_exist_ok=True)
                shutil.copytree(os.path.join(root, "pycfg"), os.path.join(dir_save, "pycfg"), dirs_exist_ok=True)
                shutil.copytree(os.path.join(root, "utils"), os.path.join(dir_save, "utils"), dirs_exist_ok=True)
                # shutil.copytree(os.path.join(root, "submodules"), os.path.join(dir_save, "submodules"), dirs_exist_ok=True)

    def __call__(self, cfg, *args, **kwargs):
        if not isinstance(cfg, dict):
            cfg = OmegaConf.to_container(cfg)
        if "_py_" in cfg:
            filepath = cfg.pop("_py_")
            filepath = os.path.splitext(filepath)[0].replace("/", ".")
            cfg.update(kwargs)
            module = importlib.import_module(filepath)
            return module.__dict__["__load__"](*args, **cfg)
        if "_target_" in cfg:
            return hydra.utils.instantiate(cfg, *args, **kwargs)
