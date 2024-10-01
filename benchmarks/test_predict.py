import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loader import Loader

os.environ["HYDRA_FULL_ERROR"] = "1"


@torch.no_grad()
@hydra.main(config_path="../cfg/benchmark", config_name="test_predict")
def test_predict(cfg: DictConfig):
    loader = Loader()

    devices = cfg.trainer.devices
    device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")

    tokenizer = loader(cfg.tokenizer)
    
    try:
        test_transforms = loader(cfg.transforms, train=False)
    except:
        test_transforms = None
    test_dataset = loader(cfg.dataset, dataset_type="test", train=False, transforms=test_transforms)
    test_loader = loader(cfg.data_loading, dataset=test_dataset)
    
    model = loader(cfg.model)

    model.load_state_dict(torch.load(cfg.model_weights_path)["state_dict"])

    if cfg.trainer.cuda_distributed_training:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=devices)
            print("cuda distributed training")

    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    test_all_outputs = []
    
    model.eval()
    for data_test in tqdm(test_loader):

        input_test, label_test = data_test
        input_test = tokenizer(input_test[0], input_test[1], padding=True, truncation=True, return_tensors="pt")
        input_test, label_test = {key: value.squeeze().to(device) for key, value in input_test.items()}, label_test.to(device)  

        if cfg.trainer.fp16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred_test = model(**input_test).logits.argmax(axis=1)
        else:
            pred_test = model(**input_test).logits.argmax(axis=1)

        test_all_outputs.append(pred_test.cpu().detach().numpy())

    test_all_outputs = np.concatenate(test_all_outputs)
    meta = test_dataset.meta[["id"]]
    meta["prediction"] = test_all_outputs

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(cfg.model_weights_path)), 
        f"test_predict_{os.path.basename(os.path.splitext(cfg.model_weights_path)[0])}.csv"
    )
    meta.to_csv(save_path, index=False)
    

if __name__ == "__main__":
    test_predict()
