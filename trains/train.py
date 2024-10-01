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

from utils.metrics import calc_stats, plot_confusion_matrix
from utils.loader import Loader
from submodules.pytorch_balanced_sampler.pytorch_balanced_sampler.sampler import SamplerFactory

os.environ["HYDRA_FULL_ERROR"] = "1"


# @torch.no_grad()
@hydra.main(config_path="../cfg/trains", config_name="train")
def train(cfg: DictConfig):
    loader = Loader(os.path.join(os.path.dirname(cfg.logger.log_dir), cfg.project_name))

    devices = cfg.trainer.devices
    device = torch.device(f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")

    tokenizer = loader(cfg.tokenizer)
    
    try:
        train_transforms = loader(cfg.transforms, train=True)
    except:
        train_transforms = None
    train_dataset = loader(cfg.dataset, dataset_type="train", train=True, transforms=train_transforms)
    if cfg.data_loading.batch_sampler:
        class_idxs = [train_dataset.meta.index[train_dataset.meta.label == i].tolist() for i in range(train_dataset.meta.label.nunique())]
        batch_sampler = SamplerFactory().get(
            class_idxs=class_idxs,
            batch_size=cfg.data_loading.batch_size,
            n_batches=min([len(x) // cfg.data_loading.batch_size * train_dataset.meta.label.nunique() for x in class_idxs]) + 1,
            alpha=cfg.data_loading.batch_sampler_alpha,
            kind=cfg.data_loading.batch_sampler_kind
        )
        train_loader = loader(cfg.data_loading.train_loader, dataset=train_dataset, batch_sampler=batch_sampler)
    else:
        train_loader = loader(cfg.data_loading.train_loader, dataset=train_dataset, batch_size=cfg.data_loading.batch_size, shuffle=cfg.data_loading.shuffle)
    
    val_loader = {}
    for val_dataset_type in cfg.val_dataset_types:
        try:
            val_transforms = loader(cfg.transforms, train=False)
        except:
            val_transforms = None
        val_dataset = loader(cfg.dataset, dataset_type=val_dataset_type, train=False, transforms=val_transforms)
        val_loader[val_dataset_type] = loader(cfg.data_loading.val_loader, dataset=val_dataset)
            
    model = loader(cfg.model)

    # for param in model.distilbert.embeddings.parameters():
    #     param.requires_grad = False

    # for i in range(len(model.distilbert.transformer.layer)):
    #     if i < 1:
    #         for param in model.distilbert.transformer.layer[i].parameters():
    #             param.requires_grad = False

    # freeze layers
    # if cfg.model.freeze_model:
    #     for name, param in model.named_parameters():
    #         if cfg.model.unfrozen_layer in name:  # Слой после которого мы хотим начать обучение
    #             break
    #         param.requires_grad = False

    optimizer = loader(cfg.optimizer, model.parameters())
    if "scheduler" in cfg:
        scheduler = loader(cfg.scheduler, optimizer)

    if cfg.trainer.cuda_distributed_training:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=devices)
            print("cuda distributed training")

    model = model.to(device)

    criterion = loader(cfg.loss)
    writer = loader(cfg.logger)
    scaler = torch.cuda.amp.GradScaler()
    
    best_monitor_metric = {metric: 0 for metric in cfg.trainer.monitor_metric}
    for epoch in range(cfg.trainer.epochs):
        print("start_training")

        all_train_loss = 0
        train_all_labeles = []
        train_all_outputs = []
        all_metrics = {}
        
        model.train()
        for data_train in tqdm(train_loader):
            optimizer.zero_grad()

            input_train, label_train = data_train
            input_train = tokenizer(input_train[0], input_train[1], padding=True, truncation=True, return_tensors="pt")
            input_train, label_train = {key: value.squeeze().to(device) for key, value in input_train.items()}, label_train.to(device)  

            if cfg.trainer.fp16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_train = model(**input_train).logits
                    train_loss = criterion(pred_train, label_train)
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_train = model(**input_train).logits
                train_loss = criterion(pred_train, label_train)
                train_loss.backward()
                optimizer.step()

            all_train_loss += train_loss.item()
            train_all_labeles.append(label_train.cpu().detach().numpy())
            train_all_outputs.append(pred_train.cpu().detach().numpy())

        train_all_labeles = np.concatenate(train_all_labeles)
        train_all_outputs = np.concatenate(train_all_outputs)

        calculate_metrics, cm = calc_stats(
            target_classes=train_all_labeles, 
            calced_logits=train_all_outputs, 
            dataset_type="train",
            logits=cfg.metrics.logits
        )
        fig = plot_confusion_matrix(cm, epoch)
        
        os.makedirs(os.path.join(cfg.logger.log_dir, "confusion_matrix", "train"), exist_ok=True)
        fig.savefig(os.path.join(cfg.logger.log_dir, "confusion_matrix", "train", f"epoch_{epoch}.jpg"))
        writer.add_figure('Confusion Matrix/train', fig, global_step=epoch)

        all_metrics.update(calculate_metrics)
        all_metrics.update({"loss/train": all_train_loss})
        
        model.eval()
        with torch.no_grad():
            for val_dataset_type in val_loader:
                
                all_val_loss = 0
                val_all_labeles = []
                val_all_outputs = []
                
                for data_val in tqdm(val_loader[val_dataset_type]):
                    input_val, label_val = data_val
                    input_val = tokenizer(input_val[0], input_val[1], padding=True, truncation=True, return_tensors="pt")
                    input_val, label_val = {key: value.squeeze().to(device) for key, value in input_val.items()}, label_val.to(device)  
                    
                    if cfg.trainer.fp16:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            pred_val = model(**input_val).logits
                            val_loss = criterion(pred_val, label_val)
                    else:
                        pred_val = model(**input_val).logits
                        val_loss = criterion(pred_val, label_val)

                    all_val_loss += val_loss.item()
                    val_all_labeles.append(label_val.cpu().detach().numpy())
                    val_all_outputs.append(pred_val.cpu().detach().numpy())
                
                val_all_labeles = np.concatenate(val_all_labeles)
                val_all_outputs = np.concatenate(val_all_outputs)

                calculate_metrics_val, cm_val = calc_stats(
                    target_classes=val_all_labeles, 
                    calced_logits=val_all_outputs, 
                    dataset_type=val_dataset_type,
                    logits=cfg.metrics.logits
                )
                fig_val = plot_confusion_matrix(cm_val, epoch)
        
                os.makedirs(os.path.join(cfg.logger.log_dir, "confusion_matrix", val_dataset_type), exist_ok=True)
                fig_val.savefig(os.path.join(cfg.logger.log_dir, "confusion_matrix", val_dataset_type, f"epoch_{epoch}.jpg"))
                writer.add_figure(f'Confusion Matrix/{val_dataset_type}', fig_val, global_step=epoch)

                all_metrics.update(calculate_metrics_val)
                all_metrics.update({f"loss/{val_dataset_type}": all_val_loss})

        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        for metric_name in all_metrics:
            writer.add_scalar(metric_name, all_metrics[metric_name], epoch)

        if "scheduler" in cfg:
            scheduler.step(all_metrics[cfg.trainer.monitor_metric[0]])

        os.makedirs(f"{os.path.dirname(cfg.logger.log_dir)}/checkpoints/", exist_ok=True)
        save = False
        for metric in cfg.trainer.monitor_metric:
            if cfg.trainer.save_all:
                save = True
            if all_metrics[metric] > best_monitor_metric[metric]:
                best_monitor_metric[metric] = all_metrics[metric]
                save = True
        if save:
            torch.save(
                {"epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                f"{os.path.dirname(cfg.logger.log_dir)}/checkpoints/checkpoint_{epoch}.pth",
            )
            print(f"Model saved in {epoch} epoch")

    writer.close()
    

if __name__ == "__main__":
    train()
