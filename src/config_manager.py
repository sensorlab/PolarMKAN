import os
import json
import copy

import torch
import numpy as np
import wandb
import pywt
from torch import nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torchvision import transforms

from src.datasets import OracleDataset, WiSig_Dataset_ManySig, LoRaDataset, WiSig_Dataset_ManyTx
from src.architectures import side_networks
from src.architectures.features_extractors.cnn import Simple_CNN_1D, AE_CNN_1D, AE_CNN_2D, Simple_CNN_2D
from src.architectures.features_extractors.transformers import TS_Transformer
from src.architectures.features_extractors.cnn_lstm import CNN_LSTM
from src.architectures.features_extractors.kan import Encoder as KANS_Encoder
from src.architectures.features_extractors.kan import Autoencoder as KANS_AE
from src.architectures.features_extractors.mkan import Autoencoder as MKANS_AE
from src.architectures.features_extractors.mkan import Encoder as MKANS_Encoder

from src.architectures.features_extractors.cnn_transformer import CNN_Transformer
from src.architectures.features_extractors.resnet1d import ResNet1D
from src.architectures.features_extractors.resnet2d import ResNet2D
from src.architectures.features_extractors.vit import Vit_14
from src.trainers import (
    SIM_CLR_Trainer,
    Deep_Clustering_Trainer,
    AE_Trainer,
    PCA_Trainer,
)

import random

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

DATASETS = {"WiSig_ManyTx": WiSig_Dataset_ManyTx, 'WiSig_ManySig': WiSig_Dataset_ManySig  ,"LoRa": LoRaDataset, "Oracle": OracleDataset}

FEATURES_EXCTRACTORS = {
    "Simple_CNN_1D": Simple_CNN_1D,
    "ResNet_1D": ResNet1D,
    "ResNet_2D": ResNet2D,
    "CNN_Trasnformer": CNN_Transformer,
    "AE_KAN": KANS_AE,
    "AE_CNN_1D": AE_CNN_1D,
    'AE_CNN_2D': AE_CNN_2D,
    "CNN_LSTM": CNN_LSTM,
    "Vit": Vit_14,
    'KANS_Encoder': KANS_Encoder,
    'Simple_CNN_2D': Simple_CNN_2D,
    'AE_MKAN': MKANS_AE, 
    'MKANS_Encoder': MKANS_Encoder
    
}

def report(metrics, trainer, exp_config, fold, iteration, epoch, train_config, test_config):
    """
    Save metrics, configs, and model checkpoints to logs and report to wandb.
    """
    log_dir = os.path.join(
        exp_config["logs_dir"],
        f"{exp_config['exp_id']}/{exp_config['dataset']['name']}/{fold}/"
        f"{exp_config['approach']['name']}/{exp_config['feature_extractor']['name']}/{exp_config['params_set']}/"
        f"{iteration}/{epoch}"
    )
    os.makedirs(log_dir, exist_ok=True)

    trainer.save_checkpoint(os.path.join(log_dir, "models.pt"))

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=4)
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    wandb.log(metrics, step=epoch)
    
def get_optimizer(models, name, config):
    optimizers = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }
    return optimizers[name](models.parameters(), **config)


def get_data_configs(dataset_config):
    """
    Return training and testing config sets along with unknown device IDs for k-fold or static splits.
    """
    test_specs = dataset_config.pop('test_config')
    train_specs = dataset_config.pop('train_config')
    
    if "k_fold" in dataset_config:
        ratio = dataset_config["k_fold"]["ratio"]
        total_devices = dataset_config["k_fold"]["total_devices"]
        test_size = total_devices // ratio

        test_devices = tuple(np.arange(total_devices))
        test_config = copy.deepcopy(dataset_config)
        test_config.pop("k_fold")
        test_config["devices"] = test_devices
        test_config = test_config | test_specs
        test_configs = [test_config] * ratio

        train_configs = []
        unknown_devices_folds = []

        for i in range(ratio):
            train_devices = tuple(
                list(np.arange(i * test_size)) + list(np.arange((i + 1) * test_size, total_devices))
            )
            unknown_devices = np.arange(i * test_size, (i + 1) * test_size)
            unknown_devices_folds.append(unknown_devices)

            train_config = copy.deepcopy(dataset_config)
            train_config.pop("k_fold")
            train_config["devices"] = train_devices
            train_config = train_config | train_specs
            train_configs.append(train_config)
        return train_configs, test_configs, unknown_devices_folds

    # Static train/test split
    train_cfg = copy.deepcopy(dataset_config) | train_specs
    test_cfg = copy.deepcopy(dataset_config) | train_specs

    unknown_devices = [d for d in test_cfg["devices"] if d not in train_cfg["devices"]]
    return [train_cfg], [test_cfg], [unknown_devices]

def evauate_config(exp_config: dict):
    """
    Train and evaluate models across k-folds or static train/test splits.
    """
    train_cfgs, test_cfgs, unknown_folds = get_data_configs(exp_config["dataset"]["config"])
    dataset_cls = DATASETS[exp_config["dataset"]["name"]]
    report_interval = exp_config["report_interval"]
    starting_fold = exp_config.get("starting_fold",0)
    for fold, (train_cfg, test_cfg, unknown_devices) in enumerate(zip(train_cfgs, test_cfgs, unknown_folds)):
        if fold < starting_fold:
            continue

        for iteration in range(exp_config["starting_iteration"], exp_config["num_iterations"]):
            train_set = dataset_cls(**train_cfg)
            test_set = dataset_cls(**test_cfg)

            targets = np.array([test_set[i][1] in unknown_devices for i in range(len(test_set))])
            train_loader = DataLoader(train_set, **exp_config["train_loader"])
            test_loader = DataLoader(test_set, **exp_config["test_loader"])
            
            trainer = get_trainer(exp_config)

            wandb.init(
                project=f"{exp_config['dataset']['name']}_evaluations",
                config={**exp_config, **train_cfg, **test_cfg, "iteration": iteration, "fold": fold},
                name=f"fold_{fold}, iter_{iteration}, approach_{exp_config['approach']['name']}, f_extractor_{exp_config['feature_extractor']['name']}",
            )

            for epoch in range(trainer.num_epochs):
                loss = trainer.train_epoch(train_loader)

                if epoch % report_interval == 0:
                    metrics = trainer.evaluate(test_loader=test_loader, train_loader=train_loader,
                                               targets=targets, clusters_numbers=exp_config["evaluation"]["clusters_numbers"])
                    metrics["loss"] = loss
                    report(metrics, trainer, exp_config, fold, iteration, epoch, train_cfg, test_cfg)

            wandb.finish()


def get_trainer(exp_config: dict):
    """
    Build and return the trainer object according to the experiment config.
    """
    approach = exp_config["approach"]["name"]
    approach_cfg = exp_config["approach"]["config"]


    if approach == "PCA":
        return PCA_Trainer(**approach_cfg)
        
    trainer_cfg = exp_config["approach"]["trainer"]

    extractor_name = exp_config["feature_extractor"]["name"]
    extractor_cfg = exp_config["feature_extractor"]["config"]
    extractor = FEATURES_EXCTRACTORS[extractor_name](**extractor_cfg)

    optimizer_name = trainer_cfg["main_optimizer"]["name"]
    optimizer_cfg = trainer_cfg["main_optimizer"]["config"]

    if approach == "Sim_CLR":
        mlp = side_networks.Mlp(**trainer_cfg["mlp_head"])
        models = nn.ModuleDict({"feature_extractor": extractor, "mlp_instance": mlp})
        optimizers = {"main_optimizer": get_optimizer(models, optimizer_name, optimizer_cfg)}

        models["augs"] = side_networks.get_augmentations(
            **trainer_cfg["augmentations"], type=approach_cfg["augs_type"]
        )

        if approach_cfg["augs_type"] == 'learnable':
            optimizers["augs_optimizer"] = get_optimizer(
                models["augs"],
                trainer_cfg["augs_optimizer"]["name"],
                trainer_cfg["augs_optimizer"]["config"],
            )
        
        if approach_cfg.get("large_augs"):
            models["large_augs"] = side_networks.get_augmentations(
                **trainer_cfg["large_augmentations"], type="large_augs"
            )
            optimizers["large_augs_optimizer"] = get_optimizer(
                models["large_augs"],
                trainer_cfg["large_augs_optimizer"]["name"],
                trainer_cfg["large_augs_optimizer"]["config"],
            )

        if approach_cfg.get("clusters_loss"):
            models["mlp_cluster"] = side_networks.Mlp(**trainer_cfg["mlp_head"], apply_softmax=True)

        return SIM_CLR_Trainer(models=models, optimizers=optimizers, **approach_cfg)

    if approach == "Deep Clustering":
        models = nn.ModuleDict({"feature_extractor": extractor})
        optimizers = {"main_optimizer": get_optimizer(models, optimizer_name, optimizer_cfg)}
        return Deep_Clustering_Trainer(models=models, optimizers=optimizers, **approach_cfg)


    if approach == "AE":
        models = nn.ModuleDict({"feature_extractor": extractor})
        optimizers = {"main_optimizer": get_optimizer(models, optimizer_name, optimizer_cfg)}
        return AE_Trainer(models=models, optimizers=optimizers, **approach_cfg)


def parse_configs(exp_configs, params):
    """
    Expand configs that have parameter sweeps into multiple individual configs.
    """
    stack = [exp_configs]

    while stack:
        node = stack.pop()
        for k, v in node.items():
            if isinstance(v, list):
                next_configs = []
                next_params = []
                for val in v:
                    node[k] = val
                    cur_config = copy.deepcopy(exp_configs)
                    config, param = parse_configs(cur_config, params + f"{k}_{val}") 
                    next_configs += config
                    next_params += param
                return next_configs, next_params
            if isinstance(v, dict):
                stack.append(v)

    return [exp_configs], [params]
