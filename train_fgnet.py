# Source: https://blog.csdn.net/Bro_Jun/article/details/119519031, 
#         https://blog.csdn.net/qq_43219379/article/details/123381194
#         https://github.com/NICE-FUTURE/predict-gender-and-age-from-camera/blob/master/models.py
#         https://blog.csdn.net/john_bh/article/details/107731443

import torch
import torch.nn as nn
import numpy as np
import clip
import argparse
import json

from torch.utils.data import DataLoader
from datasets.Dataset import build_dataset
from models.Networks import build_model

from utils.log import get_logger_fgnet
from utils.utils import seed_torch, _init_fn
from utils.utils import testing, training
from utils.utils import LR_schedule, Get_cfg

 
def main(logger):
     
    #Argument
    parser = argparse.ArgumentParser(description="AGMixer")
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--test', action='store_true', help='Testing mode')
    parser.add_argument('--fin', action='store_true', help='Fintune mode')
    parser.add_argument('--split', type=int, default=0, help='data split')
    args = parser.parse_args()
    
    # Parameters
    cfg_train, cfg_test, cfg_fin = Get_cfg("config/config_fgnet.yaml")
    bs_train = cfg_train['batch_size']
    num_epochs = cfg_train['num_epochs']
    lr = float(cfg_train['learning_rate'])
    lr_farl = float(cfg_train['lr_farl'])
    lr_AGM = float(cfg_train['lr_AGM'])
    lr_decay_rate = float(cfg_train['lr_decay_rate'])
    early_stop = cfg_train['early_stop']

    seed = cfg_train['seed']
    dataset = cfg_train['dataset']['name']
    bs_test = cfg_test['batch_size']
    split = args.split
    

    if logger != None:
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Train BS: {bs_train}")
        logger.info(f"Test BS: {bs_test}")
        logger.info(f"Num Epochs: {num_epochs}")
        logger.info(f"lr: {lr}")
        logger.info(f"lr_farl: {lr_farl}")
        logger.info(f"lr_AGM: {lr_AGM}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Split: {split}")
        
    print(f"Dataset Split {split}")
        
    if logger != None:
        logger.info(f"Dataset Split {split}")
        
    start_epoch = -1
                   
    seed_torch(seed)

    # Train on cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    if logger != None:
        logger.info(f"Training on {device}")
    _, preprocess = clip.load("ViT-B/16", device="cpu")
    
    # Freeze Layers
    model = build_model(dataset)
    model = model.to(device)
    for name, param in model.FeatureExtractor.named_parameters():
        param.requires_grad = False
        if 'transformer' in name:
            if int(name.split('.')[2]) > 5:
                param.requires_grad = True
        elif 'ln_post' in name:
                param.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.age_classifier1.parameters()}, 
        {'params': model.gender_classifier.parameters()},
        {'params': model.age_classifier2.parameters()},
        {'params': model.AGmixer.parameters(), 'lr':lr_AGM}, 
        {'params': model.FeatureExtractor.parameters(), 'lr':lr_farl},
        {'params': model.proj.parameters()}],
        lr=lr)  # weight decay default 0.01
   
    min_loss = np.inf
    min_mae = np.inf
    max_cs = 0
    max_acc = 0
    cnt = 0

    # Dataset
    print("Training Dataset is "+dataset)
    if logger != None:
        logger.info("Training Dataset is "+dataset)
    dataset_path = cfg_train['dataset']['data_path']
    annotation_path = cfg_train['dataset']['annotation_path']
    ds_train = build_dataset(dataset_path, annotation_path, 'train', transform=preprocess, dataset=dataset, split=split)
    ds_val = build_dataset(dataset_path, annotation_path, 'val', transform=preprocess, dataset=dataset, split=split)
    
    dataloader_train = DataLoader(ds_train, batch_size=bs_train, shuffle=True, num_workers=8, worker_init_fn=_init_fn, pin_memory=True)
    dataloader_val = DataLoader(ds_val, batch_size=bs_test, shuffle=True, num_workers=8, worker_init_fn=_init_fn, pin_memory=True)
   

    for epoch in range(start_epoch+1, num_epochs):

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        if logger != None:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
        
        LR_schedule(args, optimizer, cfg_train, cfg_fin, epoch, lr_decay_rate, logger)
    
        train_loss, train_mae, _ = training(cfg_train, model, dataloader_train, optimizer, device, logger)

        test_mae, test_CS, Acc = testing(cfg_test, model, dataloader_val, device, logger)
        
        if train_loss < min_loss or test_mae < min_mae:
            if train_loss < min_loss:
                print("Update loss...")
                if logger != None:
                    logger.info("Update loss...")
                min_loss = train_loss
            if test_mae < min_mae:
                cnt = 0
                print("Update mae...")
                if logger != None:
                    logger.info("Update mae...")
                min_mae = test_mae
                max_cs = test_CS
                max_acc = Acc
  
        cnt += 1
        
        if epoch+1 == num_epochs or cnt > early_stop:
            r_name = "/mnt/d/chinghsien/Downloads/NTU_Thesis_Code_backup/SmallExp/checkpoints/fgnet/LOPO_result.json"
            new_data_list = []
            j_file = open(r_name, 'r') 
            try:
                j_result = json.load(j_file)
            except json.JSONDecodeError as e:
                j_result = []
            for data in j_result:
                new_data_list.append(data)
            result = {"split": split, "test_mse": min_mae, "test_CS": max_cs, "Acc": max_acc}
            new_data_list.append(result)
            with open(r_name, 'w') as file:
                json.dump(new_data_list, file, indent=4)
            exit()


if __name__ == "__main__":
    logger = get_logger_fgnet("train.log")
    main(logger)