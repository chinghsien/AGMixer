# Reference: https://blog.csdn.net/Bro_Jun/article/details/119519031, 
#            https://blog.csdn.net/qq_43219379/article/details/123381194
#            https://github.com/NICE-FUTURE/predict-gender-and-age-from-camera/blob/master/models.py
#            https://blog.csdn.net/john_bh/article/details/107731443

import torch
import torch.nn as nn
import numpy as np
import clip
import argparse, wandb

from torch.utils.data import DataLoader
from datasets.Dataset import build_dataset
from models.Networks import build_model

from utils.log import get_logger
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
    cfg_train, cfg_test, cfg_fin = Get_cfg()
    bs_train = cfg_train['batch_size']
    num_epochs = cfg_train['num_epochs']
    lr = float(cfg_train['learning_rate'])
    lr_farl = float(cfg_train['lr_farl'])
    lr_AGM = float(cfg_train['lr_AGM'])
    lr_decay_rate = float(cfg_train['lr_decay_rate'])
    early_stop = cfg_train['early_stop']

    seed = cfg_train['seed']
    dataset = cfg_train['dataset']['name']
    dataset_test = cfg_test['dataset']['name']
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
        
    train_resume = args.resume
    start_epoch = -1
             
    seed_torch(seed)

    # Train on cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    if logger != None:
        logger.info(f"Training on {device}")
    _, preprocess = clip.load("ViT-B/16", device="cpu")
    
    # Freeze Layers
    if args.fin == True:
        model = build_model('imdb')
        ckpt_path = cfg_fin['ckpt']
        ckpt = torch.load(ckpt_path)     
        model.load_state_dict(ckpt['model_state_dict'])
        
        lr = float(cfg_fin['learning_rate'])
        lr_farl = float(cfg_fin['lr_farl'])
        lr_AGM = float(cfg_fin['lr_AGM'])
            
        print("Fintuning..., change learning rate")
        print(f"lr_fin: {lr}")
        print(f"lr_farl_fin: {lr_farl}")
        print(f"lr_AGM_fin: {lr_AGM}")
        if logger != None:
            logger.info("Fintuning..., change learning rate")
            logger.info(f"lr_fin: {lr}")
            logger.info(f"lr_farl_fin: {lr_farl}")
            logger.info(f"lr_AGM_fin: {lr_AGM}")
        
        if dataset == "cacd":
            model.age_classifier2 = nn.Linear(128, 49)
        else:
            print("Only CACD2000 would finetune!")
            exit()
        model = model.to(device)

        for name, param in model.FeatureExtractor.named_parameters():
            param.requires_grad = False

    else:
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
    cnt = 0
    
    if train_resume:
        print("Resume...")
        if logger != None:
            logger.info("Resume...")
        ckpt_path = cfg_train['ckpt']
        ckpt = torch.load(ckpt_path)

        model.load_state_dict(ckpt['model_state_dict'])
        
        optimizer.load_state_dict(ckpt['optimizer_state_dict']) 
    
        start_epoch = ckpt['epoch'] 
        print(f"Resume training from epoch {start_epoch+1}")
        if logger != None:
            logger.info(f"Resume training from epoch {start_epoch+1}")
               
    if args.test == True:
        print("Loading ckpt...")
        ckpt_path = cfg_test["ckpt"]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        
        # Dataset
        print("Testing Dataset is "+ dataset_test)
        if logger != None:
            logger.info("Testing Dataset is "+ dataset_test)
        dataset_path = cfg_test['dataset']['data_path']
        annotation_path = cfg_test['dataset']['annotation_path']
        ds_test = build_dataset(dataset_path, annotation_path, 'test', transform=preprocess, dataset=dataset_test, split=split)
        dataloader_test = DataLoader(ds_test, batch_size=bs_test, shuffle=True, num_workers=8, worker_init_fn=_init_fn, pin_memory=True)   
        
        test_mae, test_CS, Acc = testing(cfg_test, model, dataloader_test, device, logger)
        exit()

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
        
        save_path = cfg_train['ckpt_save_path'] + "split{}/{}Epochs_AgeGender_MixerV1_FaRL_64epckpt_loss{:.6f}_TrainMae{:.3f}_ValMae{:.3f}_CS@5_{:.4f}_GAcc_{:.4f}.pt".format(split, epoch+1, train_loss, train_mae, test_mae, test_CS, Acc)
        
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
                if epoch+1 > 1:
                    print("Save model...")
                    if logger != None:
                        logger.info("Save model...")
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    }, save_path)
        cnt += 1
        
        if cnt > early_stop:
            print("Loss didn't improve...")
            print("Save model...")
            if logger != None:
                logger.info("Loss didn't improve...")
                logger.info("Save model...")
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, save_path)
            exit()

if __name__ == "__main__":
    logger = get_logger("train.log")
    main(logger)