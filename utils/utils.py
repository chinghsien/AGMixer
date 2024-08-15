import cv2 as cv
import numpy as np
import torch, random
import yaml


from typing import List, Dict, Tuple
from tqdm import tqdm
from utils.order_loss import order_loss, iorder_loss


# Source: https://github.com/paplhjak/Facial-Age-Estimation-Benchmark/blob/main/lib/utils.py
def crop_image(img: np.ndarray,
               bbox: List[int],
               out_size: Tuple[int],
               margin: Tuple[float] = (0, 0),
               one_based_bbox: bool = True):
    """
        Crop subimage around bounding box extended by a margin.

    Input:
     img 
     bbox = [A_col,A_row,B_col,B_row,C_col,C_row,D_col,D_row] bounding box
     out_size (cols,rows) size of output image
     margin 
     one_based_bbox [bool] if True assumes bbox to be given on 1-base coordinates
    Output:
     dst: output image [numpy array]
     M: affine transformation used for the crop

    Args:
        img (np.ndarray): Input image.
        bbox (List[int]): [A_col,A_row,B_col,B_row,C_col,C_row,D_col,D_row] bounding box, see README for more information.
        out_size (Tuple[int]): (cols,rows) size of output image.
        margin (Tuple[float], optional): (horizontal, vertical) margin; portion of bonding box size by which to extend the specified bounding box. Defaults to (0, 0).
        one_based_bbox (bool, optional): If True assumes that the bbox is given on coordinates starting with 1 instead of 0. Defaults to True.

    Returns:
        _type_: _description_
    """

    A = np.float32([bbox[0], bbox[1]])
    B = np.float32([bbox[2], bbox[3]])
    C = np.float32([bbox[4], bbox[5]])
    D = np.float32([bbox[6], bbox[7]])

    if one_based_bbox:
        A = A - 1
        B = B - 1
        C = C - 1
        D = D - 1

    ext_A = A + (A-B)*margin[0] + (A-D)*margin[1]
    ext_B = B + (B-A)*margin[0] + (B-C)*margin[1]
    ext_C = C + (C-D)*margin[0] + (C-B)*margin[1]

    pts1 = np.float32([ext_A, ext_B, ext_C])
    pts2 = np.float32([[0, 0], [out_size[0]-1, 0],
                      [out_size[0]-1, out_size[1]-1]])

    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(img, M, (out_size[0], out_size[1]))

    return dst, M

def seed_torch(seed=8299):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    seed = 42
    np.random.seed(int(seed)+worker_id)
    
def Age_EV(age_out, age_min, age_max, device):
    
    age_prob = torch.nn.functional.softmax(age_out, dim=1)
    labels = torch.arange(age_min, age_max+1).to(device)
    age_expected = torch.sum((age_prob * labels), dim=1)
    
    return age_expected
    
def Cal_Acc(gender_out, gender_batch):
    
    g_pred = torch.max(torch.nn.functional.softmax(gender_out, dim=1), dim=1)[1]
    g_pred = g_pred.cpu().numpy().squeeze()
    g_target = gender_batch.cpu().numpy().squeeze()
    result = (g_pred ==  g_target)
    if isinstance(result, np.bool_):
        acc_t = 1 if result == True else 0
    else:
        acc_t = sum(result)
    
    return acc_t

def Criterion_cal(output, target_batch, age_min, age_max, device, class_weight=None):

    # Assign Value
    age_out_1, gender_out, age_out_2, features1, features2 = output
    age_batch, gender_batch = target_batch
        
    # Loss Functions
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    ce_age_loss = torch.nn.CrossEntropyLoss() #weight=class_weight
    ce_gender_loss = torch.nn.CrossEntropyLoss()
    order_age_loss = iorder_loss
    
    # Target Batch Processing
    age_batch = age_batch.view(-1, 1) # expand to 1 row
    age_batch = age_batch.to(torch.float32)
    age_batch_for_oloss = age_batch
    
    if age_batch.shape[0]==1:
        age_batch = age_batch.squeeze()
        age_batch = age_batch.unsqueeze(dim=0)
    else:
        age_batch = age_batch.squeeze()
       
    age_real_batch = age_batch + age_min

    age_batch = age_batch.to(torch.int64)
    
    gender_batch = gender_batch.view(-1, 1) # expand to 1 row
    if gender_batch.shape[0] == 1:
        gender_batch = gender_batch.squeeze()
        gender_batch = gender_batch.unsqueeze(dim=0)
    else:
        gender_batch = gender_batch.squeeze()
    gender_batch = gender_batch.to(torch.int64)
    
    # Regressor Age Processing
    if age_out_1.shape[0] == 1:
        age_out_1 = age_out_1.squeeze().unsqueeze(dim=0)
    else:
        age_out_1 = age_out_1.squeeze() # for regressor
    age_max_tensor = torch.tensor(age_max, device=device, dtype=age_out_1.dtype)
    age_min_tensor = torch.tensor(age_min, device=device, dtype=age_out_1.dtype)
    age1 = age_out_1.mul(age_max_tensor).add(age_min_tensor)
           
    # Classification Age Processing
    age2 = Age_EV(age_out_2, age_min, age_max, device)
    
    # Final Age Calculating
    final_age = age2
    
    # Cummulativa Score
    diff = np.abs((final_age.cpu() - age_real_batch.cpu()).detach().numpy())
    cnt_diff = sum(diff <= 5)
     
    # Calculating Loss and Other criterion
    gender_loss = ce_gender_loss(gender_out, gender_batch)
    
    # Order loss
    oloss = torch.tensor(0.0).to(device)
     
    l1_w = 0.5 
    loss1 = torch.sqrt(mse_loss(age1, age_real_batch)) # RMSE

    l2_w = 1
    oloss = order_age_loss(features2, age_batch_for_oloss)
    loss2 = ce_age_loss(age_out_2, age_batch) + oloss
 
    loss = l1_w * loss1 + l2_w * loss2 + gender_loss
    
    mae = mae_loss(final_age, age_real_batch)
    mae1 = mae_loss(age1, age_real_batch)
    mae2 = mae_loss(age2, age_real_batch)
        
    # Gender Acc Calculating
    acc_t = Cal_Acc(gender_out, gender_batch)
    
    return loss, mae, loss1, loss2, mae1, mae2, gender_loss, acc_t, cnt_diff, oloss

def Get_age_bound(dataset):
    if dataset.split("-")[0] == 'utk':
        age_min = 1
        age_max = 116
    elif dataset.split("-")[0] == 'imdb':
        age_min = 1
        age_max = 95
    elif dataset.split("-")[0] == 'afad':
        age_min = 15
        age_max = 72
    elif dataset.split("-")[0] == 'cacd':
        age_min = 14
        age_max = 62
    elif dataset.split("-")[0] == 'agedb':
        age_min = 1
        age_max = 101
    elif dataset.split("-")[0] == 'fgnet':
        age_min = 0
        age_max = 69
    elif dataset.split("-")[0] == 'clap':
        age_min = 1
        age_max = 96
    return age_min, age_max

def testing(cfg_test, model, dataloader, device, logger):
    
    print("======================================================================")
    print("Testing...") 
    if logger != None:
        logger.info("======================================================================")
        logger.info("Testing...") 
    
    dataset = cfg_test['dataset']['name']
 
    # Get Age Bound
    age_min, age_max = Get_age_bound(dataset)
     
    # Initialize
    test_loss = 0.0
    test_mae = 0.0
    test_mae1 = 0.0
    test_mae2 = 0.0
    test_ce_g = 0.0
    test_oloss = 0.0
    
    test_loss1 = 0.0 
    test_loss2 = 0.0

    total_cnt_g = 0.0
    total_cnt_diff = 0.0
    total_label_cnt = 0.0
    
    for _, sample_batched in enumerate(tqdm(dataloader)):
        
        model.eval()
        img = sample_batched['image'].to(device)
        gender_batch = sample_batched['gender'].to(device)
        age_batch = sample_batched['age'].to(device)
        
        with torch.no_grad():
            output = model(img)    
        
        # Get Criterion Result
        loss, mae, loss1, loss2, mae1, mae2, ce_g, acc_t, cnt_diff, oloss = Criterion_cal(output, (age_batch, gender_batch), age_min, age_max, device)
 
        # CS@5 Calculating
        total_cnt_diff = total_cnt_diff + cnt_diff
 
        # Gender Acc Calculating
        total_cnt_g = total_cnt_g + acc_t
        total_label_cnt = total_label_cnt + len(gender_batch)
        
        # update testing loss
        test_loss += loss.item()
        test_loss1 += loss1.item()
        test_loss2 += loss2.item()
        test_oloss += oloss.item()
        test_mae += mae.item()
        test_mae1 += mae1.item()
        test_mae2 += mae2.item()
        test_ce_g += ce_g.item()
                       
    # Calculating Average Loss
    test_loss = test_loss / len(dataloader)
    test_loss1 = test_loss1 / len(dataloader)
    test_loss2 = test_loss2 / len(dataloader)
    test_oloss = test_oloss / len(dataloader)
    test_mae = test_mae / len(dataloader)
    test_mae1 = test_mae1 / len(dataloader)
    test_mae2 = test_mae2 / len(dataloader)
    
    test_ce_g = test_ce_g / len(dataloader)

    Acc = total_cnt_g/total_label_cnt
    cs_5 = total_cnt_diff / total_label_cnt
    

    print(f"\033[32mTesting MAE: {np.round(test_mae, 3)}, \033[36mCS@5: {np.round(cs_5, 4)},\033[0m SexAcc: {np.round(Acc, 4)}")
    print(f"MAE1 : {np.round(test_mae1, 3)}, MAE2: {np.round(test_mae2, 3)}")  
    print(f"Testing Loss: {np.round(test_loss, 3)}, Loss1: {np.round(test_loss1, 3)}, Loss2: {np.round(test_loss2, 3)}, OLoss: {np.round(test_oloss, 3)}, G_CE: {np.round(test_ce_g, 3)}") 
    print("======================================================================")
    
    if logger != None:
        logger.info(f"Testing MAE: {np.round(test_mae, 3)}, CS@5: {np.round(cs_5, 4)}, SexAcc: {np.round(Acc, 4)}")
        logger.info(f"MAE1 : {np.round(test_mae1, 3)}, MAE2: {np.round(test_mae2, 3)}")  
        logger.info(f"Testing Loss: {np.round(test_loss, 3)}, Loss1: {np.round(test_loss1, 3)}, Loss2: {np.round(test_loss2, 3)}, OLoss: {np.round(test_oloss, 3)}, G_CE: {np.round(test_ce_g, 3)}") 
        logger.info("======================================================================")
    
    return test_mae, cs_5, Acc

def training(cfg_train, model, dataloader, optimizer, device, logger, class_weight=None):
    
    dataset = cfg_train['dataset']['name']
 
    # Get Age Bound
    age_min, age_max = Get_age_bound(dataset)
     
    # Initialize
    train_loss = 0.0
    train_mae = 0.0
    train_mae1 = 0.0
    train_mae2 = 0.0
    train_ce_g = 0.0
    
    train_loss1 = 0.0 
    train_loss2 = 0.0
    train_oloss = 0.0

    total_cnt_g = 0.0
    total_cnt_diff = 0.0
    total_label_cnt = 0.0
    
    print(f"lr_farl: {optimizer.param_groups[4]['lr']}, lr_AGM: {optimizer.param_groups[3]['lr']}, lr_others: {optimizer.param_groups[0]['lr']}")
    if logger != None:
        logger.info(f"lr_farl: {optimizer.param_groups[4]['lr']}, lr_AGM: {optimizer.param_groups[3]['lr']}, lr_others: {optimizer.param_groups[0]['lr']}")
    
    for _, sample_batched in enumerate(tqdm(dataloader)):
            # Train mode
            model.train()
            img = sample_batched['image'].to(device)
            age_batch = sample_batched['age'].to(device)
            gender_batch = sample_batched['gender'].to(device)

            optimizer.zero_grad()

            # # forward pass: compute predicted outputs by passing inputs to the model
            output  = model(img)

            # Calculate batch loss
            loss, mae, loss1, loss2, mae1, mae2, ce_g, acc_t, cnt_diff, oloss =  Criterion_cal(output, (age_batch, gender_batch), age_min, age_max, device, class_weight)
            
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # update training loss
            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_oloss += oloss.item()
            train_mae += mae.item()
            train_mae1 += mae1.item()
            train_mae2 += mae2.item()
 
            train_ce_g += ce_g.item()
            
            # CS@5 Calculating
            total_cnt_diff = total_cnt_diff + cnt_diff

            # Gender Acc Calculating
            total_cnt_g = total_cnt_g + acc_t
            total_label_cnt = total_label_cnt + len(gender_batch)
            
    
    # Calculating Average Loss of a epoch
    train_loss = train_loss / len(dataloader)
    train_loss1 = train_loss1 / len(dataloader)
    train_loss2 = train_loss2 / len(dataloader)
    train_oloss = train_oloss / len(dataloader)
    train_mae = train_mae / len(dataloader)
    train_mae1 = train_mae1 / len(dataloader)
    train_mae2 = train_mae2 / len(dataloader)
    
    train_ce_g = train_ce_g / len(dataloader)

    Acc = total_cnt_g/total_label_cnt
    cs_5 = total_cnt_diff / total_label_cnt

    print('\033[33mTraining MAE: {:.6f} \033[0m\tCS@5: {:.4f} \tSexAcc: {:.4f}'.format(train_mae, cs_5, Acc))
    print('MAE1: {:.6f} \tMAE2: {:.6f}'.format(train_mae1, train_mae2))   
    print('Training Loss: {:.6f} \tTraining Loss1: {:.6f} \tTraining Loss2: {:.6f} \tTraining OLoss: {:.6f} \tGender CE: {:.6f}'.format(train_loss, train_loss1, train_loss2, train_oloss, train_ce_g)) 
    
    if logger != None:
        logger.info('Training MAE: {:.6f} \tCS@5: {:.4f} \tSexAcc: {:.4f}'.format(train_mae, cs_5, Acc))
        logger.info('MAE1: {:.6f} \tMAE2: {:.6f}'.format(train_mae1, train_mae2))  
        logger.info('Training Loss: {:.6f} \tTraining Loss1: {:.6f} \tTraining Loss2: {:.6f} \tTraining OLoss: {:.6f} \tGender CE: {:.6f}'.format(train_loss, train_loss1, train_loss2, train_oloss, train_ce_g)) 
      
    return train_loss, train_mae, Acc

def LR_schedule(args, optimizer, cfg_train, cfg_fin, epoch, decay_rate, logger):
    if args.fin == True:
        lr = float(cfg_fin['learning_rate'])
        lr_farl = float(cfg_fin['lr_farl'])
        lr_AGM = float(cfg_fin['lr_AGM'])
    else:
        lr = float(cfg_train['learning_rate'])
        lr_farl = float(cfg_train['lr_farl'])
        lr_AGM = float(cfg_train['lr_AGM'])
    print("\033[35mUpdating Learning Rate...\033[0m")
    if logger != None:
        logger.info("Updating Learning Rate...")
    optimizer.param_groups[0]['lr'] = lr / (1 +  decay_rate*(epoch))       # age_predictor or age_regressor1
    optimizer.param_groups[1]['lr'] = lr / (1 +  decay_rate*(epoch))       # gender_classifier
    optimizer.param_groups[2]['lr'] = lr / (1 +  decay_rate*(epoch))       # age_predictor2 or age_regressor2
    optimizer.param_groups[3]['lr'] = lr_AGM / (1 +  decay_rate*(epoch))   # AGM
    optimizer.param_groups[4]['lr'] = lr_farl / (1 +  decay_rate*(epoch))  # farl
    optimizer.param_groups[5]['lr'] = lr / (1+ decay_rate*(epoch))         # proj

def Get_cfg(cfg_path="config/config.yaml"):
    with open(cfg_path, 'r') as file:
        cfg_file = yaml.safe_load(file)
    cfg_train = cfg_file['training']
    cfg_test = cfg_file['testing']
    cfg_fin = cfg_file['fintune']
    return cfg_train, cfg_test, cfg_fin