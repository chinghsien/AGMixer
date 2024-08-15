from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.transforms import functional as TF
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image
import json
from utils.utils import crop_image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')



class DatasetUTKFace(Dataset):
    def __init__(self, img_dir, img_label_dir, transform_farl, mode, split):
        self.transform = transforms.Compose([transforms.ToTensor()])                                  
        self.img_dir = img_dir
        json_data = open(img_label_dir, "r") 
        self.img_labels = json.load(json_data)
        self.age_min = 1
        self.age_max = 116
        self.transform_farl = transform_farl
        self.mode = mode

        # age: 1~116, 0 :male, 1 :female
        self.genders = []
        self.weight_list = []
        
        if split == 0:
            if mode == 'train':
                pool = [0, 1, 2, 3, 4, 5]
            elif mode == 'val':
                pool = [6, 7]
            elif mode == 'test':
                pool = [8, 9]
        elif split == 1:
            if mode == 'train':
                pool = [2, 3, 4, 5, 6, 7]
            elif mode == 'val':
                pool = [8, 9]
            elif mode == 'test':
                pool = [0, 1]
        elif split == 2:
            if mode == 'train':
                pool = [4, 5, 6, 7, 8, 9]
            elif mode == 'val':
                pool = [0, 1]
            elif mode == 'test':
                pool = [2, 3]
        elif split == 3:
            if mode == 'train':
                pool = [5, 6, 7, 8, 9, 0]
            elif mode == 'val':
                pool = [1, 2]
            elif mode == 'test':
                pool = [3, 4]
        elif split == 4:
            if mode == 'train':
                pool = [6, 7, 8, 9, 0, 1]
            elif mode == 'val':
                pool = [2, 3]
            elif mode == 'test':
                pool = [4, 5]
        
        # Assign list
        self.img_path_list = []
        self.age_list = []
        self.gender_list = []
        self.bbox_list = []
        
        for idx, data_list in enumerate(self.img_labels):
            if data_list["folder"] not in pool:
                continue
            age = int(data_list['age'])
            gender = data_list['img_path'].split('/')[2].split('_')[1]
            if gender != str(0) and gender != str(1):
                print(type(gender))
                print(data_list['img_path'].split('/')[2])
                exit()
            gender = int(gender)
            img_path = self.img_dir + data_list['img_path'].split('/', 1)[1]
            aligned_bbox = data_list['aligned_bbox']
            
            self.img_path_list.append(img_path)
            self.age_list.append(age)
            self.gender_list.append(gender)
            self.bbox_list.append(aligned_bbox)
      
        # Calculate weight
        if mode == 'train':
            print("======================================================================")
            self.age_w_list = np.zeros(self.age_max - self.age_min + 1)
            for age in self.age_list:
                self.age_w_list[age-1] = self.age_w_list[age-1] + 1
            for i in range(0, 100, 10):
                print("age {:d} ~ {:d} : {:.0f} images".format((i+1 if i==0 else i), i+9, (sum(self.age_w_list[i:i+9]) if i == 0 else sum(self.age_w_list[i-1:i+9]))))
            print("age upon 100 : {:.0f} images".format(sum(self.age_w_list[99:])))
            print("Total \033[32m{:.0f}\033[0m images in Training Dataset".format(sum(self.age_w_list)))
            print("======================================================================")
            
            self.weight_list = max(self.age_w_list) / (self.age_w_list + 1)
        elif mode == 'val':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Validation Dataset".format(len(self.img_path_list)))
            print("======================================================================")
        elif mode == 'test':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Testing Dataset".format(len(self.img_path_list)))
            print("======================================================================")
        
    def __len__(self):
        return len(self.img_path_list)
    
    def get_weight(self):
        return torch.from_numpy(self.weight_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        age = self.age_list[idx]
        gender = self.gender_list[idx]    # Male: 0, Female:1
        bbox = self.bbox_list[idx]
    
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
        ])

        # img = cv.imread(img_path)
        img = Image.open(img_path)
        img = np.array(img)
        if img is None:
            print(img_path)
            raise FileNotFoundError("Image Not Found")
        out_size = (int(224), int(224))
        img, _ = crop_image(img, bbox, out_size)
        img = Image.fromarray(img)
        if self.mode =='train':
            img = transform_aug(img)
        img = self.transform_farl(img)

        sample = {'image': img, 'age': age-1, 'gender': torch.tensor(gender)}

        return sample
    
class DatasetIMDB(Dataset):
    def __init__(self, img_dir, img_label_dir, transform_farl, mode):
        self.transform = transforms.Compose([transforms.ToTensor()])                                  
        self.img_dir = img_dir
        json_data = open(img_label_dir, "r") 
        self.img_labels = json.load(json_data)
        self.age_min = 1
        self.age_max = 95
        self.transform_farl = transform_farl
        self.mode = mode

        # age: 1~95, 0 :male, 1 :female
        self.genders = []
        self.weight_list = []
        
        if mode == "train":
            pool = [0]
        elif mode == "val":
            pool = [1]
        elif mode == "test":
            pool = [2]
        
        # Assign list
        self.img_path_list = []
        self.age_list = []
        self.gender_list = []
        self.align_bbox_list = []
        self.bbox_list = []
        
        for idx, data_list in enumerate(self.img_labels):
            if data_list["folder"] not in pool:
                continue
            if len(data_list['aligned_bbox']) < 8:
                continue
            age = int(data_list['age'])
            gender = 0 if data_list['gender'] == "M" else 1
            img_path = self.img_dir + data_list['img_path'].split('/', 2)[2]
            aligned_bbox = data_list['aligned_bbox']
            bbox = data_list['bbox']
            
            self.img_path_list.append(img_path)
            self.age_list.append(age)
            self.gender_list.append(gender)
            self.align_bbox_list.append(aligned_bbox)
            self.bbox_list.append(bbox)
        
        # Calculate weight
        if mode == 'train':
            print("======================================================================")
            self.age_w_list = np.zeros(self.age_max - self.age_min + 1)
            for age in self.age_list:
                self.age_w_list[age-1] = self.age_w_list[age-1] + 1
            for i in range(0, 90, 10):
                print("age {:d} ~ {:d} : {:.0f} images".format((i+1 if i==0 else i), i+9, (sum(self.age_w_list[i:i+9]) if i == 0 else sum(self.age_w_list[i-1:i+9]))))
            print("age 90~95  : {:.0f} images".format(sum(self.age_w_list[89:])))
            print("Total \033[32m{:.0f}\033[0m images in Training Dataset".format(sum(self.age_w_list)))
            print("======================================================================")
            
            self.weight_list = max(self.age_w_list) / (self.age_w_list + 1)
        elif mode == 'val':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Validation Dataset".format(len(self.img_path_list)))
            print("======================================================================")
        elif mode == 'test':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Testing Dataset".format(len(self.img_path_list)))
            print("======================================================================")
            
    def __len__(self):
        return len(self.img_path_list)
    
    def get_weight(self):
        return torch.from_numpy(self.weight_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        age = self.age_list[idx]
        gender = self.gender_list[idx]    # Male: 0, Female:1
        align_bbox = self.align_bbox_list[idx]
        bbox = self.bbox_list[idx]
    
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
        ])

        # img = cv.imread(img_path)
        img = Image.open(img_path)
        img = np.array(img)
        if img is None:
            print(img_path)
            raise FileNotFoundError("Image Not Found")
        if align_bbox != []:
            out_size = (int(224), int(224))
            img, _ = crop_image(img, align_bbox, out_size)
        else:
            out_size = (int(224), int(224))
            img, _ = crop_image(img, bbox, out_size)
        img = Image.fromarray(img)
        if self.mode =='train':
            img = transform_aug(img)
        img = self.transform_farl(img)

        sample = {'image': img, 'age': age-1, 'gender': torch.tensor(gender)}

        return sample
   
class DatasetAFAD(Dataset):
    def __init__(self, img_dir, img_label_dir, transform_farl, mode, split):
        self.transform = transforms.Compose([transforms.ToTensor()])                                  
        self.img_dir = img_dir
        json_data = open(img_label_dir, "r") 
        self.img_labels = json.load(json_data)
        self.transform_farl = transform_farl
        self.mode = mode

        # age: 15~72, 0 :male, 1 :female
        self.genders = []
        self.weight_list = []
        
        if split == 0:
            if mode == 'train':
                pool = [0, 1, 2, 3, 4, 5]
            elif mode == 'val':
                pool = [6, 7]
            elif mode == 'test':
                pool = [8, 9]
        elif split == 1:
            if mode == 'train':
                pool = [2, 3, 4, 5, 6, 7]
            elif mode == 'val':
                pool = [8, 9]
            elif mode == 'test':
                pool = [0, 1]
        elif split == 2:
            if mode == 'train':
                pool = [4, 5, 6, 7, 8, 9]
            elif mode == 'val':
                pool = [0, 1]
            elif mode == 'test':
                pool = [2, 3]
        elif split == 3:
            if mode == 'train':
                pool = [5, 6, 7, 8, 9, 0]
            elif mode == 'val':
                pool = [1, 2]
            elif mode == 'test':
                pool = [3, 4]
        elif split == 4:
            if mode == 'train':
                pool = [6, 7, 8, 9, 0, 1]
            elif mode == 'val':
                pool = [2, 3]
            elif mode == 'test':
                pool = [4, 5]
        
        # Assign list
        self.img_path_list = []
        self.age_list = []
        self.gender_list = []
        
        for idx, data_list in enumerate(self.img_labels):
            if data_list["folder"] not in pool:
                continue
            age = int(data_list['age'])
            gender = 0 if data_list['gender'] == 'M' else 1
            if data_list['gender'] == 'U':
                print(type(gender))
                print(data_list['img_path'].split('/', 2)[2])
                exit()
            gender = int(gender)
            img_path = self.img_dir + data_list['img_path'].split('/', 2)[2]
            
            self.img_path_list.append(img_path)
            self.age_list.append(age)
            self.gender_list.append(gender)

        self.age_min = 15
        self.age_max = 72
      
        # Calculate weight
        if mode == 'train':
            print("======================================================================")
            self.age_w_list = np.zeros(self.age_max)
            for age in self.age_list:
                self.age_w_list[age-1] = self.age_w_list[age-1] + 1
            for i in range(0, 70, 10):
                print("age {:d} ~ {:d} : {:.0f} images".format((i+1 if i==0 else i), i+9, (sum(self.age_w_list[i:i+9]) if i == 0 else sum(self.age_w_list[i-1:i+9]))))
            print("age upon 70 : {:.0f} images".format(sum(self.age_w_list[69:])))
            print("Total \033[32m{:.0f}\033[0m images in Training Dataset".format(sum(self.age_w_list)))
            print("======================================================================")

            self.weight_list = max(self.age_w_list) / (self.age_w_list + 1)
        elif mode == 'val':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Validation Dataset".format(len(self.img_path_list)))
            print("======================================================================")
  
        elif mode == 'test':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Testing Dataset".format(len(self.img_path_list)))
            print("======================================================================")

        
    def __len__(self):
        return len(self.img_path_list)
    
    def get_weight(self):
        return torch.from_numpy(self.weight_list)
    
    def get_age_bound(self):
        return self.age_min, self.age_max

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        age = self.age_list[idx]
        gender = self.gender_list[idx]    # Male: 0, Female:1
    
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
        ])


        img = Image.open(img_path)
        if img is None:
            print(img_path)
            raise FileNotFoundError("Image Not Found")
        if self.mode =='train':
            img = transform_aug(img)
        img = self.transform_farl(img)

        sample = {'image': img, 'age': age-self.age_min, 'gender': torch.tensor(gender)}

        return sample 

class DatasetCACD(Dataset):
    def __init__(self, img_dir, img_label_dir, transform_farl, mode):
        self.transform = transforms.Compose([transforms.ToTensor()])                                  
        self.img_dir = img_dir
        json_data = open(img_label_dir, "r") 
        self.img_labels = json.load(json_data)
        self.transform_farl = transform_farl
        self.mode = mode

        # age: 16~77, 0 :male, 1 :female
        self.genders = []
        self.weight_list = []
        
        
        if mode == 'train':
            pool = [0]
        elif mode == 'val':
            pool = [1]
        elif mode == 'test':
            pool = [2]
        
        # Assign list
        self.img_path_list = []
        self.age_list = []
        self.gender_list = []
        self.bbox_list = []
        
        for idx, data_list in enumerate(self.img_labels):
            if data_list["folder"] not in pool:
                continue
            age = int(data_list['age'])
            gender = 0 if data_list['gender'] == 'M' else 1
            if data_list['gender'] != 'F' and data_list['gender'] != "M":
                print(type(gender))
                print(data_list['img_path'].split('/', 2)[2])
                exit()
            gender = int(gender)
            img_path = self.img_dir + data_list['img_path'].split('/', 2)[2]
            aligned_bbox = data_list['aligned_bbox']
            
            self.img_path_list.append(img_path)
            self.age_list.append(age)
            self.gender_list.append(gender)
            self.bbox_list.append(aligned_bbox)

        self.age_min = 14
        self.age_max = 62
        
        # Calculate weight
        if mode == 'train':
            print("======================================================================")
            self.age_w_list = np.zeros(self.age_max)
            for age in self.age_list:
                self.age_w_list[age-1] = self.age_w_list[age-1] + 1
            for i in range(0, 60, 10):
                print("age {:d} ~ {:d} : {:.0f} images".format((i+1 if i==0 else i), i+9, (sum(self.age_w_list[i:i+9]) if i == 0 else sum(self.age_w_list[i-1:i+9]))))
            print("age upon 60 : {:.0f} images".format(sum(self.age_w_list[59:])))
            print("Total \033[32m{:.0f}\033[0m images in Training Dataset".format(sum(self.age_w_list)))
            print("======================================================================")

            self.weight_list = max(self.age_w_list) / (self.age_w_list + 1)
        elif mode == 'val':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Validation Dataset".format(len(self.img_path_list)))
            print("======================================================================")
  
        elif mode == 'test':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Testing Dataset".format(len(self.img_path_list)))
            print("======================================================================")

        
    def __len__(self):
        return len(self.img_path_list)
    
    def get_weight(self):
        return torch.from_numpy(self.weight_list)
    
    def get_age_bound(self):
        return self.age_min, self.age_max

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        age = self.age_list[idx]
        gender = self.gender_list[idx]    # Male: 0, Female:1
        bbox = self.bbox_list[idx]
    
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
        ])

        # img = cv.imread(img_path)
        img = Image.open(img_path)
        img = np.array(img)
        if img is None:
            print(img_path)
            raise FileNotFoundError("Image Not Found")
        if bbox != []:
            out_size = (int(224), int(224))
            img, _ = crop_image(img, bbox, out_size)
        img = Image.fromarray(img)
        if self.mode =='train':
            img = transform_aug(img)
        img = self.transform_farl(img)

        sample = {'image': img, 'age': age-self.age_min, 'gender': torch.tensor(gender)}

        return sample 

class DatasetAgeDB(Dataset):
    def __init__(self, img_dir, img_label_dir, transform_farl, mode, split):
        self.transform = transforms.Compose([transforms.ToTensor()])                                  
        self.img_dir = img_dir
        json_data = open(img_label_dir, "r") 
        self.img_labels = json.load(json_data)
        self.transform_farl = transform_farl
        self.mode = mode

        # age: 1~101, 0 :male, 1 :female
        self.genders = []
        self.weight_list = []
        
        if split == 0:
            if mode == 'train':
                pool = [0, 1, 2, 3, 4, 5]
            elif mode == 'val':
                pool = [6, 7]
            elif mode == 'test':
                pool = [8, 9]
        elif split == 1:
            if mode == 'train':
                pool = [2, 3, 4, 5, 6, 7]
            elif mode == 'val':
                pool = [8, 9]
            elif mode == 'test':
                pool = [0, 1]
        elif split == 2:
            if mode == 'train':
                pool = [4, 5, 6, 7, 8, 9]
            elif mode == 'val':
                pool = [0, 1]
            elif mode == 'test':
                pool = [2, 3]
        elif split == 3:
            if mode == 'train':
                pool = [5, 6, 7, 8, 9, 0]
            elif mode == 'val':
                pool = [1, 2]
            elif mode == 'test':
                pool = [3, 4]
        elif split == 4:
            if mode == 'train':
                pool = [6, 7, 8, 9, 0, 1]
            elif mode == 'val':
                pool = [2, 3]
            elif mode == 'test':
                pool = [4, 5]
        
        # Assign list
        self.img_path_list = []
        self.age_list = []
        self.gender_list = []
        self.bbox_list = []
        
        for idx, data_list in enumerate(self.img_labels):
            if data_list["folder"] not in pool:
                continue
            age = int(data_list['age'])
            gender = 0 if data_list['gender'] == 'M' else 1
            if data_list['gender'] != 'F' and data_list['gender'] != "M":
                print(type(gender))
                print(data_list['img_path'].split('/', 2)[2])
                exit()
            gender = int(gender)
            img_path = self.img_dir + data_list['img_path'].split('/', 2)[2]
            aligned_bbox = data_list['aligned_bbox']
            
            self.img_path_list.append(img_path)
            self.age_list.append(age)
            self.gender_list.append(gender)
            self.bbox_list.append(aligned_bbox)

        self.age_min = 1
        self.age_max = 101
        
        # Calculate weight
        if mode == 'train':
            print("======================================================================")
            self.age_w_list = np.zeros(self.age_max)
            for age in self.age_list:
                self.age_w_list[age-1] = self.age_w_list[age-1] + 1
            for i in range(0, 100, 10):
                print("age {:d} ~ {:d} : {:.0f} images".format((i+1 if i==0 else i), i+9, (sum(self.age_w_list[i:i+9]) if i == 0 else sum(self.age_w_list[i-1:i+9]))))
            print("age upon 100 : {:.0f} images".format(sum(self.age_w_list[99:])))
            print("Total \033[32m{:.0f}\033[0m images in Training Dataset".format(sum(self.age_w_list)))
            print("======================================================================")

            self.weight_list = max(self.age_w_list) / (self.age_w_list + 1)
        elif mode == 'val':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Validation Dataset".format(len(self.img_path_list)))
            print("======================================================================")
  
        elif mode == 'test':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Testing Dataset".format(len(self.img_path_list)))
            print("======================================================================")

        
    def __len__(self):
        return len(self.img_path_list)
    
    def get_weight(self):
        return torch.from_numpy(self.weight_list)
    
    def get_age_bound(self):
        return self.age_min, self.age_max

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        age = self.age_list[idx]
        gender = self.gender_list[idx]    # Male: 0, Female:1
        bbox = self.bbox_list[idx]
    
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
        ])

        # img = cv.imread(img_path)
        img = Image.open(img_path)
        img = np.array(img)
        if img is None:
            print(img_path)
            raise FileNotFoundError("Image Not Found")
        if bbox != []:
            out_size = (int(224), int(224))
            img, _ = crop_image(img, bbox, out_size)
        img = Image.fromarray(img)
        if self.mode =='train':
            img = transform_aug(img)
        img = self.transform_farl(img)

        sample = {'image': img, 'age': age-self.age_min, 'gender': torch.tensor(gender)}

        return sample 

class DatasetFGNET(Dataset):
    def __init__(self, img_dir, img_label_dir, transform_farl, mode, split):
        self.transform = transforms.Compose([transforms.ToTensor()])                                  
        self.img_dir = img_dir
        json_data = open(img_label_dir, "r") 
        self.img_labels = json.load(json_data)
        self.transform_farl = transform_farl
        self.mode = mode

        # age: 16~77, 0 :male, 1 :female
        self.genders = []
        self.weight_list = []
        
        self.age_min = 0
        self.age_max = 69
        
        if mode == 'train':
            pool = [k for k in range(1, 83) if k != split]
        elif mode == 'test' or mode == 'val':
            pool = [split]
        
        # Assign list
        self.img_path_list = []
        self.age_list = []
        self.gender_list = []
        self.bbox_list = []
        
        for idx, data_list in enumerate(self.img_labels):
            folder = data_list["folder"]
            if  folder not in pool:
                continue
            age = int(data_list['age'])
            gender = 0 if data_list['gender'] == 'M' else 1
            if data_list['gender'] != 'F' and data_list['gender'] != "M":
                print(type(gender))
                print(data_list['img_path'].split('/', 2)[2])
                exit()
            gender = int(gender)
            img_path = self.img_dir + data_list['img_path'].split('/', 2)[2].split(".")[0] + "." + data_list['img_path'].split('/', 2)[2].split(".")[1].upper()
            aligned_bbox = data_list['aligned_bbox']
            
            self.img_path_list.append(img_path)
            self.age_list.append(age)
            self.gender_list.append(gender)
            self.bbox_list.append(aligned_bbox)


        
        # Calculate weight
        if mode == 'train':
            print("======================================================================")
            self.age_w_list = np.zeros(self.age_max)
            for age in self.age_list:
                self.age_w_list[age-1] = self.age_w_list[age-1] + 1
            for i in range(0, 70, 10):
                print("age {:d} ~ {:d} : {:.0f} images".format((i+1 if i==0 else i), i+9, (sum(self.age_w_list[i:i+9]) if i == 0 else sum(self.age_w_list[i-1:i+9]))))
            print("age upon 70 : {:.0f} images".format(sum(self.age_w_list[69:])))
            print("Total \033[32m{:.0f}\033[0m images in Training Dataset".format(sum(self.age_w_list)))
            print("======================================================================")

            self.weight_list = max(self.age_w_list) / (self.age_w_list + 1)
        elif mode == 'val':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Validation Dataset".format(len(self.img_path_list)))
            print("======================================================================")
  
        elif mode == 'test':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Testing Dataset".format(len(self.img_path_list)))
            print("======================================================================")

        
    def __len__(self):
        return len(self.img_path_list)
    
    def get_weight(self):
        return torch.from_numpy(self.weight_list)
    
    def get_age_bound(self):
        return self.age_min, self.age_max

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        age = self.age_list[idx]
        gender = self.gender_list[idx]    # Male: 0, Female:1
        bbox = self.bbox_list[idx]
    
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
        ])

        # img = cv.imread(img_path)
        img = Image.open(img_path)
        img = np.array(img)
        if img is None:
            print(img_path)
            raise FileNotFoundError("Image Not Found")
        if bbox != []:
            out_size = (int(224), int(224))
            img, _ = crop_image(img, bbox, out_size)
        img = Image.fromarray(img)
        if self.mode =='train':
            img = transform_aug(img)
        img = self.transform_farl(img)

        sample = {'image': img, 'age': age-self.age_min, 'gender': torch.tensor(gender)}

        return sample 

class DatasetCLAP(Dataset):
    def __init__(self, img_dir, img_label_dir, transform_farl, mode):
        self.transform = transforms.Compose([transforms.ToTensor()])                                  
        self.img_dir = img_dir
        json_data = open(img_label_dir, "r") 
        self.img_labels = json.load(json_data)
        self.transform_farl = transform_farl
        self.mode = mode

        # age: 1~96, 0 :male, 1 :female
        self.genders = []
        self.weight_list = []
        
        self.age_min = 1
        self.age_max = 96
        
   
        if mode == 'train':
            pool = [0]
        elif mode == 'val':
            pool = [1]
        elif mode == 'test':
            pool = [2]
       
        
        # Assign list
        self.img_path_list = []
        self.age_list = []
        self.gender_list = []
        self.bbox_list = []
        
        for idx, data_list in enumerate(self.img_labels):
            folder = data_list["folder"]
            if  folder not in pool:
                continue
            age = int(data_list['age'])
            gender = 0 if data_list['gender'] == 'M' else 1
            if data_list['gender'] != 'F' and data_list['gender'] != "M":
                print(type(gender))
                print(data_list['img_path'].split('/', 1)[1])
                exit()
            gender = int(gender)
            img_path = self.img_dir + data_list['img_path'].split('/', 1)[1]
            aligned_bbox = data_list['aligned_bbox']
            
            self.img_path_list.append(img_path)
            self.age_list.append(age)
            self.gender_list.append(gender)
            self.bbox_list.append(aligned_bbox)


        
        # Calculate weight
        if mode == 'train':
            print("======================================================================")
            self.age_w_list = np.zeros(self.age_max)
            for age in self.age_list:
                self.age_w_list[age-1] = self.age_w_list[age-1] + 1
            for i in range(0, 90, 10):
                print("age {:d} ~ {:d} : {:.0f} images".format((i+1 if i==0 else i), i+9, (sum(self.age_w_list[i:i+9]) if i == 0 else sum(self.age_w_list[i-1:i+9]))))
            print("age upon 90 : {:.0f} images".format(sum(self.age_w_list[89:])))
            print("Total \033[32m{:.0f}\033[0m images in Training Dataset".format(sum(self.age_w_list)))
            print("======================================================================")

            self.weight_list = max(self.age_w_list) / (self.age_w_list + 1)
        elif mode == 'val':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Validation Dataset".format(len(self.img_path_list)))
            print("======================================================================")
  
        elif mode == 'test':
            print("======================================================================")
            print("Total \033[32m{:.0f}\033[0m images in Testing Dataset".format(len(self.img_path_list)))
            print("======================================================================")

        
    def __len__(self):
        return len(self.img_path_list)
    
    def get_weight(self):
        return torch.from_numpy(self.weight_list)
    
    def get_age_bound(self):
        return self.age_min, self.age_max

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        age = self.age_list[idx]
        gender = self.gender_list[idx]    # Male: 0, Female:1
        bbox = self.bbox_list[idx]
    
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
        ])
        
        img = Image.open(img_path)
        img = np.array(img)
        if img is None:
            print(img_path)
            raise FileNotFoundError("Image Not Found")
        if bbox != []:
            out_size = (int(224), int(224))
            img, _ = crop_image(img, bbox, out_size)
        img = Image.fromarray(img)
        if self.mode =='train':
            img = transform_aug(img)
        img = self.transform_farl(img)

        sample = {'image': img, 'age': age-self.age_min, 'gender': torch.tensor(gender)}
        return sample 
    
def build_dataset(data_path, csv_path, mode, transform=None, dataset="utk", split=0):

    if dataset == "utk":
        ds = DatasetUTKFace(data_path, csv_path, transform, mode, split)
    elif dataset == "imdb-fp":
        ds = DatasetIMDB(data_path, csv_path, transform, mode)
    elif dataset == "afad":
        ds = DatasetAFAD(data_path, csv_path, transform, mode, split)
    elif dataset == "cacd":
        ds = DatasetCACD(data_path, csv_path, transform, mode)
    elif dataset == "agedb":
        ds = DatasetAgeDB(data_path, csv_path, transform, mode, split)
    elif dataset == "fgnet":
        ds = DatasetFGNET(data_path, csv_path, transform, mode, split)
    elif dataset == "clap":
        ds = DatasetCLAP(data_path, csv_path, transform, mode)
    return ds

if __name__ == "__main__":
    pass