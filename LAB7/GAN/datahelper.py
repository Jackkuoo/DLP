from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class ICLEVRLoader(Dataset):
    def __init__(self, root_folder, img_path, trans=None, cond=False, mode='train'):
        self.root_folder = root_folder
        self.img_path = img_path
        self.mode = mode
        self.img_list, self.label_list = get_iCLEVR_data(root_folder,mode)
        if self.mode == 'train':
            print("> Found %d images..." % (len(self.img_list)))

        with open(os.path.join('task_1','objects.json'),'r') as file:
            self.classes = json.load(file)
        self.cond = cond
        self.num_classes = 24
        self.img_names=[]
        self.img_conditions=[]
        self.max_objects=0
        with open(os.path.join(self.root_folder,'train.json'),'r') as file:
            dict=json.load(file)
            for img_name,img_condition in dict.items():
                self.img_names.append(img_name)
                self.max_objects=max(self.max_objects,len(img_condition))
                self.img_conditions.append([self.classes[condition] for condition in img_condition])
        self.transformations=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_names)

    def __getitem__(self, index):
        img=Image.open(os.path.join(self.img_path,self.img_names[index])).convert('RGB')
        img=self.transformations(img)
        condition=self.int2onehot(self.img_conditions[index])
        return img,condition

    def int2onehot(self,int_list):
        onehot=torch.zeros(self.num_classes)
        for i in int_list:
            onehot[i]=1.
        return onehot