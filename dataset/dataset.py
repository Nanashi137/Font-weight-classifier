import cv2 as cv
from torch.utils.data import Dataset
import numpy as np
import torch
from dataset.transformations import Ptrans


LABELS2ID = {'unbold': 0, 'bold': 1, 'italic': 2, 'bold_italic': 3}
ID2LABELS = {0: 'unbold', 1: 'bold', 2: 'italic', 3: 'bold_italic'}

class FontWeightDataset(Dataset): 
    def __init__(self, img_folder, transform=None): 
        self.img_paths = img_folder 
        self.transform = transform 
    
    def __len__(self): 
        return len(self.img_paths)
    
    def __getitem__(self, idx): 
        img_filepath = self.img_paths[idx]
        img = cv.imread(img_filepath, cv.IMREAD_GRAYSCALE) 

        img_np = np.array(img, dtype="uint8")
        label = img_filepath.split(",")[1][:-4]
        ID = LABELS2ID[label]
        x = self.transform(image=img_np, force_apply= True)["image"]
        
        return x, torch.as_tensor(ID, dtype= torch.long)

class FWDataset(Dataset): 
    def __init__(self, img_folder): 
        self.img_paths = img_folder 
    
    def __len__(self): 
        return len(self.img_paths)
    
    def __getitem__(self, idx): 
        img_filepath = self.img_paths[idx]
        img = cv.imread(img_filepath, cv.IMREAD_GRAYSCALE) 

        img_np = np.array(img, dtype="uint8")
        label = img_filepath.split(",")[1][:-4]
        ID = LABELS2ID[label]
        x = Ptrans(img_np)

        return x, torch.as_tensor(ID, dtype= torch.long)


#debug 
if __name__ == "__main__": 
    img = cv.imread("F:/FWC216/t.jpg", cv.IMREAD_GRAYSCALE) 
    cv.imshow("n", img)
    cv.waitKey(0)

