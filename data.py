
import cv2
import random
import os
import numpy as np
import torch 
from torch.utils import data
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class COCO(data.Dataset):
    def __init__(self, root, split, req_label=False, req_augment=False,  
                crop_size=448, scales=(1), flip=False):

        super().__init__()

        self.root = root
        self.split = split
        self.image_path = os.path.join(root, split)
        self.id_to_filename = self._get_COCOid_to_filename()
        self.ids = self.id_to_filename.keys()

        self.req_label = req_label
        if req_label:
            self.label_path = os.join(root, "label", split)

        self.req_augment = req_augment
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip

        self.transform = transforms.Compose([
                transforms.ToTensor(), #to tensor and normalize to [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                ])

        cv2.setNumThreads(0)

    def _get_COCOid_to_filename(self):
        id_to_filename = {}

        for filename in os.listdir(self.image_path):
            if not filename.endswith(".jpg"):
                continue
    
            name = (filename.split('_')[-1]).split('.')[0]
            id = int(name)
            id_to_filename[id] = filename
        
        return id_to_filename

    def _augmentation(self, image, label):
        scale_factor = random.choice(self.scales)
        h, w = image.shape()
        th, tw = int(scale_factor * h), int(scale_factor * w)
        
        image = cv2.resize(image, (th, tw), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((th, tw), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        if self.flip:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW

        return image, label

    def _load_data(self, index):
        COCOid = self.ids[index]
        filename = self.id_to_filename[COCOid]

        image = cv2.imread(os.path.join(self.image_path, filename), cv2.IMREAD_COLOR).astype(np.float32)
        
        if self.req_label:
            filename = filename.split('.')[0] + ".png"
            label = cv2.imread(os.path.join(self.label_path, filename, cv2.IMREAD_GRAYSCALE))
        else:
            label = None

        return image, label

    def __getitem__(self, index):
        image, label = self._load_data(index)
        image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_LINEAR)
        
        label = None
        
        if self.req_label:
            label = Image.fromarray(label).resize((448, 448), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int64)

            if self.req_augment:
                image, label = self._augmentation(image, label)

        image = image[:, :, ::-1] #bgr to rgb
        plt.imsave('{:d}_img.jpg'.format(index))
        plt.imsave('{:d}_lab.jpg'.format(label))
        image = image.transpose(2, 0, 1) #CHW
        image = self.transform(image)

        return image, torch.from_numpy(label)

        
if __name__ == "__main__":
    coco_data = COCO('../dataset/COCO', 'train2017', req_label=True)
    img, label = coco_data[0]
    print(img.size())
    print(label.size())
    img, label = coco_data[10]
    print(img)
    print(label)
        
        


    
    


