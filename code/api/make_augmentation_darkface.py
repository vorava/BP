# make_augmentation_darkface.py
# Vojtech Orava (xorava02)
# BP 2022/2023, FIT VUT
import os
import cv2
import numpy as np
import albumentations as alb
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--train_images', type=str, required=False, help="train images folder path")
parser.add_argument('-v', '--val_images', type=str, required=False, help="val images folder path")
parser.add_argument('-l', '--train_labels', type=str, required=False, help="train labels folder path")
parser.add_argument('-e', '--val_labels', type=str, required=False, help="val labels folder path")
parser.add_argument('-o', '--output_path', type=str, required=False, help="output folder path")

args = parser.parse_args()

augmentor = alb.Compose([
    alb.HorizontalFlip(p=0.5),
    alb.VerticalFlip(p=0.5),
    alb.RandomGamma(p=0.1),
    alb.ISONoise(p=0.2),
    alb.RandomBrightnessContrast()],
    bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['class_labels'])
)

# cesty
TRAIN_PATH = "darkface/train/images" if args.train_images is None else args.train_images
VAL_PATH = "darkface/val/images" if args.val_images is None else args.val_images
TRAIN_LABELS = "darkface/train/labels" if args.train_labels is None else args.train_labels
VAL_LABELS = "darkface/val/labels" if args.val_labels is None else args.val_labels
OUTPUT_PATH = "data/workspace/aug_darkface" if args.output_path is None else args.output_path


def proccess_one(img_path, labels_path, part):
    
    with open(labels_path, "r") as f:
        count = int(f.readline())
        bboxes = []
        
        for i in range(count):
            bbox = np.array(f.readline().split(), dtype=np.int32)
            bboxes.append(bbox.tolist())
            
    img_name = img_path.split("/")[-1].split(".")[0]+".png"
    json_name = img_path.split("/")[-1].split(".")[0] + ".json"
    
    print(img_name)

    annotation = {}
    annotation['image'] = img_name
    annotation['bboxes'] = bboxes
    annotation["class"] = ['face']*len(bboxes)

    with open(os.path.join(OUTPUT_PATH, part, 'labels', json_name), 'w') as b:
        json.dump(annotation, b)    

        
    image = cv2.imread(img_path)
         
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(os.path.join(OUTPUT_PATH, part, 'images', img_name),
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if part != "val":
    
        create_aug(img_name.split(".")[0], bboxes, image, part)      
        
        

def create_aug(filename, bboxes, image, part):
    
    for b in bboxes:
        if(b[0] >= b[2]):
            b[2] = b[0]+3
        if(b[1] >= b[3]):
            b[3] = b[1]+3
    
    augmented = augmentor(image=image, bboxes=bboxes, class_labels=(['face']*len(bboxes)))
    img_name = f'{filename}.0.png'
    json_name = f'{filename}.0.json'

    cv2.imwrite(os.path.join(OUTPUT_PATH, part, 'images', img_name),
                cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB))

    annotation = {}
    annotation['image'] = img_name
    annotation['bboxes'] = augmented["bboxes"]
    annotation["class"] = augmented["class_labels"]

    with open(os.path.join(OUTPUT_PATH, part, 'labels', json_name), 'w') as b:
        json.dump(annotation, b)
  


if __name__ == "__main__":
    print("Starting train part")
    
    files = sorted(os.listdir(TRAIN_PATH), key=lambda x: int(x.split(".")[0]))
    labels = sorted(os.listdir(TRAIN_LABELS), key=lambda x: int(x.split(".")[0]))
    
    for i in range(len(files)):
        proccess_one(os.path.join(TRAIN_PATH, files[i]), os.path.join(TRAIN_LABELS, labels[i]), 'train')
   
    print("Starting val part")   
    
    files = sorted(os.listdir(VAL_PATH), key=lambda x: int(x.split(".")[0]))
    labels = sorted(os.listdir(VAL_LABELS), key=lambda x: int(x.split(".")[0]))
    
    for i in range(len(files)):
        proccess_one(os.path.join(VAL_PATH, files[i]), os.path.join(VAL_LABELS, labels[i]), 'val')

    
        
    
    