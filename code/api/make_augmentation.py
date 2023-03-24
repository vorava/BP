import os
import cv2
import numpy as np
import albumentations as alb
import json
from matplotlib import pyplot as plt


augmentor = alb.Compose([
    alb.HorizontalFlip(p=0.5),
    alb.VerticalFlip(p=0.5),
    alb.RandomBrightnessContrast(brightness_limit=[-0.3,-0.1], contrast_limit=[-0.3,-0.1], p=0.7),
    alb.RandomGamma(p=0.1),
    alb.GaussNoise(p=0.2),
    alb.ISONoise(p=0.4)],
    bbox_params=alb.BboxParams(format='pascal_voc', label_fields=['class_labels'])
)

# cesty
WIDER_TRAIN_PATH = "../data/WIDER_train/images"
TRAIN_GT_PATH = "../data/wider_face_split/wider_face_train_bbx_gt.txt"
#TRAIN_GT_PATH = "../data/wider_face_split/short.txt"
OUTPUT_PATH = "data/workspace/aug/train"


name_counter = 1

def get_data(gt_path, imgs_path):
    global name_counter
    with open(gt_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            filename = line.rstrip()

            print(filename, "->", str(name_counter))

            filepath = os.path.join(imgs_path, filename)

            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes = []
            count = int(f.readline())
            if count == 0:
                print(filename)  
                f.readline()
            
            else:
       
                for i in range(count):
                    bbox = np.array(f.readline().split()[:4], dtype=np.int32)
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                 
                    if bbox[0] >= int(image.shape[1]):
                            bbox[0] = int(image.shape[1])-2
                            
                    if bbox[1] >= int(image.shape[0]):
                            bbox[1] = int(image.shape[0])-2
                            
                    if bbox[2] >= int(image.shape[1]):
                            bbox[2] = int(image.shape[1])-2
                    
                    if bbox[3] >= int(image.shape[0]):
                            bbox[3] = int(image.shape[0])-2
                            
                    bboxes.append(bbox.tolist())
                    
                img_name = str(name_counter) + ".jpg"
                json_name = str(name_counter) + ".json"

                print(img_name)

                annotation = {}
                annotation['image'] = img_name
                annotation['bboxes'] = bboxes
                annotation["class"] = ['face']*len(bboxes)

                with open(os.path.join(OUTPUT_PATH, 'labels', json_name), 'w') as b:
                    json.dump(annotation, b)  
                    
                cv2.imwrite(os.path.join(OUTPUT_PATH, 'images', img_name),
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                create_aug(bboxes, image)
                name_counter += 1
           
        
def create_aug(bboxes, image):
        global name_counter
        for b in bboxes:
            if(b[0] >= b[2]):
                b[2] = b[0]+1
            if(b[1] >= b[3]):
                b[3] = b[1]+1
 
     
        augmented = augmentor(image=image, bboxes=bboxes, class_labels=(['face']*len(bboxes)))
        img_name = f'{name_counter}.0.jpg'
        json_name = f'{name_counter}.0.json'

        cv2.imwrite(os.path.join(OUTPUT_PATH, 'images', img_name),
                    cv2.cvtColor(augmented['image'], cv2.COLOR_BGR2RGB))

        annotation = {}
        annotation['image'] = img_name
        annotation['bboxes'] = augmented["bboxes"]
        annotation["class"] = augmented["class_labels"]

        with open(os.path.join(OUTPUT_PATH, 'labels', json_name), 'w') as b:
            json.dump(annotation, b)
    


if __name__ == "__main__":
    print("Starting train part")
    get_data(TRAIN_GT_PATH, WIDER_TRAIN_PATH)
    

    
        
    
    