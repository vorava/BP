from mtcnn import MTCNN
import cv2
import json

img = cv2.cvtColor(cv2.imread("../../api/data/workspace/aug/val/images/1.0.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
res = detector.detect_faces(img)




for r in res:
    box = r["box"]
    print(box)
    img = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,0,0), 2)
    
cv2.imwrite("output.jpg", img)