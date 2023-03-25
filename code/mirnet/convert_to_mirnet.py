import tensorflow as tf  
from keras.models import load_model
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt 
import time 

mirnet = load_model("myMirnet")

def infer(image):  
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    
    
    output = mirnet.predict(image)
   
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    
    output_image = output_image.astype("uint8")
    return output_image

start = time.time()
cap = cv2.VideoCapture("video3.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (1280,720))

print(f"Resolution: {frame_width} x {frame_height}")
print(f"FPS: {fps}")

f = 1

while cap.isOpened():
    _, image = cap.read()
   

    print(f"{f}/{frames}")
    f+=1

    try:
        image = cv2.resize(image, (1280,720))
        image = infer(image)

        output.write(image)
    except Exception:
        break





cap.release()
output.release()

end = time.time()

print("Duration: " + str(end-start))