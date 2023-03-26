# quantization.py
# Vojtech Orava (xorava02)
# BP 2022/2023 FIT VUT
import openvino.runtime as ov
import tensorflow as tf
import argparse
import nncf

# dokumentace https://github.com/openvinotoolkit/nncf

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-d', '--dataset', type=str, required=True)

args = parser.parse_args()

ie = ov.Core()

#model = ie.read_model(r"models\DARKFACE_mobilenet\accelerated\saved_model.xml")
model = ie.read_model(args.model)

#images = tf.data.Dataset.list_files(r'C:\VUTFIT\BP\code\api\darkface\val\images\*.png', shuffle=False)
images = tf.data.Dataset.list_files(args.dataset, shuffle=False)

def load_image(x):
    byte_img = tf.io.read_file(x)
    image = tf.io.decode_png(byte_img)
    return image

def resize_img(img):
    return tf.image.resize(img, (640,640))

images = images.map(load_image)
images = images.map(resize_img)

calibration_dataset = nncf.Dataset(images)
quantized_model = nncf.quantize(model, calibration_dataset)

quantized_path = 'saved_model.xml'
ov.serialize(quantized_model, quantized_path)
print(f"Saving model: {quantized_path}")
