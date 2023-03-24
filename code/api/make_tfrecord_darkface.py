import tensorflow as tf
import os
import numpy as np
import json
import matplotlib.pyplot as plt


AUG_PATH = "data/workspace/aug_darkface"
OUTPUT_PATH = "data/workspace/annotations_darkface"


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode("utf-8")]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def make_record(part, filename):
    
    f = open(os.path.join(AUG_PATH, part, "labels", filename), "r")
    j = json.load(f)
    bboxes = np.array(j["bboxes"], dtype=np.float32)
    image = tf.io.decode_jpeg(tf.io.read_file(os.path.join(AUG_PATH, part, "images", j["image"])))

    height = image.shape[0]
    width = image.shape[1]

    xmin = []
    ymin = []
    xmax = []
    ymax = []


    [xmin.append(max(0.001, (float(item[0]) / width))) for item in bboxes]
    [ymin.append(max(0.001, (float(item[1]) / height))) for item in bboxes]
    [xmax.append(min(0.999, (float(item[2]) / width))) for item in bboxes]
    [ymax.append(min(0.999, (float(item[3]) / height))) for item in bboxes]
    
    
    class_text = []
    class_label = []
    for q in range(len(bboxes)):
        class_text.append(b"face")
        class_label.append(1)
        
        if(xmin[q] >= xmax[q]):
            xmax[q] = xmin[q]+0.001
            
        if(ymin[q] >= ymax[q]):
            ymax[q] = ymin[q]+0.001
            
        
    image_encoded = open(os.path.join(AUG_PATH, part, "images", j["image"]), "rb").read()

    tf_example = tf.train.Example(features=tf.train.Features(feature={    
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_encoded])),
        'image/height':  int64_feature(int(height)),
        'image/width': int64_feature(int(width)),
        'image/filename': bytes_feature(j["image"]),
        'image/format': bytes_feature("png"),
        'image/source_id': bytes_feature(j["image"]),
        'image/object/bbox/xmin': float_feature_list(xmin),
        'image/object/bbox/xmax': float_feature_list(xmax),
        'image/object/bbox/ymin': float_feature_list(ymin),
        'image/object/bbox/ymax': float_feature_list(ymax),
        'image/object/class/text': bytes_list_feature(class_text),
        'image/object/class/label': int64_list_feature(class_label)     
            }))

    print(filename)
        
    return tf_example



def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded':  tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),   
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image/encoded"] = tf.io.decode_jpeg(example["image/encoded"], channels=3)   
    return example

def create_record(part, end, nop):
    counter = nop*200 - 199
    
    labels = sorted(os.listdir(os.path.join(AUG_PATH, part, "labels")), key=lambda x: int(x.split(".")[0]))[counter-1:end]

    writer = tf.io.TFRecordWriter(os.path.join(OUTPUT_PATH, part + str(nop) + ".tfrec"))
    
    for i in range(len(labels)):
        tf_example = make_record(part, labels[i])
        writer.write(tf_example.SerializeToString())
   
   
    writer.close()


if __name__ == "__main__":
    
    test = False
    nop = 1
    
    if test == False:
        while(nop <= 48):
            print("Starting part train", str(nop))
            create_record("train", 200*nop, nop)
            nop+=1
        
        
        nop = 1
        print("Starting part val")
        create_record("val", 1200, nop)
    
    
    if test:
        raw_dataset = tf.data.TFRecordDataset(os.path.join(OUTPUT_PATH, "val1.tfrec"))
        parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

        for features in parsed_dataset.skip(273).take(2):
            for key in features.keys():
                if key != "image":
                    print(f"{key}: {features[key]}")

            print(f"Image shape: {features['image/encoded'].shape}")
            plt.figure(figsize=(7, 7))
            plt.imshow(features["image/encoded"].numpy())
            plt.show()