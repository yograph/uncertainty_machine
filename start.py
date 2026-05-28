
import gc
import warnings
import os
import random
import logging
import io
from io import BytesIO

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.stats import entropy
from sklearn.manifold import TSNE
from tabulate import tabulate
from tqdm.notebook import tqdm
from PIL import Image as PILImage
import imageio
from IPython.display import Image
from tensorflow import keras
import gradio as gr

IMAGE_SIZE = 256
BATCH_SIZE = 1
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000

PATH_OF_TEST_A = '/Users/youseffarahat/Documents/uncertainty_machine/testA'
PATH_OF_TEST_B = '/Users/youseffarahat/Documents/uncertainty_machine/testB'
PATH_OF_TRAIN_A = '/Users/youseffarahat/Documents/uncertainty_machine/trainA'
PATH_OF_TRAIN_B = '/Users/youseffarahat/Documents/uncertainty_machine/trainB'

def get_image_paths(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)])
    return paths

print(get_image_paths(PATH_OF_TRAIN_A))

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    return img

def preprocess_train(path):
    img = load_image(path)
    img = tf.image.resize(img, [286, 286])
    img = tf.image.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
    img = tf.image.random_flip_left_right(img)
    img = (img / 127.5) - 1.0
    return img

def preprocess_test(path):
    img = load_image(path)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = (img / 127.5) - 1.0
    return img

def make_dataset(paths, preprocess_fn, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_a_paths = get_image_paths(PATH_OF_TRAIN_A)
train_b_paths = get_image_paths(PATH_OF_TRAIN_B)

val_a_paths = train_a_paths
val_b_paths = train_b_paths

train_a_paths = train_a_paths
train_b_paths = train_b_paths

test_a_paths = get_image_paths(PATH_OF_TEST_A)
test_b_paths = get_image_paths(PATH_OF_TEST_B)

train_A = make_dataset(train_a_paths, preprocess_train, shuffle=True)
train_B = make_dataset(train_b_paths, preprocess_train, shuffle=True)
val_A   = make_dataset(val_a_paths,   preprocess_test)
val_B   = make_dataset(val_b_paths,   preprocess_test)
test_A  = make_dataset(test_a_paths,  preprocess_test)
test_B  = make_dataset(test_b_paths,  preprocess_test)

train_ds = tf.data.Dataset.zip((train_A, train_B))
val_ds   = tf.data.Dataset.zip((val_A,   val_B))
test_ds  = tf.data.Dataset.zip((test_A,  test_B))

print(f"Train A: {len(train_a_paths)}, Train B: {len(train_b_paths)}")
print(f"Val A: {len(val_a_paths)}, Val B: {len(val_b_paths)}")
print(f"Test A: {len(test_a_paths)}, Test B: {len(test_b_paths)}")

for a, b in train_ds.take(1):
    print(f"Batch shape A: {a.shape}, B: {b.shape}")
    print(f"Min/Max A: {tf.reduce_min(a):.2f} / {tf.reduce_max(a):.2f}")
    print(f"Min/Max B: {tf.reduce_min(b):.2f} / {tf.reduce_max(b):.2f}")