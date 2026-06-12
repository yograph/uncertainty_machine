import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
LEARNING_RATE = 0.0005
CLASSES = 2

PATH_OF_TEST_A = '/Users/youseffarahat/Documents/uncertainty_machine/horse2zebra/testA'
PATH_OF_TEST_B = '/Users/youseffarahat/Documents/uncertainty_machine/horse2zebra/testB'
PATH_OF_TRAIN_A = '/Users/youseffarahat/Documents/uncertainty_machine/horse2zebra/trainA'
PATH_OF_TRAIN_B = '/Users/youseffarahat/Documents/uncertainty_machine/horse2zebra/trainB'


def get_image_paths(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)])
    return paths


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


def make_dataset(paths, labels, preprocess_fn, shuffle=False):
    # (path, label) pairs so the classifier gets supervised targets:
    # label 0 = domain A (horse), label 1 = domain B (zebra)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE)
    ds = ds.map(lambda p, y: (preprocess_fn(p), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_a_paths = get_image_paths(PATH_OF_TRAIN_A)
train_b_paths = get_image_paths(PATH_OF_TRAIN_B)

val_a_paths = train_a_paths[:200]
val_b_paths = train_b_paths[:200]

train_a_paths = train_a_paths[200:1000]  # 800 train
train_b_paths = train_b_paths[200:1000]

test_a_paths = get_image_paths(PATH_OF_TEST_A)
test_b_paths = get_image_paths(PATH_OF_TEST_B)

train_paths = train_a_paths + train_b_paths
train_labels = [0] * len(train_a_paths) + [1] * len(train_b_paths)

val_paths = val_a_paths + val_b_paths
val_labels_list = [0] * len(val_a_paths) + [1] * len(val_b_paths)

test_paths = test_a_paths + test_b_paths
test_labels_list = [0] * len(test_a_paths) + [1] * len(test_b_paths)

train_ds = make_dataset(train_paths, train_labels, preprocess_train, shuffle=True)
val_ds = make_dataset(val_paths, val_labels_list, preprocess_test)
test_ds = make_dataset(test_paths, test_labels_list, preprocess_test)


def build_model(dropout_rate=0.5):
    # Same architecture as the PyTorch version in torching.py
    model = keras.Sequential([
        keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(dropout_rate),
        layers.Conv2D(64, kernel_size=5, activation='relu'),
        layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(CLASSES, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


model = build_model()
model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
    ],
)

# Plot the training and validation loss curves
epochs_ran = len(history.history['loss'])
plt.figure()
plt.plot(range(1, epochs_ran + 1), history.history['loss'], label='Training Loss')
plt.plot(range(1, epochs_ran + 1), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy curves
plt.figure()
plt.plot(range(1, epochs_ran + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(1, epochs_ran + 1), history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# Monte Carlo Dropout inference: calling the model with training=True keeps
# the Dropout layers sampling at prediction time, so repeated forward passes
# give different predictions whose spread measures the model's uncertainty.
def monte_carlo_dropout_inference(model, dataset, num_samples, num_classes):
    predictions = []
    all_labels = []
    for images, labels in dataset:
        outputs = np.stack(
            [model(images, training=True).numpy() for _ in range(num_samples)]
        )  # (num_samples, batch, num_classes)
        predictions.append(outputs)
        all_labels.append(labels.numpy())

    # Concatenate along the batch axis -> (num_samples, total_images, num_classes)
    return np.concatenate(predictions, axis=1), np.concatenate(all_labels)


num_samples = 50
predictions, val_labels = monte_carlo_dropout_inference(model, val_ds, num_samples, CLASSES)
print(predictions.shape)

# Predictive mean and variance over the MC sample axis (axis 0)
mean_predictions = predictions.mean(axis=0)   # (total_val_images, num_classes)
var_predictions = predictions.var(axis=0)     # (total_val_images, num_classes)

# ---------------------------------------------------------------------------
# Uncertainty decomposition
#
# Total uncertainty   = predictive entropy H[ E[p] ]
# Aleatoric (data)    = expected entropy   E[ H[p] ]
# Epistemic (model)   = mutual information = total - aleatoric
# ---------------------------------------------------------------------------
eps = 1e-12

predictive_entropy = -np.sum(mean_predictions * np.log(mean_predictions + eps), axis=1)
expected_entropy = -np.mean(np.sum(predictions * np.log(predictions + eps), axis=2), axis=0)
mutual_information = predictive_entropy - expected_entropy

print(f"Mean predictive entropy (total):     {predictive_entropy.mean():.4f}")
print(f"Mean expected entropy (aleatoric):   {expected_entropy.mean():.4f}")
print(f"Mean mutual information (epistemic): {mutual_information.mean():.4f}")

pred_labels = mean_predictions.argmax(axis=1)
correct_mask = pred_labels == val_labels
mc_accuracy = 100 * correct_mask.mean()
print(f"MC-dropout val accuracy: {mc_accuracy:.2f}%")
print(f"Mean entropy on correct predictions: {predictive_entropy[correct_mask].mean():.4f}")
if (~correct_mask).any():
    print(f"Mean entropy on wrong predictions:   {predictive_entropy[~correct_mask].mean():.4f}")

# Distribution of total uncertainty over the validation set
plt.figure()
plt.hist(predictive_entropy[correct_mask], bins=30, alpha=0.6, label='Correct')
if (~correct_mask).any():
    plt.hist(predictive_entropy[~correct_mask], bins=30, alpha=0.6, label='Wrong')
plt.xlabel('Predictive Entropy')
plt.ylabel('Number of Images')
plt.title('Uncertainty Distribution (MC Dropout)')
plt.legend()
plt.show()


def transform_image(image):
    # Min-max scaling to transform image tensor from -1 to 1 to 0 to 1
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)


# Show uncertainty estimates for a single image from the val set
image_index = 1
class_names = ["A (Horse)", "B (Zebra)"]

image = preprocess_test(val_paths[image_index]).numpy()
true_label = class_names[val_labels_list[image_index]]

plt.imshow(transform_image(image))
plt.axis('off')
plt.title('True Label: ' + true_label)
plt.show()

for i in range(CLASSES):
    print(f"{class_names[i]} - Mean: {mean_predictions[image_index][i]:.4f}, "
          f"Variance: {var_predictions[image_index][i]:.6f}")

plt.figure()
plt.bar(class_names, mean_predictions[image_index], yerr=np.sqrt(var_predictions[image_index]))
plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Predicted Probabilities with Uncertainty')
plt.xticks(rotation=45)
plt.show()


# Show the validation images the model is most and least certain about
def show_examples(indices, title):
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    for ax, idx in zip(np.atleast_1d(axes), indices):
        img = preprocess_test(val_paths[idx]).numpy()
        ax.imshow(transform_image(img))
        ax.axis('off')
        ax.set_title(f"true: {class_names[val_labels_list[idx]]}\n"
                     f"pred: {class_names[pred_labels[idx]]}\n"
                     f"H={predictive_entropy[idx]:.3f}  MI={mutual_information[idx]:.3f}")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


order = np.argsort(predictive_entropy)
n_show = min(3, len(order))
show_examples(order[:n_show], 'Most certain predictions')
show_examples(order[-n_show:][::-1], 'Most uncertain predictions')
