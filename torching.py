import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
BUFFER_SIZE = 1000
LEARNING_RATE = 0.0005
CLASSES = 2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_OF_TEST_A = os.path.join(BASE_DIR, "testA")
PATH_OF_TEST_B = os.path.join(BASE_DIR, "testB")
PATH_OF_TRAIN_A = os.path.join(BASE_DIR, "trainA")
PATH_OF_TRAIN_B = os.path.join(BASE_DIR, "trainB")


def get_image_paths(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])
    return paths


# print(get_image_paths(PATH_OF_TRAIN_A))


class ImageFolderDataset(Dataset):
    def __init__(self, paths, label, transform=None):
        self.paths = paths
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.label


# Equivalent to:
# resize 286 -> random crop 256 -> random horizontal flip -> normalise to [-1, 1]
train_transform = T.Compose([
    T.Resize((286, 286)),
    T.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),  # converts to [0, 1] and shape [C, H, W]
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])  # converts to [-1, 1]
])


# Equivalent to:
# resize 256 -> normalise to [-1, 1]
test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
])


train_a_paths = get_image_paths(PATH_OF_TRAIN_A)
train_b_paths = get_image_paths(PATH_OF_TRAIN_B)

val_a_paths = train_a_paths[:200]
val_b_paths = train_b_paths[:200]

train_a_paths = train_a_paths[200:1000]  # 800 train
train_b_paths = train_b_paths[200:1000]

test_a_paths = get_image_paths(PATH_OF_TEST_A)
test_b_paths = get_image_paths(PATH_OF_TEST_B)


train_A_dataset = ImageFolderDataset(train_a_paths, label=0, transform=train_transform)
train_B_dataset = ImageFolderDataset(train_b_paths, label=1, transform=train_transform)
train_datasets = torch.utils.data.ConcatDataset([train_A_dataset, train_B_dataset])

val_A_dataset = ImageFolderDataset(val_a_paths, label=0, transform=test_transform)
val_B_dataset = ImageFolderDataset(val_b_paths, label=1, transform=test_transform)
val_datasets = torch.utils.data.ConcatDataset([val_A_dataset, val_B_dataset])

test_A_dataset = ImageFolderDataset(test_a_paths, label=0, transform=test_transform)
test_B_dataset = ImageFolderDataset(test_b_paths, label=1, transform=test_transform)
test_datasets = torch.utils.data.ConcatDataset([test_A_dataset, test_B_dataset])


train_loader = DataLoader(
    train_datasets,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_datasets,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

test_loader = DataLoader(
    test_datasets,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# print(f"Train A: {len(train_a_paths)}, Train B: {len(train_b_paths)}")
# print(f"Val A: {len(val_a_paths)}, Val B: {len(val_b_paths)}")
# print(f"Test A: {len(test_a_paths)}, Test B: {len(test_b_paths)}")

import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, dropout_variable = 0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        # After conv1+pool -> 32x64x64
        # After conv2+pool -> 64x30x30
        # GAP averages 30x30 -> 1x1, leaving a 64-dim vector per sample
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout_variable)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x)                # (B, 64, 30, 30) -> (B, 64, 1, 1)
        x = torch.flatten(x, 1)        # -> (B, 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Lists to store training and validation metrics
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

# Early stopping state
best_val_loss = float("inf")
best_state = None
patience = 5          # epochs without improvement before stopping
bad_epochs = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Calculate training accuracy and loss
    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # Evaluation on validation set
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            val_loss += criterion(outputs, labels).item()

            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    # Calculate validation accuracy and loss
    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}], "
          f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Store metrics in lists
    train_loss_list.append(train_loss)
    train_acc_list.append(train_accuracy)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_accuracy)

    # Early stopping check: save best weights, stop if no improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1} (best val_loss = {best_val_loss:.4f})")
            break

# Restore the best-val checkpoint for downstream inference / MC dropout
if best_state is not None:
    model.load_state_dict(best_state)

# Plot the training and validation loss curves
epochs_ran = len(train_loss_list)
plt.figure()
plt.plot(range(1, epochs_ran + 1), train_loss_list, label='Training Loss')
plt.plot(range(1, epochs_ran + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy curves
plt.figure()
plt.plot(range(1, epochs_ran + 1), train_acc_list, label='Training Accuracy')
plt.plot(range(1, epochs_ran + 1), val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Loss and Accuracy on Validation set it better than on Training set because of dropout

def enable_mc_dropout(model):
    # eval() everywhere, then switch ONLY the dropout layers back to train mode.
    # Plain model.train() would also flip layers like BatchNorm into training
    # behaviour, which corrupts the predictions.
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


# Define a function for Monte Carlo Dropout inference
def monte_carlo_dropout_inference(model, dataloader, num_samples, num_classes):
    enable_mc_dropout(model)
    predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = torch.zeros((num_samples, images.size(0), num_classes)).to(device)

            # For each batch in the dataloader, 'num_samples' predictions are being made
            for i in range(num_samples):
                outputs[i] = model(images)

            # Apply softmax to all 'num_samples' predictions made for that batch, then store them.
            predictions.append(outputs.softmax(dim=-1).cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate along the batch axis -> (num_samples, total_images, num_classes).
    # Using concatenate (not np.array) handles the final partial batch correctly.
    return np.concatenate(predictions, axis=1), np.concatenate(all_labels)

# Perform Monte Carlo Dropout inference
num_samples = 50
predictions, val_labels = monte_carlo_dropout_inference(model, val_loader, num_samples, CLASSES)
print(predictions.shape)
# shape: (num_samples, total_val_images, num_classes)


# Predictive mean and variance over the MC sample axis (axis 0)
mean_predictions = predictions.mean(axis=0)   # (total_val_images, num_classes)
var_predictions  = predictions.var(axis=0)    # (total_val_images, num_classes)

print(mean_predictions.shape)

# ---------------------------------------------------------------------------
# Uncertainty decomposition
#
# Total uncertainty   = predictive entropy H[ E[p] ]
# Aleatoric (data)    = expected entropy   E[ H[p] ]   (noise the model cannot
#                                                       remove with more data)
# Epistemic (model)   = mutual information = total - aleatoric  (uncertainty
#                       about the weights; shrinks with more training data)
# ---------------------------------------------------------------------------
eps = 1e-12

# Entropy of the averaged prediction -> total uncertainty, shape (N,)
predictive_entropy = -np.sum(mean_predictions * np.log(mean_predictions + eps), axis=1)

# Average entropy of each individual MC prediction -> aleatoric, shape (N,)
expected_entropy = -np.mean(
    np.sum(predictions * np.log(predictions + eps), axis=2), axis=0
)

# Mutual information between prediction and weights -> epistemic, shape (N,)
mutual_information = predictive_entropy - expected_entropy

print(f"Mean predictive entropy (total):     {predictive_entropy.mean():.4f}")
print(f"Mean expected entropy (aleatoric):   {expected_entropy.mean():.4f}")
print(f"Mean mutual information (epistemic): {mutual_information.mean():.4f}")

# Is the uncertainty useful? Compare it for correct vs. wrong predictions:
# a good uncertainty estimate should be higher on the mistakes.
pred_labels = mean_predictions.argmax(axis=1)
correct_mask = pred_labels == val_labels
mc_accuracy = 100 * correct_mask.mean()
print(f"MC-dropout val accuracy: {mc_accuracy:.2f}%")
print(f"Mean entropy on correct predictions: {predictive_entropy[correct_mask].mean():.4f}")
if (~correct_mask).any():
    print(f"Mean entropy on wrong predictions:   {predictive_entropy[~correct_mask].mean():.4f}")

cm = confusion_matrix(val_labels, pred_labels)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(val_labels, pred_labels, target_names=["A (Horse)", "B (Zebra)"]))

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A (Horse)", "B (Zebra)"])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix (MC Dropout)')
plt.show()

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
    transformed_image = (image - image_min) / (image_max - image_min)
    return transformed_image

# Show uncertainty estimates for a single image from the val set
image_index = 1

# label=0 -> horses (folder A), label=1 -> zebras (folder B). Index matches that.
class_names = ["A (Horse)", "B (Zebra)"]

# DataLoaders aren't indexable; pull the sample directly from the underlying dataset.
image, label = val_loader.dataset[image_index]
true_label = class_names[label]

# Convert the image tensor to a NumPy array and transpose the dimensions
image = np.transpose(image.numpy(), (1, 2, 0))

# Plot the image
plt.imshow(transform_image(image))
plt.axis('off')
plt.title('True Label: ' + true_label)
plt.show()

# Display uncertainty estimates for each class
for i in range(CLASSES):
    class_name = class_names[i]
    mean_prediction = mean_predictions[image_index][i]
    variance_prediction = var_predictions[image_index][i]
    print(f"{class_name} - Mean: {mean_prediction:.4f}, Variance: {variance_prediction:.6f}")

# Visualize the uncertainty for each class
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
        img, lbl = val_loader.dataset[idx]
        ax.imshow(transform_image(np.transpose(img.numpy(), (1, 2, 0))))
        ax.axis('off')
        ax.set_title(f"true: {class_names[lbl]}\n"
                     f"pred: {class_names[pred_labels[idx]]}\n"
                     f"entropy={predictive_entropy[idx]:.3f}  epistemic={mutual_information[idx]:.3f}")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

order = np.argsort(predictive_entropy)
n_show = min(3, len(order))
show_examples(order[:n_show], 'Most certain predictions')
show_examples(order[-n_show:][::-1], 'Most uncertain predictions')

# Harder cases: corrupted images
class CorruptedDataset(Dataset):
    def __init__(self, base_dataset, corruption_fn):
        self.base_dataset = base_dataset
        self.corruption_fn = corruption_fn

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        img = self.corruption_fn(img)
        return img, label


def add_gaussian_noise(img, std=0.3):
    return img + torch.randn_like(img) * std

def apply_blur(img, kernel_size=15, sigma=5.0):
    return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=sigma)

def apply_center_crop(img, crop_fraction=0.5):
    _, h, w = img.shape
    crop_size = int(min(h, w) * crop_fraction)
    img = TF.center_crop(img, [crop_size, crop_size])
    img = TF.resize(img, [h, w])
    return img


corruption_configs = {
    'Clean': None,
    'Gaussian Noise': lambda img: add_gaussian_noise(img, std=0.3),
    'Blurred': lambda img: apply_blur(img, kernel_size=15, sigma=5.0),
    'Center Crop (50%)': lambda img: apply_center_crop(img, crop_fraction=0.5),
}

results_harder = {}

for name, corruption_fn in corruption_configs.items():
    if corruption_fn is None:
        loader = val_loader
    else:
        corrupted_ds = CorruptedDataset(val_loader.dataset, corruption_fn)
        loader = DataLoader(corrupted_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds, labels = monte_carlo_dropout_inference(model, loader, num_samples, CLASSES)
    mean_preds = preds.mean(axis=0)

    predictive_entropy_c = -np.sum(mean_preds * np.log(mean_preds + eps), axis=1)
    expected_entropy_c = -np.mean(np.sum(preds * np.log(preds + eps), axis=2), axis=0)
    mutual_information_c = predictive_entropy_c - expected_entropy_c

    pred_cls = mean_preds.argmax(axis=1)
    acc = 100 * (pred_cls == labels).mean()

    results_harder[name] = {
        'accuracy': acc,
        'total_entropy': predictive_entropy_c.mean(),
        'aleatoric': expected_entropy_c.mean(),
        'epistemic': mutual_information_c.mean(),
    }
    print(f"\n--- {name} ---")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Total entropy (predictive): {predictive_entropy_c.mean():.4f}")
    print(f"  Aleatoric (expected entropy): {expected_entropy_c.mean():.4f}")
    print(f"  Epistemic (mutual information): {mutual_information_c.mean():.4f}")

# Bar chart comparing uncertainty across conditions
condition_names = list(results_harder.keys())
total_vals = [results_harder[c]['total_entropy'] for c in condition_names]
aleatoric_vals = [results_harder[c]['aleatoric'] for c in condition_names]
epistemic_vals = [results_harder[c]['epistemic'] for c in condition_names]

x = np.arange(len(condition_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width, total_vals, width, label='Total (Predictive Entropy)')
ax.bar(x, aleatoric_vals, width, label='Aleatoric (Expected Entropy)')
ax.bar(x + width, epistemic_vals, width, label='Epistemic (Mutual Information)')
ax.set_ylabel('Uncertainty')
ax.set_title('Uncertainty Under Different Conditions')
ax.set_xticks(x)
ax.set_xticklabels(condition_names, rotation=20, ha='right')
ax.legend()
plt.tight_layout()
plt.show()

# Show example corrupted images
sample_img, sample_lbl = val_loader.dataset[0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ['Clean', 'Gaussian Noise', 'Blurred', 'Center Crop (50%)']
images_to_show = [
    sample_img,
    add_gaussian_noise(sample_img, std=0.3),
    apply_blur(sample_img, kernel_size=15, sigma=5.0),
    apply_center_crop(sample_img, crop_fraction=0.5),
]
for ax, img, title in zip(axes, images_to_show, titles):
    ax.imshow(transform_image(np.transpose(img.numpy(), (1, 2, 0))))
    ax.set_title(title)
    ax.axis('off')
fig.suptitle('Example Corrupted Images')
plt.tight_layout()
plt.show()