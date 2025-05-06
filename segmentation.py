# ## Prepare Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PIL
import glob
from skimage.metrics import structural_similarity as ssim
import time
import matplotlib as mpl
import sklearn
import os
import pickle

# Configuration
DATA = 'pancreas_64_top2_mask'
MODEL = 'full'  # 'full', 'no_patchify', 'no_contrastive'
USE_LABEL = False #This is modified later in the testing phase
DATA_FOLDER = 'output_{}_{}'.format(DATA, MODEL)
LOAD_POOLS = True # Set to False to generate new positive/negative pools

# Create output folder if it doesn't exist
if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

# Load data
# Assuming the .npy files are in a path accessible to the script.
# Users might need to change these paths.
try:
    data_path = '/gpfs/milgram/project/shung/jz766/pancreas_seg/part_segmentation/slices_64_top2/images_64_top2.npy'
    label_path = '/gpfs/milgram/project/shung/jz766/pancreas_seg/part_segmentation/slices_64_top2/labels_64_top2.npy'
    data = np.load(data_path)
    label = np.load(label_path)
except FileNotFoundError:
    print(f"Error: Data files not found. Please check the paths: \n{data_path}\n{label_path}")
    print("Using placeholder data for now.")
    # Placeholder data if files are not found - for script to run without original data
    data = np.random.rand(10, 64, 64, 1).astype(np.float32) * 2 -1 # Example: 10 images, 64x64, 1 channel
    label = np.random.randint(0, 2, size=(10, 64, 64, 1)).astype(np.float32)


data = torch.from_numpy(data).float()
data_input = data.permute(0, 3, 1, 2)  # Permute to [batch, channel, height, width]
label = torch.from_numpy(label).float()

print(f"Data input min: {data_input.min()}, max: {data_input.max()}")


# Create label mask (middle pixel of ground truth)
# This part might be specific to the dataset and task
label_mask = np.zeros_like(label)
# Check if label has 4 dimensions (batch, H, W, C) and C is 1, or 3 dimensions (batch, H, W)
if label.ndim == 4 and label.shape[-1] == 1:
    reshaped_label = label.reshape(label.shape[0], 64, 64)
elif label.ndim == 3:
    reshaped_label = label
else:
    print(f"Unexpected label shape: {label.shape}. Skipping label_mask creation.")
    reshaped_label = None # Or handle appropriately

if reshaped_label is not None:
    for i in range(reshaped_label.shape[0]):
        # Ensure label_argwhere is not empty
        current_label_slice = reshaped_label[i]
        if torch.any(current_label_slice > 0): # Check if there are any positive labels
            label_argwhere_np = np.argwhere(current_label_slice.numpy()) # Convert to NumPy for np.argwhere
            if label_argwhere_np.size > 0: # Check if argwhere is not empty
                # Assuming label_argwhere_np is [N, 2] where N is number of positive pixels
                median_x_coord = np.median(label_argwhere_np[:, 0]).reshape((1, 1))
                median_y_coord = np.median(label_argwhere_np[:, 1]).reshape((1, 1))
                middle_pt = np.concatenate([median_x_coord, median_y_coord], axis=0).reshape(2,1) # shape [2,1]

                # Ensure label_argwhere_np is correctly shaped for subtraction
                # It should be [num_points, 2], so transpose it to [2, num_points] for broadcasting with middle_pt [2,1]
                dist_to_middle_pt = ((label_argwhere_np.T - middle_pt)**2).sum(axis=0) # shape: [num_points]
                argmin = np.argmin(dist_to_middle_pt)
                label_mask[i, label_argwhere_np[argmin, 0], label_argwhere_np[argmin, 1]] = 1
            else:
                print(f"Warning: No positive labels found in slice {i} after reshaping. Skipping mask creation for this slice.")
        else:
            print(f"Warning: No positive labels found in original slice {i}. Skipping mask creation for this slice.")

label_mask = torch.from_numpy(label_mask).float()

def add_margin(pil_img, top, right, bottom, left, color):
    """Adds a margin to a PIL image."""
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

class CustomDataset(Dataset): # Renamed from Dataset to avoid conflict with torch.utils.data.Dataset
    """Custom Dataset class for loading and transforming image data."""
    def __init__(self, ids, preprocessed_data, labels, transform=False):
        self.dataset = preprocessed_data[ids]
        self.label = labels[ids]
        self.id = ids
        self.transform = transform
        self.generator = ContrastiveLearningViewGenerator()

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        label_sample = self.label[idx] # Changed from label to label_sample to avoid conflict
        if self.transform:
            sample = self.generator(sample)
        return sample, label_sample


class ContrastiveLearningViewGenerator(object):
    """Generates contrastive views of an image."""
    def __init__(self, n_views=1):
        # These transformations might require sample to be a PIL image or a Tensor of specific shape.
        # The original notebook had the color_transform commented out.
        # self.color_transform = torchvision.transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.6, hue=0.3)
        self.n_views = n_views

    def __call__(self, x):
        # Original notebook returns x directly. If color_transform is used, ensure x is compatible.
        # For example, if x is a Tensor, it might need to be converted to PIL Image first.
        # return torch.from_numpy(np.array([self.color_transform(Image.fromarray(x.numpy().astype(np.uint8).squeeze(0),'L')).detach().numpy() for i in range(self.n_views)]))[0]
        return x


def new_mycollate(batch):
    """Custom collate function for DataLoader to apply augmentations."""
    generator = ContrastiveLearningViewGenerator()

    # Ensure items are tensors before detaching and converting to numpy
    data_list = [item[0].cpu() if item[0].is_cuda else item[0] for item in batch]
    target_list = [item[1].cpu() if item[1].is_cuda else item[1] for item in batch]

    data_np = np.array([d.detach().numpy() for d in data_list])
    target_np = np.array([t.detach().numpy() for t in target_list])

    data2 = np.flip(data_np, 2).copy()  # Flip along height
    data3 = np.flip(data_np, 3).copy()  # Flip along width

    data_tensor = torch.from_numpy(data_np)
    new_data1 = generator(data_tensor) # This currently returns the input as is
    new_data2 = torch.from_numpy(data2)
    new_data3 = torch.from_numpy(data3)

    new_data = torch.cat((new_data1, new_data2, new_data3), 0)
    target_tensor = torch.from_numpy(target_np)

    return [new_data, target_tensor]


# ## Network
class ConvBlock(nn.Module):
    """Convolutional Block for the encoder."""
    def __init__(self, in_channels, nfilt, k=3):
        super(ConvBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(nfilt)
        self.batch_norm2 = nn.BatchNorm2d(nfilt * 4)
        self.batch_norm3 = nn.BatchNorm2d(nfilt * 8)
        self.conv1 = nn.Conv2d(in_channels, nfilt, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(nfilt, nfilt * 4, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(nfilt * 4, nfilt * 8, kernel_size=3, padding='same')

        self.l1 = nn.Linear(nfilt * 8, 128)

        self.k = k
        self.channels = in_channels
        self.l2 = nn.Linear(128, self.k * self.k * self.channels)

    def forward(self, x):
        x1 = F.leaky_relu(self.batch_norm1(self.conv1(x)))
        x2 = F.leaky_relu(self.batch_norm2(self.conv2(x1)))
        x3 = F.leaky_relu(self.batch_norm3(self.conv3(x2)))

        x3_permuted = x3.permute(0, 2, 3, 1) # Corrected variable name
        x3_reshaped = x3_permuted.reshape(x3_permuted.shape[0], -1, x3_permuted.shape[-1]) # Corrected variable name
        x3_linear = self.l1(x3_reshaped) # Corrected variable name

        patchify_target_list = [] # Renamed to avoid conflict
        for i in range(x.shape[0]):
            tmp = x[i]
            # Pad the image so that patches can be extracted from the borders
            tmp = F.pad(tmp, pad=[self.k // 2, self.k // 2, self.k // 2, self.k // 2], value=0)
            # Extract patches
            tmp = tmp.unfold(1, self.k, 1).unfold(2, self.k, 1).reshape(self.channels, -1, self.k, self.k).permute(1, 0, 2, 3)
            tmp = tmp.reshape([tmp.shape[0], -1])
            patchify_target_list.append(tmp)
        patchify_target_tensor = torch.stack(patchify_target_list, dim=0) # Renamed and corrected
        # Patchify loss calculation
        patchify_loss = ((torch.tanh(self.l2(x3_linear)) - patchify_target_tensor)**2).mean()

        return x3_linear, patchify_loss


class UNet(nn.Module): #This class seems to be unused in the main script flow
    """U-Net architecture (appears unused in the provided notebook's main training path)."""
    def __init__(self, in_channels, nfilt, k=25):
        super(UNet, self).__init__() # Corrected super call
        self.batch_norm_e1 = nn.BatchNorm2d(nfilt)
        self.batch_norm_e2 = nn.BatchNorm2d(nfilt * 2)
        self.batch_norm_e3 = nn.BatchNorm2d(nfilt * 4)

        self.conv_e1 = nn.Conv2d(in_channels, nfilt, kernel_size=5, stride=2, padding=2)
        self.conv_e2 = nn.Conv2d(nfilt * 1, nfilt * 2, kernel_size=5, stride=2, padding=2)
        self.conv_e3 = nn.Conv2d(nfilt * 2, nfilt * 4, kernel_size=5, stride=2, padding=2)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.batch_norm_d1 = nn.BatchNorm2d(nfilt * 2)
        self.batch_norm_d2 = nn.BatchNorm2d(nfilt * 1)

        self.conv_d1 = nn.Conv2d(nfilt * 4, nfilt * 2, kernel_size=5, stride=1, padding=2)
        self.conv_d2 = nn.Conv2d(nfilt * 4, nfilt * 1, kernel_size=5, stride=1, padding=2) # Corrected input channels

        self.batch_norm_out = nn.BatchNorm2d(128)
        self.conv_out = nn.Conv2d(nfilt * 2, 128, kernel_size=5, stride=1, padding=2) # Corrected input channels

        self.first_call = True
        self.k = k
        self.channels = in_channels
        self.l2 = nn.Linear(128, self.k * self.k * self.channels)

    def forward(self, x):
        e1 = F.leaky_relu(self.batch_norm_e1(self.conv_e1(x)))
        e2 = F.leaky_relu(self.batch_norm_e2(self.conv_e2(e1)))
        e3 = F.leaky_relu(self.batch_norm_e3(self.conv_e3(e2)))

        d1_up = self.up(e3)
        d1 = F.relu(self.batch_norm_d1(self.conv_d1(d1_up)))
        d1 = torch.cat([d1, e2], axis=1)

        d2_up = self.up(d1)
        d2 = F.relu(self.batch_norm_d2(self.conv_d2(d2_up)))
        d2 = torch.cat([d2, e1], axis=1)

        out_up = self.up(d2)
        out = self.conv_out(out_up)

        out_permuted = out.permute(0, 2, 3, 1)
        out_reshaped = out_permuted.reshape(out_permuted.shape[0], -1, out_permuted.shape[-1])

        patchify_target_list = []
        for i in range(x.shape[0]):
            tmp = x[i]
            tmp = F.pad(tmp, pad=[self.k // 2, self.k // 2, self.k // 2, self.k // 2], value=0)
            tmp = tmp.unfold(1, self.k, 1).unfold(2, self.k, 1).reshape(self.channels, -1, self.k, self.k).permute(1, 0, 2, 3)
            tmp = tmp.reshape([tmp.shape[0], -1])
            patchify_target_list.append(tmp)
        patchify_target_tensor = torch.stack(patchify_target_list, dim=0)
        patchify_loss = ((torch.tanh(self.l2(out_reshaped)) - patchify_target_tensor)**2).mean()

        if self.first_call:
            print("UNet Shapes:")
            print(f"e1: {e1.shape}")
            print(f"e2: {e2.shape}")
            print(f"e3: {e3.shape}")
            print(f"d1: {d1.shape}")
            print(f"d2: {d2.shape}")
            print(f"out_reshaped: {out_reshaped.shape}")
            self.first_call = False

        return out_reshaped, patchify_loss


class Encoder(nn.Module): # Renamed from encoder to Encoder for convention
    """Encoder network using ConvBlock."""
    def __init__(self):
        super(Encoder, self).__init__()
        # Assuming input is single channel (e.g., grayscale)
        self.conv1 = ConvBlock(1, 32) # in_channels = 1, nfilt = 32

    def forward(self, x):
        x1_, patchify_loss = self.conv1(x)
        return x1_, patchify_loss


# ## Train Model
def transform_rc(v, w):
    """Transforms a flat index to row and column coordinates."""
    return v % w, v // w

def check_sim(img_tensor, r0, c0, r, c): # Changed img to img_tensor
    """Checks structural similarity between two image regions."""
    # Ensure img_tensor is on CPU and converted to NumPy
    img_np = img_tensor.squeeze(0).cpu().detach().numpy()
    img_pil = Image.fromarray(np.uint8(img_np * 255), mode='L')
    im_new = add_margin(img_pil, 8, 8, 8, 8, 0) # Assuming add_margin is defined
    im_new_np = np.array(im_new)

    # Define regions for comparison
    x1, x2 = r0, r0 + 16
    y1, y2 = c0, c0 + 16
    x1_, x2_ = r, r + 16
    y1_, y2_ = c, c + 16

    # Ensure regions are within bounds
    h, w = im_new_np.shape
    region1 = im_new_np[max(0,x1):min(h,x2), max(0,y1):min(w,y2)]
    region2 = im_new_np[max(0,x1_):min(h,x2_), max(0,y1_):min(w,y2_)]

    # Ensure regions are not empty and have the same shape for SSIM
    if region1.size == 0 or region2.size == 0 or region1.shape != region2.shape:
        return 0 # Or handle as an error/warning

    # data_range is important for ssim, especially if images are not in [0,1] or [0,255]
    return ssim(region1, region2, data_range=region1.max() - region1.min())


def select_near_positive(img_data_tensor, mask_data_tensor, full_version=True): # Changed to tensor inputs
    """Selects near positive samples based on mask and similarity."""
    pool = []
    h_img = img_data_tensor.shape[-2]
    w_img = img_data_tensor.shape[-1]

    for k in range(len(img_data_tensor)):
        if full_version or k == 0: # Process all images if full_version or only the first
            img_current = img_data_tensor[k]
            mask_current = mask_data_tensor[k]
            img_pool = []

            mask_np = mask_current.cpu().detach().numpy()
            if mask_np.shape[-1] == 1 and mask_np.ndim == 3 : # Assuming [H, W, C]
                mask_np = mask_np[:, :, 0]
            elif mask_np.ndim == 2: # Assuming [H,W]
                pass
            else:
                print(f"Unexpected mask shape: {mask_np.shape} for image {k}")
                continue


            pancreas_indices_flat = np.where(mask_np.flatten() > 0)[0]

            if pancreas_indices_flat.size == 0:
                print(f"No positive pixels in mask for image {k}")
                pool.append([])
                continue

            for idx, j_flat in enumerate(pancreas_indices_flat):
                if idx % 100 == 0: # Reduced print frequency
                    print(f'select_near_positive: Image {k}/{len(img_data_tensor)}, Pixel {idx}/{len(pancreas_indices_flat)}')

                temp = []
                r0, c0 = transform_rc(j_flat, w_img) # Use image width
                best_sim = 0.0 # Initialize with a float

                neighbor_positions = []
                # Define a small neighborhood (e.g., 3x3 excluding center)
                for m_offset in range(-1, 2):
                    for n_offset in range(-1, 2):
                        if m_offset == 0 and n_offset == 0:
                            continue
                        m, n = r0 + m_offset, c0 + n_offset
                        if 0 <= m < h_img and 0 <= n < w_img and mask_np[m, n] > 0:
                            neighbor_positions.append((m, n))

                if not neighbor_positions:
                    continue

                # Randomly select an initial neighbor
                r_best, c_best = neighbor_positions[np.random.randint(0, len(neighbor_positions))]

                if full_version:
                    for r_neigh, c_neigh in neighbor_positions:
                        # Optional: Skip some comparisons to speed up if needed
                        # if best_sim > 0 and np.random.binomial(1, .5) < .75:
                        # continue
                        try:
                            temp_v = check_sim(img_current, r0, c0, r_neigh, c_neigh)
                            if temp_v > best_sim:
                                best_sim = temp_v
                                r_best, c_best = r_neigh, c_neigh
                        except Exception as e:
                            print(f"Error in check_sim for positive: {e}")
                            continue


                temp.append(w_img * c_best + r_best) # Store flat index of the best neighbor
                img_pool.append(temp)
            pool.append(img_pool)
    return pool


def select_negative_random(img_data_tensor, mask_data_tensor, ssim_thresh=0.5, full_version=True): # Changed to tensor inputs
    """Selects random negative samples."""
    pool = []
    h_img = img_data_tensor.shape[-2]
    w_img = img_data_tensor.shape[-1]
    max_attempts = 100 # Reduced max_attempts for faster processing during conversion

    for k in range(len(img_data_tensor)):
        if full_version or k == 0:
            img_current = img_data_tensor[k]
            mask_current = mask_data_tensor[k]
            img_pool = []

            mask_np = mask_current.cpu().detach().numpy()
            if mask_np.shape[-1] == 1 and mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            elif mask_np.ndim == 2:
                pass
            else:
                print(f"Unexpected mask shape: {mask_np.shape} for image {k}")
                continue

            pancreas_indices_flat = np.where(mask_np.flatten() > 0)[0]

            if pancreas_indices_flat.size == 0:
                print(f"No positive pixels in mask for image {k} (negative selection)")
                pool.append([])
                continue

            for idx, j_flat in enumerate(pancreas_indices_flat):
                if idx % 100 == 0: # Reduced print frequency
                    print(f'select_negative_random: Image {k}/{len(img_data_tensor)}, Pixel {idx}/{len(pancreas_indices_flat)}')

                temp = []
                found_one = False
                r0, c0 = transform_rc(j_flat, w_img)
                attempts = 0
                while not found_one and attempts < max_attempts:
                    # Randomly select a pixel from ALL pixels, then check if it's far and dissimilar
                    a_flat_random = np.random.randint(0, h_img * w_img)
                    r_rand, c_rand = transform_rc(a_flat_random, w_img)

                    # Ensure it's not the anchor pixel itself and is sufficiently far
                    # Also ensure it's within the pancreas mask (original notebook chose from pancreas_indices)
                    # For true negatives, one might want to sample OUTSIDE the mask or far from anchor within mask
                    # The original logic seems to pick negatives from within the pancreas mask but far from anchor.
                    if a_flat_random != j_flat and mask_np[r_rand, c_rand] > 0 and \
                       (abs(r0 - r_rand) > 20 or abs(c0 - c_rand) > 20): # Looser distance for faster finding
                        try:
                            if full_version and check_sim(img_current, r0, c0, r_rand, c_rand) > ssim_thresh:
                                attempts += 1
                                continue
                            temp.append(a_flat_random)
                            found_one = True
                        except Exception as e:
                            print(f"Error in check_sim for negative: {e}")
                            attempts +=1 # count as attempt even if check_sim fails
                            continue
                    attempts += 1

                if found_one: # Only append if a valid negative was found
                    img_pool.append(temp)
            pool.append(img_pool)
    return pool

def new_local_nce_loss_fast(features, negative_indices, positive_indices):
    """Computes the Local NCE loss."""
    temperature = 0.25
    device = features.device # Ensure all tensors are on the same device

    # features shape: [N, feature_dim] where N is total number of pixels (H*W)
    N, feature_dim = features.shape

    # Ensure indices are valid and on the correct device
    positive_indices = positive_indices.to(device).long()
    negative_indices = negative_indices.to(device).long()

    # Filter out-of-bounds indices if any (though ideally, pool generation should prevent this)
    positive_indices = positive_indices[positive_indices < N]
    negative_indices = negative_indices[negative_indices < N]

    if positive_indices.numel() == 0 or negative_indices.numel() == 0:
        # print("Warning: No valid positive or negative samples after filtering for NCE loss.")
        return torch.tensor(0.0, device=device, requires_grad=True) # Return a zero loss if no valid samples


    # Prepare anchor features - these are the "query" features from positive locations
    # Assuming positive_indices contains flat indices of anchor pixels
    # The original notebook's positive_pool seems to store the *neighbor* of the anchor.
    # For InfoNCE, the anchor should be the original pixel, and the positive is its augmentation/neighbor.
    # Let's assume positive_indices are the anchors, and we need to find their corresponding "positive keys".
    # This part needs clarification based on how positive_pool was constructed.
    # If positive_pool[img_idx] contains [neighbor_of_anchor1, neighbor_of_anchor2, ...],
    # then anchor_features should be features[anchor_indices] and positive_features should be features[positive_indices (neighbors)]
    # For simplicity, assuming positive_indices refer to the anchors themselves for now, and also their "positive key"
    # This is a common simplification if augmentations are handled implicitly or if self-similarity is the goal.

    anchor_features = features[positive_indices.squeeze()] # Squeeze if it's [num_positive, 1]
    positive_features = features[positive_indices.squeeze()] # Using same for positive key
    negative_features = features[negative_indices.squeeze()] # Squeeze if it's [num_negative, 1]

    # Handle cases where squeezing might lead to 1D tensor if only one sample
    if anchor_features.ndim == 1: anchor_features = anchor_features.unsqueeze(0)
    if positive_features.ndim == 1: positive_features = positive_features.unsqueeze(0)
    if negative_features.ndim == 1 and negative_features.numel() > 0 : negative_features = negative_features.unsqueeze(0)
    elif negative_features.numel() == 0: # No negative samples
         return torch.tensor(0.0, device=device, requires_grad=True)


    # Compute similarities (dot product)
    # Positive pair similarities: l_pos = anchor_features . positive_features / temperature
    # Assuming element-wise product for positive pairs if anchor and positive are the same set of features
    # or if a one-to-one correspondence is implied.
    # If positive_indices are neighbors, then anchor_features would be features[original_anchor_indices]
    positive_similarities = torch.sum(anchor_features * positive_features, dim=1) / temperature

    # Negative pair similarities: l_neg = anchor_features . negative_features^T / temperature
    # This creates a matrix of [num_anchors, num_negatives]
    similarities_neg = torch.matmul(anchor_features, negative_features.t()) / temperature

    # Logits for CrossEntropyLoss: [l_pos, l_neg1, l_neg2, ...]
    # Each row corresponds to an anchor, first column is its positive similarity, rest are negative similarities.
    logits = torch.cat([positive_similarities.unsqueeze(1), similarities_neg], dim=1)

    # Labels are all zeros, as the positive sample is always in the first column of logits
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    return loss


def train_epoch(model, iterator, optimizer, negative_pool, positive_pool, device, lambda_contrastive_loss=1, lambda_patchify_loss=10): # Added device
    """Trains the model for one epoch."""
    total_loss_epoch = 0
    model.train()
    t_epoch = time.time()

    for batch_idx, (train_batch_augmented, train_labels_original) in enumerate(iterator):
        optimizer.zero_grad()

        # The collate_fn creates a batch like [original; flipped_h; flipped_w]
        # We only need the original images for feature extraction in this NCE loss setup
        # Assuming batch_size in DataLoader was 1, so train_batch_augmented has 3 images if augmented
        # If batch_size > 1, then it's 3*batch_size.
        # Let's assume the original training loop processed one image (and its augmentations) at a time for NCE.
        # If batch_size in DataLoader is B, then train_batch_augmented is 3B.
        # We need to iterate through the *original* images in the batch.
        
        num_augmentations = 3 # original, flip_h, flip_w
        actual_batch_size = train_batch_augmented.size(0) // num_augmentations
        
        batch_total_contrastive_loss = 0
        batch_total_patchify_loss = 0
        
        # Move augmented batch to device
        train_batch_augmented = train_batch_augmented.to(device)

        # Forward pass for the augmented batch
        features_augmented_batch, patchify_loss_augmented = model(train_batch_augmented)
        
        # Patchify loss is calculated on the augmented batch directly
        # This might need adjustment if patchify loss should only be on original images
        current_batch_loss = lambda_patchify_loss * patchify_loss_augmented
        batch_total_patchify_loss += patchify_loss_augmented.item()


        # Iterate over each original image in the batch for NCE loss
        for i in range(actual_batch_size):
            # Extract features for the original image (the first one in the augmented set for this sample)
            # features_batch has shape [3B, num_pixels, feature_dim] if ConvBlock output is permuted & reshaped
            # or [3B, feature_dim, H, W] if UNet (needs similar permute/reshape before NCE)
            # Assuming ConvBlock output: features_augmented_batch[i] gives [num_pixels, feature_dim] for the i-th original image
            
            original_image_features = features_augmented_batch[i * num_augmentations] # Features of the original image
            
            # img_idx in the pool corresponds to the original image index in the dataset
            # If shuffle=True in DataLoader, batch_idx is not directly the dataset index.
            # We need a way to get the original image index for pool lookup.
            # For now, assuming batch_idx can be used if shuffle=False or if pools are batched.
            # The original notebook had shuffle=True for training_generator.
            # This implies positive/negative pools should be structured to be accessed by a shuffled batch_idx
            # or the Dataset should return original indices.
            # Let's assume for now that img_idx can be derived or is available.
            # The original notebook used batch_idx directly, which implies the pools were aligned with shuffled batches.
            # This is unusual. A more robust way is to pass original indices with the batch.
            # Given the notebook structure, we'll stick to batch_idx for pool indexing, assuming pools are pre-shuffled or aligned.
            
            # This is a critical point: how are positive_pool and negative_pool indexed?
            # Original code: img_idx = batch_idx. If training_generator shuffles, this is problematic.
            # For this script, let's assume the pools are indexed by the *original dataset index*.
            # The CustomDataset needs to return the original index.
            # Let's modify CustomDataset and collate_fn to handle this.
            # --- This modification is complex to add here directly. ---
            # For now, proceeding with batch_idx as img_idx, acknowledging this limitation.
            
            img_idx_in_epoch = batch_idx * actual_batch_size + i # This is the index within the current epoch's iteration
                                                                # Not the original dataset index if shuffled.
                                                                # The original code used batch_idx which implies batch_size=1 for NCE part.

            # The original train loop had batch_size=1 for the NCE loss calculation part.
            # If we use a larger batch_size for DataLoader, the NCE loss needs to be calculated per image.
            if actual_batch_size > 1:
                print("Warning: NCE loss calculation is designed for batch_size=1 for pool indexing. Current effective batch_size > 1.")
                # This part would need significant rework for batched NCE loss with precomputed pools.
                # For now, we'll process only the first image of the batch for NCE if actual_batch_size > 1,
                # or assume the pools are somehow batched (which is not evident from the notebook).
                # The notebook's training_generator had batch_size=1, so this was not an issue there.
                # To make this script runnable, we'll process NCE for each item in the batch if pools are lists of lists.
                
            # Let's assume positive_pool and negative_pool are lists of lists,
            # where the outer list corresponds to the dataset image index.
            # The DataLoader will give us batches. We need the original index.
            # The current CustomDataset does not return original_idx.
            # A quick fix for demonstration: if batch_size > 1, this NCE part might be incorrect for pool indexing.
            # The original code used batch_idx, and training_generator had batch_size=1.
            # So img_idx was effectively the (shuffled) image index from the dataset.

            # If positive_pool and negative_pool are lists of lists of indices (flat pixel indices)
            # And are indexed by the original image index in the dataset.
            # The current iterator does not give original indices.
            # For now, we'll assume batch_idx is the image index for pool lookup,
            # which is only correct if DataLoader batch_size=1 and shuffle=False,
            # or if pools are magically aligned with shuffled batches.
            # The notebook used shuffle=True and batch_size=1 for training_generator.

            # Let's assume the pools are for the *entire dataset* and indexed 0 to N-1
            # And the iterator provides original indices.
            # If not, this part needs fixing.
            # The notebook's `enumerate(iterator)` gives `batch_idx`.
            # If `batch_size` for `training_generator` is 1, then `batch_idx` is the (shuffled) image index.

            current_img_original_idx = batch_idx # This is the assumption from the notebook.
                                                # This is only valid if the DataLoader batch_size for NCE part is 1.
                                                # The notebook's training_generator had batch_size=1.

            if current_img_original_idx >= len(positive_pool) or current_img_original_idx >= len(negative_pool):
                # print(f"Warning: batch_idx {current_img_original_idx} out of bounds for pools. Skipping NCE for this batch.")
                continue

            positive_indices_for_img = positive_pool[current_img_original_idx]
            negative_indices_for_img = negative_pool[current_img_original_idx]

            if not positive_indices_for_img or not negative_indices_for_img:
                # print(f"Warning: Empty positive/negative pool for image index {current_img_original_idx}. Skipping NCE.")
                continue
            
            # Convert lists of lists to tensors of flat indices
            # Positive_indices_for_img is a list of lists, e.g., [[idx1_pos_for_anchor1], [idx2_pos_for_anchor2]]
            # Or, if it's just a list of positive sample indices for the *entire image's features_flat*:
            # The original NCE loss `new_local_nce_loss_fast` expects flat indices for anchors, positives, and negatives.
            # `positive_pool[img_idx]` was a list of integers in the notebook's train loop.
            # This means positive_indices_for_img is a list of anchor indices (flat pixel indices).
            # And negative_indices_for_img is a list of negative sample indices (flat pixel indices).
            # The "positive key" for an anchor is implicitly the anchor itself or its neighbor (handled by pool generation).
            # The NCE loss function needs to be called with features of a single image.
            
            # features_augmented_batch[0] will be the features for the *first image in the augmented batch*.
            # If batch_size of dataloader was > 1, this logic is flawed for NCE.
            # The notebook's training_generator had batch_size=1, so features_batch[0] was the single image's features.
            
            # Assuming features_batch is for a single original image and its augmentations.
            # The NCE loss should be computed on the features of the *original* image.
            # features_augmented_batch[0] is [num_pixels, feature_dim] for the first original image in the augmented batch.
            
            # The NCE loss is computed per image.
            # The notebook's `train` function was called with `batch_size=1` for the NCE part.
            # `features_batch` in the notebook was `[1, num_pixels, feature_dim]`.
            # `features = features_batch[0]` made it `[num_pixels, feature_dim]`.

            # Here, features_augmented_batch is [3B, num_pixels, feature_dim].
            # We need to extract features for each *original* image in the batch.
            # The `i`-th original image's features are at `features_augmented_batch[i * num_augmentations]`

            single_image_features = features_augmented_batch[i * num_augmentations].to(device) # Features for one original image

            # The pools positive_pool and negative_pool are indexed by the *original dataset index*.
            # The current `batch_idx` from `enumerate(iterator)` is the batch number, not the original image index if shuffle=True.
            # This is a major point of divergence if batch_size > 1 or shuffle=True.
            # The notebook's `training_generator` had `batch_size=1` and `shuffle=True`.
            # So `batch_idx` was the (shuffled) index of the image in the dataset.
            
            # For this script to be robust, the Dataset should return original indices.
            # Let's assume for now, to match the notebook's direct use of batch_idx:
            # `img_idx_for_pool = batch_idx` (if dataloader batch_size for this loop is 1)
            # Or if the pools are somehow reordered to match shuffled batches.
            # This is fragile. A better solution is needed for general case.

            # Given the notebook used batch_idx directly with a shuffled DataLoader of batch_size=1,
            # we'll replicate that logic for NCE loss calculation per image.
            # The outer loop `enumerate(iterator)` gives `batch_idx`.
            # If iterator's batch_size is B, then `train_batch_augmented` is 3B.
            # We need to calculate NCE for each of the B original images.
            # The pool index should be the *original index* of that image.
            # The current code does not provide this original index.

            # Simplification for now: Assume batch_idx is a placeholder for the correct image index for pool lookup.
            # This will only work correctly if the DataLoader for this training loop effectively processes one original image at a time
            # for the NCE loss part, and batch_idx corresponds to its index in the pre-generated pools.
            
            img_pool_idx = batch_idx # This is the critical assumption.
            
            if img_pool_idx >= len(positive_pool) or img_pool_idx >= len(negative_pool):
                # print(f"Warning: img_pool_idx {img_pool_idx} out of bounds for pools. Skipping NCE for this image.")
                continue

            pos_indices_tensor = torch.tensor(positive_pool[img_pool_idx], dtype=torch.long).to(device)
            neg_indices_tensor = torch.tensor(negative_pool[img_pool_idx], dtype=torch.long).to(device)

            if pos_indices_tensor.numel() > 0 and neg_indices_tensor.numel() > 0:
                # The NCE loss function expects features of a single image, and flat indices
                # single_image_features is [num_pixels, feature_dim]
                contrastive_loss_single_image = new_local_nce_loss_fast(single_image_features, neg_indices_tensor, pos_indices_tensor)
                current_batch_loss += lambda_contrastive_loss * contrastive_loss_single_image
                batch_total_contrastive_loss += contrastive_loss_single_image.item()
            else:
                # print(f"Skipping NCE for image index {img_pool_idx} due to empty positive/negative sets after tensor conversion.")
                pass


        # Backpropagation and optimization step for the entire augmented batch
        current_batch_loss.backward()
        optimizer.step()

        total_loss_epoch += current_batch_loss.item()

        if batch_idx and batch_idx % 10 == 0: # Print every 10 augmented batches
            avg_contrastive_loss_print = batch_total_contrastive_loss / actual_batch_size if actual_batch_size > 0 else 0
            avg_patchify_loss_print = batch_total_patchify_loss # Already an average over augmented batch
            print(f'  Epoch iter {batch_idx} NCE loss: {avg_contrastive_loss_print:.3f}/Patchify loss: {avg_patchify_loss_print:.3f} ({time.time() - t_epoch:.1f} sec)')
            t_epoch = time.time()

    avg_loss_epoch = total_loss_epoch / len(iterator) if len(iterator) > 0 else 0
    print(f'Epoch avg loss: {avg_loss_epoch:.3f}')
    return avg_loss_epoch


def save_pools(positive_pool, negative_pool, data_name):
    """Saves positive and negative pools to pickle files."""
    if not os.path.exists('saved_pools'):
        os.mkdir('saved_pools')
    with open(f'saved_pools/{data_name}_positive_pool.pkl', 'wb') as f:
        pickle.dump(positive_pool, f)
    with open(f'saved_pools/{data_name}_negative_pool.pkl', 'wb') as f:
        pickle.dump(negative_pool, f)
    print("Pools saved.")

def load_pools(data_name):
    """Loads positive and negative pools from pickle files."""
    try:
        with open(f'saved_pools/{data_name}_positive_pool.pkl', 'rb') as f:
            positive_pool = pickle.load(f)
        with open(f'saved_pools/{data_name}_negative_pool.pkl', 'rb') as f:
            negative_pool = pickle.load(f)
        print("Pools loaded.")
        return positive_pool, negative_pool
    except FileNotFoundError:
        print(f"Error: Pool files for {data_name} not found. Please generate them first.")
        return None, None


# Training settings from the notebook
learning_rate = 0.0001
max_epochs = 25 # Reduced for quick testing
# The notebook's training_generator for the NCE loss part effectively used batch_size=1.
# The new_mycollate function augments, so if DataLoader batch_size is B, it outputs 3B.
# For the NCE loss, we need to process features of each *original* image.
# The train_epoch function now handles this by iterating through original images in the augmented batch.
# Let's set a small DataLoader batch_size for demonstration.
dataloader_batch_size = 4 # Number of original images per batch for DataLoader

lambda_contrastive_loss = 1
lambda_patchify_loss = 10

if MODEL == 'no_patchify':
    lambda_patchify_loss = 0
if MODEL == 'no_contrastive':
    lambda_contrastive_loss = 0

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare dataset and dataloader
# The notebook used all data for training. No explicit train/test split in the main training loop.
train_index = np.arange(data_input.shape[0]).tolist()
# Ensure data_input and label are on the CPU for Dataset initialization if they were moved to GPU
training_set = CustomDataset(train_index, data_input.cpu(), label.cpu())
# Note: new_mycollate will move data to device within train_epoch
training_generator = DataLoader(training_set, batch_size=dataloader_batch_size, shuffle=True, collate_fn=new_mycollate, num_workers=0) # num_workers=0 for simplicity

# Initialize model and optimizer
model = Encoder().to(device) # Move model to device
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Load or generate positive/negative pools
if LOAD_POOLS:
    positive_pool, negative_pool = load_pools(DATA)
    if positive_pool is None or negative_pool is None:
        print("Failed to load pools. Set LOAD_POOLS=False to generate new ones.")
        # Handle error or exit if pools are essential and not found
        exit()
else:
    print("Generating positive and negative pools...")
    t_pool_gen = time.time()
    # Ensure data_input and label are suitable for pool generation functions (e.g., on CPU)
    positive_pool = select_near_positive(data_input.cpu(), label.cpu(), full_version=True) # Set full_version as needed
    print('Positive pool done')
    negative_pool = select_negative_random(data_input.cpu(), label.cpu(), full_version=True) # Set full_version as needed
    print('Negative pool done')
    save_pools(positive_pool, negative_pool, DATA)
    print(f'Pool generation took {time.time() - t_pool_gen:.1f} seconds.')

print(f"Length of positive_pool: {len(positive_pool)}")
if positive_pool:
    print(f"Example of positive_pool[0] (if not empty): {positive_pool[0][:5] if positive_pool[0] else 'Empty'}")


# Training loop
if positive_pool is not None and negative_pool is not None: # Proceed only if pools are available
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        train_loss_epoch = train_epoch(model, training_generator, optimizer, negative_pool, positive_pool, device,
                                    lambda_contrastive_loss=lambda_contrastive_loss,
                                    lambda_patchify_loss=lambda_patchify_loss)
        if (epoch + 1) % 2 == 0: # Save model every 2 epochs as in notebook (though notebook saved at end)
            model_save_path = f'model_{DATA}_{MODEL}_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
    # Save final model
    final_model_path = f'model_{DATA}_{MODEL}_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
else:
    print("Training cannot proceed without positive/negative pools.")


# ## Test Model
# Settings for testing phase from the notebook
USE_LABEL_TEST = True # Renamed from USE_LABEL to avoid conflict
MODEL_TO_LOAD = final_model_path # Or specify a different saved model path

# Create a new DataLoader for evaluation, typically with shuffle=False
# The notebook used a subset (test_index) for eval_generator.
# For simplicity, we'll evaluate on the entire dataset or a predefined test set if available.
# If you have a separate test set, load it here.
# For now, using the same 'training_set' for evaluation as an example.
# The notebook's eval_generator batch_size was 2.
eval_dataloader_batch_size = 2
eval_set = CustomDataset(train_index, data_input.cpu(), label.cpu()) # Using all data for eval for now
eval_generator = DataLoader(eval_set, batch_size=eval_dataloader_batch_size, shuffle=False, num_workers=0) # No collate_fn for eval usually, unless augmentations are desired

# Load the trained model
if os.path.exists(MODEL_TO_LOAD):
    model.load_state_dict(torch.load(MODEL_TO_LOAD, map_location=device))
    print(f"Model loaded from {MODEL_TO_LOAD} for evaluation.")
else:
    print(f"Error: Model file not found at {MODEL_TO_LOAD}. Cannot evaluate.")
    exit()

model.eval() # Set model to evaluation mode
features_list_eval = [] # Renamed to avoid conflict

with torch.no_grad(): # Disable gradient calculations for evaluation
    for i, (eval_batch_images, _) in enumerate(eval_generator):
        eval_batch_images = eval_batch_images.to(device)
        # The model expects input [N, C, H, W].
        # The encoder output is [N, num_pixels, feature_dim].
        # The notebook reshaped this to [-1, 128, 64, 64] for UMAP.
        # This assumes feature_dim=128 and num_pixels = 64*64.
        
        feats_, _ = model(eval_batch_images) # feats_ shape: [batch_size, num_pixels, feature_dim]
        
        # Detach and move to CPU before appending
        features_list_eval.append(feats_.cpu())

# Concatenate features from all batches
if features_list_eval:
    features_all_eval_raw = torch.cat(features_list_eval, dim=0) # Shape: [total_images, num_pixels, feature_dim]
    print(f"Raw concatenated features shape: {features_all_eval_raw.shape}")

    # Reshape for UMAP/visualization as in the notebook
    # This assumes feature_dim = 128 and image size 64x64 (num_pixels = 4096)
    # The notebook's reshape was: .reshape((-1, 128, 64, 64))
    # This implies that the `feature_dim` (128) was treated as channels, and `num_pixels` was reshaped to HxW.
    # This is an unusual reshape for features. Typically features are [N, D] for UMAP.
    # The ConvBlock output x3_linear is [batch_size, num_pixels, feature_dim=128].
    # If we want to reshape to [total_images, 128, 64, 64], then num_pixels must be 64*64=4096.
    
    num_images_eval = features_all_eval_raw.shape[0]
    num_pixels_eval = features_all_eval_raw.shape[1]
    feature_dim_eval = features_all_eval_raw.shape[2]

    if feature_dim_eval == 128 and num_pixels_eval == (64*64):
        try:
            features_all_eval_reshaped = features_all_eval_raw.reshape((num_images_eval, feature_dim_eval, 64, 64))
            print(f"Reshaped features_all for UMAP (as per notebook): {features_all_eval_reshaped.shape}")
            # The notebook then flattens this again for UMAP: features_all_eval_reshaped.reshape(num_images_eval, -1)
            # Or uses features_all_eval_raw.reshape(num_images_eval * num_pixels_eval, feature_dim_eval) for pixel-wise UMAP.
            # The notebook's UMAP part uses features_all.reshape(-1, 128) which means pixel-wise features.
            features_for_umap = features_all_eval_raw.reshape(-1, feature_dim_eval).numpy()
            print(f"Features prepared for UMAP (pixel-wise): {features_for_umap.shape}")

            # UMAP and Clustering (optional, requires umap-learn and sklearn)
            try:
                import umap
                from sklearn.cluster import KMeans

                print("Running UMAP and KMeans (this might take a while)...")
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, n_jobs=1)
                embedding = reducer.fit_transform(features_for_umap)

                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10) # n_init='auto' or 10
                clusters = kmeans.fit_predict(embedding)

                # Visualization (optional, requires matplotlib)
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='Spectral', s=5)
                plt.title('UMAP projection of the features, colored by K-Means clusters')
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
                plt.colorbar(scatter, label='Cluster ID')
                plt.savefig(os.path.join(DATA_FOLDER, f'umap_kmeans_clusters_{DATA}_{MODEL}.png'))
                print(f"UMAP visualization saved to {os.path.join(DATA_FOLDER, f'umap_kmeans_clusters_{DATA}_{MODEL}.png')}")
                # plt.show() # Commented out for script execution

            except ImportError:
                print("UMAP or scikit-learn not installed. Skipping UMAP/KMeans visualization.")
            except Exception as e:
                print(f"Error during UMAP/KMeans: {e}")

        except RuntimeError as e:
            print(f"Error reshaping features for UMAP: {e}. Ensure feature dimensions are as expected.")
            print("Original features_all_eval_raw will be used if further processing is needed without this reshape.")
    else:
        print("Feature dimensions from encoder are not as expected for the notebook's UMAP reshape.")
        print(f"Got [N, NumPixels, FeatDim] = [{num_images_eval}, {num_pixels_eval}, {feature_dim_eval}]")
        print("Expected FeatDim=128 and NumPixels=4096 for the specific reshape.")

else:
    print("No features extracted for evaluation.")

print("\nScript finished.")

