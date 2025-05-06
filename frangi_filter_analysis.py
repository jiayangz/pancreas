import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.filters import frangi
from tqdm import tqdm

def load_image_and_mask(image_path, mask_path):
    """Load medical image and its corresponding mask."""
    img = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    return img, mask

def crop_image_by_mask(image, mask):
    """Crop image based on the bounding box of the mask."""
    # Convert to numpy arrays
    img_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    
    # Find the bounding box
    nonzero_indices = np.where(mask_array > 0)
    if len(nonzero_indices[0]) == 0:  # Empty mask
        return None
    
    min_z, max_z = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    min_y, max_y = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    min_x, max_x = np.min(nonzero_indices[2]), np.max(nonzero_indices[2])
    
    # Add margin (optional)
    margin = 5
    min_z = max(0, min_z - margin)
    max_z = min(img_array.shape[0] - 1, max_z + margin)
    min_y = max(0, min_y - margin)
    max_y = min(img_array.shape[1] - 1, max_y + margin)
    min_x = max(0, min_x - margin)
    max_x = min(img_array.shape[2] - 1, max_x + margin)
    
    # Crop both image and mask
    cropped_img = img_array[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    cropped_mask = mask_array[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    
    return cropped_img, cropped_mask

def apply_frangi_filter(image):
    """Apply Frangi filter to each slice of a 3D image."""
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        slice_img = image[i]
        # Apply Frangi filter to enhance vessel-like structures
        # Adjust parameters as needed
        frangi_result = frangi(slice_img, scale_range=(1, 3), scale_step=1)
        result[i] = frangi_result
    return result

def find_largest_mask_slices(mask, n=2):
    """Find n slices with the largest segmentation area."""
    areas = []
    for i in range(mask.shape[0]):
        area = np.sum(mask[i] > 0)
        areas.append((i, area))
    
    # Sort by area in descending order
    areas.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in areas[:n]]

def plot_frangi_results(original_img, frangi_img, mask, slices_indices, image_name, output_dir):
    """Plot original and filtered slices side by side."""
    fig, axes = plt.subplots(len(slices_indices), 2, figsize=(12, 6*len(slices_indices)))
    
    for i, slice_idx in enumerate(slices_indices):
        # Original image with mask overlay
        axes[i, 0].imshow(original_img[slice_idx], cmap='gray')
        mask_overlay = np.ma.masked_where(mask[slice_idx] == 0, mask[slice_idx])
        axes[i, 0].imshow(mask_overlay, cmap='autumn', alpha=0.5)
        axes[i, 0].set_title(f"Original Image - Slice {slice_idx}")
        axes[i, 0].axis('off')
        
        # Frangi filter result
        axes[i, 1].imshow(frangi_img[slice_idx], cmap='inferno')
        axes[i, 1].set_title(f"Frangi Filter Result - Slice {slice_idx}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{image_name}_frangi_results.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file

def process_images(image_path_all, mask_path_all, output_dir="frangi_results"):
    """Process all images: crop, apply filter, and plot results."""
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, (img_path, mask_path) in enumerate(zip(image_path_all, mask_path_all)):
        image_name = os.path.basename(img_path).split('.')[0]
        print(f"Processing {image_name} ({i+1}/{len(image_path_all)})")
        
        try:
            # Load image and mask
            img, mask = load_image_and_mask(img_path, mask_path)
            
            # Crop image by mask
            cropped_result = crop_image_by_mask(img, mask)
            if cropped_result is None:
                print(f"Empty mask for {image_name}, skipping...")
                continue
                
            cropped_img, cropped_mask = cropped_result
            
            # Apply Frangi filter
            frangi_result = apply_frangi_filter(cropped_img)
            
            # Find slices with largest mask areas
            largest_slices = find_largest_mask_slices(cropped_mask)
            
            # Plot and save results
            output_file = plot_frangi_results(
                cropped_img, frangi_result, cropped_mask, 
                largest_slices, image_name, output_dir
            )
            
            results.append({
                'image_name': image_name,
                'output_file': output_file,
                'top_slices': largest_slices
            })
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
    
    return results

if __name__ == "__main__":
    # Example usage:
    # You need to define image_path_all and mask_path_all lists with paths to your images and masks
    
    # Example (replace with your actual paths):
    image_path_all = []
    mask_path_all = []
    
    # Directory containing images and masks
    data_dir = "your_data_directory_here"
    
    # Load image and mask paths - update according to your folder structure
    # Example:
    """
    for patient_id in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient_id)
        if os.path.isdir(patient_dir):
            img_file = os.path.join(patient_dir, f"{patient_id}_image.nii.gz")
            mask_file = os.path.join(patient_dir, f"{patient_id}_mask.nii.gz")
            if os.path.exists(img_file) and os.path.exists(mask_file):
                image_path_all.append(img_file)
                mask_path_all.append(mask_file)
    """
    
    # Process all images
    if image_path_all and mask_path_all:
        results = process_images(image_path_all, mask_path_all)
        print(f"Processed {len(results)} images successfully")
    else:
        print("No image paths provided. Please update the script with your image and mask paths.")
