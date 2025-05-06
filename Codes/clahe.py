# reference Shao, J., Zhou, K., Cai, Y. H., & Geng, D. Y. (2022). Application of an improved u2-net model in ultrasound median neural image segmentation. Ultrasound in Medicine & Biology, 48(12), 2512-2520.
import cv2
import numpy as np
import os
import glob
import shutil
from tqdm import tqdm  # for progress bar

def process_image(image_path):
    """
    Process a single image by applying CLAHE and a 3×3 smoothing filter.

    Parameters:
        image_path (str): Full path to the input TIFF image.

    Returns:
        tuple: (clahe_img, filtered_img), or (None, None) if image loading fails.
    """
    # Read image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error reading image: {image_path}")
        return None, None

    # === 1. Apply CLAHE ===
    # Create a CLAHE object with default parameters (tweak clipLimit and tileGridSize as needed)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    
    # === 2. Apply the 3x3 2-D filter (smoothing filter) ===
    # Using a uniform average kernel. Modify the kernel if needed for exact replication.
    kernel = np.ones((3, 3), np.float32) / 9.0
    filtered_img = cv2.filter2D(clahe_img, ddepth=-1, kernel=kernel)

    return clahe_img, filtered_img

def process_folder(input_folder, output_folder_clahe, output_folder_filter):
    """
    Process all TIFF images in the input folder:
      - If the file is in the train folder and contains "mask" in its name,
        copy the file directly to the output folders (preserving it).
      - Otherwise, apply CLAHE and a 3×3 smoothing filter,
        saving the results in the specified output folders.

    Parameters:
        input_folder (str): Directory containing input TIFF images.
        output_folder_clahe (str): Directory to save CLAHE-processed images.
        output_folder_filter (str): Directory to save 2-D filtered images.
    """
    os.makedirs(output_folder_clahe, exist_ok=True)
    os.makedirs(output_folder_filter, exist_ok=True)
    
    # Gather all TIFF files (files ending with .tif or .tiff)
    tif_files = glob.glob(os.path.join(input_folder, '*.tif')) + glob.glob(os.path.join(input_folder, '*.tiff'))
    
    for img_path in tqdm(tif_files, desc=f"Processing images in {os.path.basename(input_folder)}", total=len(tif_files)):
        basename = os.path.basename(img_path)
        
        # Check if in the train folder and file name contains "mask"
        if "mask" in basename.lower() and "train" in os.path.abspath(input_folder):
            # Simply copy the file to both output folders without processing
            shutil.copy(img_path, os.path.join(output_folder_clahe, basename))
            shutil.copy(img_path, os.path.join(output_folder_filter, basename))
            continue
        
        # Otherwise process the image
        clahe_img, filter_img = process_image(img_path)
        if clahe_img is not None and filter_img is not None:
            # Save the CLAHE-processed image
            cv2.imwrite(os.path.join(output_folder_clahe, basename), clahe_img)
            # Save the filtered image
            cv2.imwrite(os.path.join(output_folder_filter, basename), filter_img)

if __name__ == "__main__":
    # Define input directories for 'train' and 'test'
    base_input = "/home/user1/charlie/code/deep_learning/project/ultrasound-nerve-segmentation"
    input_dirs = {
        "train": os.path.join(base_input, "train"),
        "test": os.path.join(base_input, "test")
    }

    # Define output base paths for CLAHE and 2-D filter results
    clahe_base = "/home/user1/charlie/code/deep_learning/project/ultrasound-nerve-segmentation_clahe"
    filter_base = "/home/user1/charlie/code/deep_learning/project/ultrasound-nerve-segmentation_2Dfilter"

    # Set output directories for each split
    output_dirs_clahe = {
        "train": os.path.join(clahe_base, "train"),
        "test": os.path.join(clahe_base, "test")
    }
    output_dirs_filter = {
        "train": os.path.join(filter_base, "train"),
        "test": os.path.join(filter_base, "test")
    }

    # Process images in both 'train' and 'test' directories
    for key in input_dirs:
        print(f"Processing {key} images from {input_dirs[key]}...")
        process_folder(input_dirs[key], output_dirs_clahe[key], output_dirs_filter[key])
