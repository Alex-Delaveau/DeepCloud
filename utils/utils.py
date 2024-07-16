from PIL import Image
import numpy as np
import tensorflow as tf

def save_image_mask(image, mask, image_path, mask_path):
    # Extract only the RGB channels for the image
    image_rgb = image[:, :, :3]  # Keep only the first three channels (R, G, B)

    # Convert the RGB image to PIL Image format
    image_pil = Image.fromarray((image_rgb * 255).astype(np.uint8))

    # Process the predicted mask
    # If the mask is probability values (between 0 and 1)
    if mask.max() <= 1.0:
        # Threshold the mask to create a binary mask
        binary_mask = (mask > 0.5).astype(np.uint8)
    else:
        # If the mask is already in 0-255 range, just ensure it's the right data type
        binary_mask = mask.astype(np.uint8)

    # Ensure the mask is 2D (Height, Width) by squeezing out any extra dimensions
    binary_mask = np.squeeze(binary_mask)

    # Convert the binary mask to a PIL Image
    mask_pil = Image.fromarray(binary_mask * 255)  # Multiply by 255 to get 0 and 255 values

    # Save the RGB image and mask as TIFF
    save_tiff(image_pil, image_path)
    save_tiff(mask_pil, mask_path)


def save_tiff(image, path):
    image.save(path, "TIFF", compression="tiff_deflate", save_all=True, photometric='minisblack')

def extract_filename(path: str) -> str:
        """Extract filename from path, removing the color prefix."""
        # Split the path and get the filename
        full_filename = path.split('/')[-1]
        
        # Remove the file extension
        filename_without_extension = full_filename.rsplit('.', 1)[0]
        
        # Split the filename by underscore
        parts = filename_without_extension.split('_')
        
        # Remove the color prefix (first part) and join the rest
        return '_'.join(parts[1:])


def extract_scene_ids(filename: str) -> str: 
    """Extract scene id from filename, removing the row and colums."""    
    # Split the filename by underscore
    parts = filename.split('_')
    
    # Remove the color prefix (first part) and join the rest
    return '_'.join(parts[5:])


def load_model(path):
    print(f"Loading model from {path}")
    model = tf.keras.models.load_model(path)
    return model