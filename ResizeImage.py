import os
from PIL import Image

def resize_first_10_images(folder_path, size=(32, 32)):
    # List all files in the folder
    files = sorted(os.listdir(folder_path))
    
    # Filter image files (you can add more extensions if needed)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Take the first 10 images
    for i, filename in enumerate(image_files[:10]):
        try:
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                resized_img = img.resize(size)
                
                # Create new filename
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_resized{ext}"
                resized_img.save(os.path.join(folder_path, f'{name}.{ext}'))
                
                print(f"[{i+1}/10] Resized and saved: {new_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
resize_first_10_images('dataset/0')
resize_first_10_images('dataset/1')
resize_first_10_images('dataset/2')
resize_first_10_images('dataset/3')
resize_first_10_images('dataset/4')
resize_first_10_images('dataset/5')

import os

def remove_resized_images(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Loop through and delete files that contain '_resized' in the name
    removed_count = 0
    for filename in files:
        if '_resized' and '..' in filename:
            try:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                removed_count += 1
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
    
    print(f"Total removed: {removed_count}")

# Example usage
# remove_resized_images("path/to/your/folder")
# remove_resized_images('dataset/0')
# remove_resized_images('dataset/1')
# remove_resized_images('dataset/2')
# remove_resized_images('dataset/3')
# remove_resized_images('dataset/4')
# remove_resized_images('dataset/5')
# Example usage
# resize_first_10_images("path/to/your/folder")
