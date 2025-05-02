# this file is also not meant to be run since it was already used to remove low quality images.
# running file file again will do nothing or give error or mess up the dataset

import os
import json
from PIL import Image

# Paths
image_folder = "/home/public/htnguyen/project/book_cover_synthesis/dataset_for_huggingface_filter/test_set"
jsonl_file = "/home/public/htnguyen/project/book_cover_synthesis/dataset_for_huggingface_filter/train_set/metadata.jsonl"

# Load JSONL entries
with open(jsonl_file, "r") as f:
    json_lines = [json.loads(line) for line in f]

# Store valid JSON entries and images to delete
filtered_json_lines = []
images_to_delete = []

# Process images
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        image_path = os.path.join(image_folder, filename)
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size  # Get image dimensions
            
            # If dimensions are too small, mark image for deletion
            if width < 305 or height < 305 or width >= height:
                images_to_delete.append(image_path)
            else:
                # Keep valid JSON entries
                entry = next((entry for entry in json_lines if entry.get("file_name") == filename), None)
                if entry:
                    filtered_json_lines.append(entry)
        except Exception as e:
            print(f"Skipping corrupted file: {filename}, Error: {e}")

# Ensure JSON is not empty before writing
if filtered_json_lines:
    with open(jsonl_file, "w") as f:
        for entry in filtered_json_lines:
            f.write(json.dumps(entry) + "\n")
else:
    print("Warning: No valid JSON entries left after filtering!")

# Delete images **after** updating JSON
for image_path in images_to_delete:
    os.remove(image_path)

print(f"Cleanup complete! 🚀 Removed {len(images_to_delete)} images.")
