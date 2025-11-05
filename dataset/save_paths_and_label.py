import os
import re

root_dir = r"Y:\cpi_tracking\simulated_data\nn_data_5k"
output_file = r"Y:\cpi_tracking\simulated_data\nn_data_5k\tiff_paths.txt"

# Pattern to extract folder number: matches "10mm" or "-10mm"
folder_pattern = re.compile(r'([\-]?\d+(?:\.\d+)?)mm')
with open(output_file, "w") as f_out:
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        # Check if folder matches the pattern and exists
        match = folder_pattern.fullmatch(folder)
        if match and os.path.isdir(folder_path):
            number = match.group(1)
            spatial_dir = os.path.join(folder_path, "data", "spatial")
            if os.path.isdir(spatial_dir):
                for filename in os.listdir(spatial_dir):
                    if filename.lower().endswith(".tiff"):
                        full_path = os.path.join(spatial_dir, filename)
                        f_out.write(f"{full_path} {number}\n")