import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def check_images_vs_npz(folder_path, npz_subfolder):
    ground_truth = {}

    # List all jpg filenames in the main folder
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    jpg_names = {os.path.splitext(f)[0] for f in jpg_files}

    # Get all .npz file names (without extension) in the npz subfolder
    npz_path = os.path.join(folder_path, npz_subfolder)
    npz_files = [f for f in os.listdir(npz_path) if f.lower().endswith('.npz')]
    npz_names = {os.path.splitext(f)[0] for f in npz_files}

    for name in jpg_names:
        ground_truth[name] = name in npz_names

    return ground_truth

if __name__ == "__main__":
    folder = "seq/meister_bertram_mit_eindeutigen_potholes"
    ground_truth = check_images_vs_npz(folder, "masks_ground_truth")

    logging.info("ground_truth = {")
    for key, value in ground_truth.items():
        logging.info(f'    "{key}": {value},')
    logging.info("}")
