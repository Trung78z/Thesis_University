import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Dataset root containing train/, valid/, test/
dataset_root = "/path/to/your/dataset"
subsets = ["train", "valid", "test"]  # Remove "test" if not available
image_exts = ('.jpg', '.jpeg', '.png')

counter = 1  # Global counter for sequential naming

logging.info(f"Starting dataset renaming process in: {dataset_root}")

for subset in subsets:
    image_dir = os.path.join(dataset_root, subset, "images")
    label_dir = os.path.join(dataset_root, subset, "labels")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        logging.warning(f"Skipping '{subset}' due to missing images or labels directory")
        continue

    logging.info(f"Processing {subset} set...")
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(image_exts)])
    logging.info(f"Found {len(image_files)} images in {subset} set")

    for image_file in image_files:
        base_name, ext = os.path.splitext(image_file)
        txt_file = base_name + ".txt"

        # Generate new sequential names
        new_base = f"thesis-{counter:06d}"
        new_image_name = new_base + ext
        new_txt_name = new_base + ".txt"

        # Old paths
        old_image_path = os.path.join(image_dir, image_file)
        old_txt_path = os.path.join(label_dir, txt_file)

        # New paths
        new_image_path = os.path.join(image_dir, new_image_name)
        new_txt_path = os.path.join(label_dir, new_txt_name)

        try:
            # Rename image file
            os.rename(old_image_path, new_image_path)
            logging.debug(f"Renamed image: {image_file} -> {new_image_name}")

            # Rename corresponding label file if it exists
            if os.path.exists(old_txt_path):
                os.rename(old_txt_path, new_txt_path)
                logging.debug(f"Renamed label: {txt_file} -> {new_txt_name}")
            else:
                logging.warning(f"No label file found for: {image_file}")

            counter += 1

        except Exception as e:
            logging.error(f"Error renaming {image_file}: {str(e)}")

    logging.info(f"Completed processing {subset} set")

logging.info(f"Dataset renaming completed. Total files processed: {counter-1}")
