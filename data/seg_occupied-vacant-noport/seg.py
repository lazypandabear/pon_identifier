import os

# Set your paths
images_folder = r"C:\Users\Keno\source_repos\pon_map\pon_wd\seg_occupied-vacant-noport\train\images"
labels_folder = r"C:\Users\Keno\source_repos\pon_map\pon_wd\seg_occupied-vacant-noport\train\labels"


# Get lists of image and label file names (without extensions)
image_files = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith('.jpg')}
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith('.txt')}

# Find images without labels
unlabeled_images = image_files - label_files

# Remove excess images
for img in unlabeled_images:
    img_path = os.path.join(images_folder, img + ".jpg")
    os.remove(img_path)
    print(f"Deleted: {img_path}")

print("Cleanup complete!")
