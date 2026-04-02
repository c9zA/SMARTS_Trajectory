import os
import random
import shutil

RAW_ROOT = r"F:\surgsync_data\offline_recorder\raw"
DEST_ROOT = r"D:\University\SMARTS\Spring26\dissection_specific_scripts\left_image_selected"
print(os.path.exists(RAW_ROOT))
print(os.listdir(RAW_ROOT))
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
MAX_NUM = 200

os.makedirs(DEST_ROOT, exist_ok=True)


def is_valid_image(filename):
    """
    Check:
    1. is image file
    2. filename (number part) <= 200
    """
    if not filename.lower().endswith(IMAGE_EXTS):
        return False

    name, _ = os.path.splitext(filename)

    try:
        return int(name) <= MAX_NUM
    except ValueError:
        return False


# iterate all folders inside RAW_ROOT
for folder_name in os.listdir(RAW_ROOT):

    folder_path = os.path.join(RAW_ROOT, folder_name)
    if not os.path.isdir(folder_path):
        continue

    left_dir = os.path.join(folder_path, "image", "left")
    if not os.path.exists(left_dir):
        print(f"skip (no left dir): {folder_name}")
        continue

    # filter valid images
    valid_images = [
        f for f in os.listdir(left_dir)
        if is_valid_image(f)
    ]

    if not valid_images:
        print(f"skip (no images <=200): {folder_name}")
        continue

    # randomly choose up to 3
    selected = random.sample(valid_images, min(3, len(valid_images)))

    # create destination subfolder
    dest_sub = os.path.join(DEST_ROOT, folder_name)
    os.makedirs(dest_sub, exist_ok=True)

    # copy files
    for img in selected:
        src = os.path.join(left_dir, img)
        dst = os.path.join(dest_sub, img)
        shutil.copy2(src, dst)

    print(f"{folder_name}: copied {len(selected)} images")

print("DONE")