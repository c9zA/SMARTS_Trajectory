import os
import shutil

def mirror_folder_names(dir_B, dir_A):
    # Make sure directory A exists
    os.makedirs(dir_A, exist_ok=True)

    # Loop through items in directory B
    for name in os.listdir(dir_B):
        full_path = os.path.join(dir_B, name)

        # Only mirror folders, ignore files
        if os.path.isdir(full_path):
            new_folder_path = os.path.join(dir_A, name)
            os.makedirs(new_folder_path, exist_ok=True)
            print(f"Created: {new_folder_path}")

# Example usage:
dir_B = r"D:/University/SMARTS/Spring26/dissection_specific_scripts/left_image_selected"
dir_A = r"D:/University/SMARTS/Spring26/dissection_specific_scripts/left_image_output"

if __name__ == "__main__":
    mirror_folder_names(dir_B, dir_A)