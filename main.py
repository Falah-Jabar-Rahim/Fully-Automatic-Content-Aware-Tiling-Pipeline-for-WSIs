import os
import shutil

def copy_images_to_single_folder(text_file, target_folder):
    # Create the target directory if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)
    
    # Read the file paths from the text file
    with open(text_file, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        # Remove any extra whitespace or newlines
        image_path = line.strip()
        
        # Check if the path exists and is a file
        if os.path.isfile(image_path):
            # Copy the file to the target directory
            shutil.copy(image_path, target_folder)
        else:
            print(f"Warning: {image_path} is not a valid file path.")
    
    print(f"All valid images have been copied to {target_folder}.")

# Example usage
text_file = 'test_images.txt'  # Path to your text file containing image paths
target_folder = 'all_test_images'  # Path to the target directory

copy_images_to_single_folder(text_file, target_folder)

