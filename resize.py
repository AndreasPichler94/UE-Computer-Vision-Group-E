from PIL import Image
import os

def resize_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                resized_img = img.resize((512, 512))
                resized_img.save(file_path)

if __name__ == "__main__":
    folder_path = './data/train_smaller'  # Replace with your folder path
    resize_images_in_folder(folder_path)
    print("Resizing completed.")