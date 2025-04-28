# Fasion-remmendaiton-system

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os
import requests
from io import BytesIO

# Download and extract fashion dataset
def download_and_extract_dataset():
    dataset_url = "https://huggingface.co/datasets/0xJarvis/fashion-sample-dataset/resolve/main/fashion_data.zip"
    local_zip = "fashion_data.zip"
    if not os.path.exists("fashion_data"):
        print("Downloading sample fashion dataset...")
        response = requests.get(dataset_url)
        if response.status_code == 200:
            with open(local_zip, "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall("fashion_data")
            print("Dataset downloaded and extracted successfully.")
        else:
            print("Failed to download dataset. Please check the link.")
    else:
        print("Dataset already exists, skipping download.")

# Load ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Transformation for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Feature extraction from ResNet50
def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img)
    return features.squeeze().numpy()

# Load dataset
def load_dataset(folder_path):
    paths = []
    features = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file)
            paths.append(img_path)
            features.append(extract_features(img_path))
    return paths, features

# Recommend similar images
def recommend_images(query_feature, dataset_features, dataset_paths, top_k=3):
    sims = cosine_similarity([query_feature], dataset_features)[0]
    indices = sims.argsort()[-top_k:][::-1]
    return [dataset_paths[i] for i in indices]

# Show recommendations
def show_results(input_img_path, recommended_paths):
    imgs = [Image.open(input_img_path).convert('RGB')] + [Image.open(p).convert('RGB') for p in recommended_paths]
    titles = ["User Input"] + [f"Recommendation {i+1}" for i in range(len(recommended_paths))]

    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    download_and_extract_dataset()

    dataset_dir = "fashion_data"  # Dataset folder
    user_img_path = "/content/user 3.jpg"  # <-- Your uploaded user image path

    print("Loading dataset and extracting features...")
    image_paths, features = load_dataset(dataset_dir)

    if len(image_paths) < 4:
        print("Not enough images in dataset! Please upload more fashion images.")
    else:
        print("Finding similar fashion recommendations...")
        user_feature = extract_features(user_img_path)
        recommendations = recommend_images(user_feature, features, image_paths, top_k=3)
        show_results(user_img_path, recommendations)


