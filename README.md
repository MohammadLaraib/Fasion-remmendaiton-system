# Fasion-remmendaiton-system
Here’s a concise README file you can use for your fashion recommendation system:

---

# Fashion Recommendation System

This repository contains a fashion recommendation system that uses machine learning techniques to recommend similar fashion images based on a user's input image. The system extracts features from images using a pre-trained model (ResNet50 or Vision Transformer (ViT)) and compares them using cosine similarity to provide recommendations.

## Features
- Downloads a sample fashion dataset from an online source.
- Uses a pre-trained model (ResNet50 or Vision Transformer (ViT)) for feature extraction.
- Finds the top-k most similar images from the dataset based on the input image.
- Displays the input image alongside the top recommended images.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- PIL (Pillow)
- requests

To install the required dependencies, run:

```bash
pip install torch torchvision matplotlib scikit-learn pillow requests
```

## How to Use

### Step 1: Download the Fashion Dataset
The fashion dataset is automatically downloaded when you run the script. It contains fashion images that the system will use to compare with the user input.

### Step 2: Provide Your Image
- Upload an image of fashion (clothing) to be used as a query image for recommendations.
- Change the `user_img_path` in the script to the path of your image.

```python
user_img_path = "/path/to/your/image.jpg"
```

### Step 3: Run the Script
Execute the script. The system will:
1. Download the fashion dataset if it doesn't exist.
2. Extract features from the dataset and the user's input image.
3. Find the top-3 most similar fashion images based on the extracted features.
4. Display the input image along with the recommended images.

Run the script using:

```bash
python fashion_recommendation_system.py
```

### Output
- The system will display the query image and the top-k recommendations side by side.

### Example Output:
```plaintext
Loading dataset and extracting features...
Finding similar fashion recommendations...
```

The recommended images will appear in a window, with the input image on the left and the recommended images on the right.

## Customization

- **Changing the number of recommendations**: You can change the number of recommendations by modifying the `top_k` parameter in the `recommend_images` function:
  
```python
recommendations = recommend_images(user_feature, features, image_paths, top_k=5)
```

- **Model selection**: You can choose between ResNet50 and ViT (Vision Transformer) for feature extraction by adjusting the model loading line:

```python
# For ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# For Vision Transformer (ViT)
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
```

## Troubleshooting

- If the dataset fails to download, check your internet connection or verify the dataset URL.
- If the recommendations are not relevant, ensure the dataset contains sufficient variety. Consider uploading a more diverse set of fashion images.

## License

This project is licensed under the MIT License.

## Acknowledgements
- Pretrained models were sourced from the `torchvision` library.
- The dataset is provided by the HuggingFace repository.

---

Feel free to adjust the README based on any further changes you make to the code or project structure!
