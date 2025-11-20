import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
import os

class ViTFeatureExtractor:
    def __init__(self, model_name='google/vit-base-patch16-224-in21k'):
        """
        Initializes the ViT model and processor.
        We use the 'in21k' version which is pre-trained on ImageNet-21k
        but doesn't have a specific classification head fine-tuned yet,
        making it perfect for pure feature extraction.
        """
        print(f"Loading ViT model: {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set to evaluation mode (no training)
        print(f"Model loaded on {self.device}")

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses a single image.
        """
        image = Image.open(image_path).convert("RGB")
        # The processor handles resizing (to 224x224) and normalization
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.to(self.device)

    def extract_features(self, image_path):
        """
        Passes the image through the ViT and extracts the feature vector.
        """
        inputs = self.preprocess_image(image_path)

        with torch.no_grad(): # No gradients needed for extraction
            outputs = self.model(**inputs)

        # The ViT output has shape [batch_size, sequence_length, hidden_size]
        # sequence_length = 197 (196 patches + 1 CLS token)
        # hidden_size = 768 (for ViT-Base)
        last_hidden_states = outputs.last_hidden_state

        # We extract the [CLS] token (index 0), which contains the 
        # global image representation used for classification.
        cls_embedding = last_hidden_states[:, 0, :] 

        # Convert to numpy array and flatten to 1D vector
        return cls_embedding.cpu().numpy().flatten()

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Initialize the extractor
    extractor = ViTFeatureExtractor()
    
    # Create a dummy image for testing if you don't have one
    if not os.path.exists("test_face.jpg"):
        print("Creating dummy image for test...")
        dummy_img = Image.new('RGB', (100, 100), color = 'red')
        dummy_img.save("test_face.jpg")
    
    # Extract features
    image_path = "/Users/bluee/Downloads/Anhtest.JPEG"
    features = extractor.extract_features(image_path)
    
    print("\n--- Extraction Complete ---")
    print(f"Feature Vector Shape: {features.shape}")
    print(f"First 10 features: {features[:10]}")
    print("\nThis 768-dimensional vector is what you feed into your Logistic Regression!")