"""
Test script for Hybrid CNN + Swin Transformer model
Tests the model on sample images from the dataset
"""

import os
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
IMG_SIZE = 224
CLASS_NAMES = ['Crown and Root Rot', 'Healthy Wheat', 'Leaf Rust', 'Wheat Loose Smut']
MODEL_PATH = 'models/hybrid_cnn_swin_wheat_best.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Model Definition
class HybridCNNSwinTransformer(nn.Module):
    """
    Hybrid CNN + Swin Transformer model for image classification.
    """
    
    def __init__(self, num_classes=4, pretrained=False):
        super(HybridCNNSwinTransformer, self).__init__()
        
        # CNN Backbone (ResNet50)
        resnet = timm.create_model('resnet50', pretrained=pretrained)
        self.cnn_features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.act1,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        self.cnn_out_channels = 512
        
        # Swin Transformer
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0
        )
        self.swin_out_features = self.swin.num_features
        
        # CNN Global Average Pooling
        self.cnn_gap = nn.AdaptiveAvgPool2d(1)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_out_channels + self.swin_out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        cnn_features = self.cnn_features(x)
        cnn_pooled = self.cnn_gap(cnn_features)
        cnn_pooled = cnn_pooled.view(cnn_pooled.size(0), -1)
        swin_features = self.swin(x)
        combined = torch.cat([cnn_pooled, swin_features], dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """Load the trained model."""
    print(f"\nLoading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please train the model first using the Jupyter notebook.")
        return None
    
    model = HybridCNNSwinTransformer(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model

def predict_image(model, image_path):
    """Predict disease for a single image."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return CLASS_NAMES[predicted.item()], confidence.item() * 100, probabilities[0].cpu().numpy()

def test_on_samples():
    """Test model on sample images from the dataset."""
    model = load_model()
    if model is None:
        return
    
    # Find sample images from Images folder
    images_dir = Path('Images')
    if not images_dir.exists():
        print(f"ERROR: Images directory not found at {images_dir}")
        return
    
    print("\n" + "="*60)
    print("Testing on sample images from each class")
    print("="*60)
    
    results = []
    
    for class_name in CLASS_NAMES:
        class_dir = images_dir / class_name
        if not class_dir.exists():
            print(f"WARNING: {class_dir} not found")
            continue
        
        # Get first image from this class
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jfif')) + list(class_dir.glob('*.png'))
        
        if not images:
            print(f"WARNING: No images found in {class_dir}")
            continue
        
        # Test on first 2 images from each class
        for img_path in images[:2]:
            predicted_class, confidence, probs = predict_image(model, str(img_path))
            
            is_correct = predicted_class == class_name
            status = "✓" if is_correct else "✗"
            
            results.append({
                'true': class_name,
                'predicted': predicted_class,
                'confidence': confidence,
                'correct': is_correct
            })
            
            print(f"\n{status} Image: {img_path.name}")
            print(f"  True Class: {class_name}")
            print(f"  Predicted:  {predicted_class} ({confidence:.2f}%)")
    
    # Summary
    if results:
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total * 100
        
        print("\n" + "="*60)
        print(f"SUMMARY: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
        print("="*60)

def test_single_image(image_path):
    """Test model on a single image."""
    model = load_model()
    if model is None:
        return
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    print(f"\nPredicting: {image_path}")
    predicted_class, confidence, probs = predict_image(model, image_path)
    
    print(f"\nResult: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nAll class probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        bar = "█" * int(probs[i] * 30)
        print(f"  {class_name:25} {probs[i]*100:6.2f}% {bar}")
    
    # Display image
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('test_prediction.png')
    plt.show()

def interactive_test():
    """Interactive testing mode."""
    model = load_model()
    if model is None:
        return
    
    print("\n" + "="*60)
    print("Interactive Testing Mode")
    print("Enter image path to test, or 'quit' to exit")
    print("="*60)
    
    while True:
        path = input("\nImage path: ").strip()
        
        if path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        predicted_class, confidence, probs = predict_image(model, path)
        
        print(f"\n  Result: {predicted_class}")
        print(f"  Confidence: {confidence:.2f}%")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Test specific image
        test_single_image(sys.argv[1])
    else:
        # Run sample tests
        test_on_samples()
        
        # Option for interactive mode
        print("\n")
        choice = input("Enter 'i' for interactive mode, or press Enter to exit: ").strip()
        if choice.lower() == 'i':
            interactive_test()
