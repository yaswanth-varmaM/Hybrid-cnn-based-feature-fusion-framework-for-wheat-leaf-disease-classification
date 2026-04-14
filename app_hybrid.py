import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename

# PyTorch imports for Hybrid CNN + Swin Transformer
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

# Reportlab imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

# 1. Initialize Flask app
app = Flask(__name__, static_folder='static')

# 2. Define global paths and configurations
BASE_PATH = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_PATH, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = os.path.join(BASE_PATH, 'static')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
MODEL_NAME = 'Hybrid CNN + Swin Transformer'
CLASS_NAMES = ['Crown and Root Rot', 'Healthy Wheat', 'Leaf Rust', 'Wheat Loose Smut']
IMG_SIZE = 224

# PyTorch device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hybrid CNN + Swin Transformer Model Definition
class HybridCNNSwinTransformer(nn.Module):
    """
    Hybrid CNN + Swin Transformer model for image classification.
    Combines ResNet50 CNN features with Swin Transformer global context.
    """
    
    def __init__(self, num_classes=4, pretrained=False):
        super(HybridCNNSwinTransformer, self).__init__()
        
        # CNN Backbone (ResNet50 - first 3 stages)
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
        # CNN branch
        cnn_features = self.cnn_features(x)
        cnn_pooled = self.cnn_gap(cnn_features)
        cnn_pooled = cnn_pooled.view(cnn_pooled.size(0), -1)
        
        # Swin Transformer branch
        swin_features = self.swin(x)
        
        # Fusion
        combined = torch.cat([cnn_pooled, swin_features], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        output = self.classifier(fused)
        
        return output

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
MODEL_PATH = os.path.join(BASE_PATH, 'models', 'hybrid_cnn_swin_wheat_best.pth')
model = None

def load_model():
    global model
    print(f"Loading Hybrid CNN + Swin Transformer model from {MODEL_PATH}")
    try:
        model = HybridCNNSwinTransformer(num_classes=4, pretrained=False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Predict disease for a single image."""
    if model is None:
        return None, 0.0
    
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0

# Flask Routes
@app.route('/')
def index():
    model_loaded = model is not None
    return render_template('index_simple.html',
                           model_name=MODEL_NAME,
                           model_loaded=model_loaded,
                           class_names=CLASS_NAMES)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index', error='No file selected'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', error='No file selected'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if model is None:
            return redirect(url_for('index', error='Model not loaded'))
        
        # Predict
        predicted_class, confidence = predict_image(filepath)
        
        if predicted_class is None:
            return redirect(url_for('index', error='Prediction failed'))
        
        return render_template('predict_simple.html',
                               image_path=url_for('uploaded_file', filename=filename),
                               predicted_class=predicted_class,
                               confidence=f"{confidence:.2f}%",
                               model_name=MODEL_NAME,
                               filename=filename)
    else:
        return redirect(url_for('index', error='Invalid file type. Use JPG, JPEG, or PNG.'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# PDF Report Generation
def generate_report(image_filename, predicted_class, confidence):
    doc_filename = "prediction_report.pdf"
    doc = SimpleDocTemplate(doc_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    # Title
    title_style = styles['h1']
    title_style.alignment = TA_CENTER
    flowables.append(Paragraph("Wheat Disease Detection Report", title_style))
    flowables.append(Spacer(1, 0.2 * inch))

    # Model info
    flowables.append(Paragraph(f"<b>Model:</b> {MODEL_NAME}", styles['Normal']))
    flowables.append(Spacer(1, 0.1 * inch))

    # Prediction Details
    flowables.append(Paragraph(f"<b>Predicted Disease:</b> {predicted_class}", styles['Normal']))
    flowables.append(Paragraph(f"<b>Confidence:</b> {confidence}", styles['Normal']))
    flowables.append(Spacer(1, 0.2 * inch))

    # Image
    flowables.append(Paragraph("<b>Analyzed Image:</b>", styles['Normal']))
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    if os.path.exists(image_path):
        img = RLImage(image_path, width=4*inch, height=4*inch)
        flowables.append(img)
    flowables.append(Spacer(1, 0.3 * inch))

    # Classes
    flowables.append(Paragraph("<b>Detectable Diseases:</b>", styles['Normal']))
    for name in CLASS_NAMES:
        flowables.append(Paragraph(f"• {name}", styles['Normal']))
    flowables.append(Spacer(1, 0.2 * inch))

    # Footer
    footer_style = styles['Normal']
    footer_style.alignment = TA_CENTER
    flowables.append(Paragraph("<i>Generated by Wheat Disease Detection System</i>", footer_style))

    doc.build(flowables)
    return doc_filename

@app.route('/generate_report')
def generate_report_route():
    filename = request.args.get('filename')
    predicted_class = request.args.get('predicted_class')
    confidence = request.args.get('confidence')

    if not all([filename, predicted_class, confidence]):
        return redirect(url_for('index', error='Missing report data'))

    pdf_path = generate_report(filename, predicted_class, confidence)
    return send_file(pdf_path, as_attachment=True, download_name='wheat_disease_report.pdf')

# Load model on startup
load_model()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
