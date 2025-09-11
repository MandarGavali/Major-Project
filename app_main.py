import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

class OptimizedDeepfakeModel(nn.Module):
    """Same model architecture as in training script"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(OptimizedDeepfakeModel, self).__init__()
        
        # Use same architecture as training
        try:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=False)  # Don't need pretrained for inference
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            # Fallback to MobileNetV2
            self.backbone = models.mobilenet_v2(pretrained=False)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

class CustomDeepfakeDetector:
    def __init__(self, model_path="./my_trained_model/best_deepfake_model.pth"):
        """Initialize with your trained model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self):
        """Load your trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        
        # Create model
        model = OptimizedDeepfakeModel(num_classes=2)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ Loaded trained model with {checkpoint['best_val_acc']:.2f}% validation accuracy")
        return model
    
    def predict(self, image):
        """Predict using YOUR trained model"""
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction from YOUR model
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            fake_prob = probabilities[0][0].cpu().item()
            real_prob = probabilities[0][1].cpu().item()
        
        return {
            "fake": round(fake_prob, 3),
            "real": round(real_prob, 3)
        }

# Try to load custom model, fall back to baseline if not available
try:
    detector = CustomDeepfakeDetector()
    model_type = "Your Custom Trained Model"
    model_status = "🎯 Using your custom trained model!"
except FileNotFoundError:
    print("⚠️  Custom trained model not found. Train it first using:")
    print("   python train_lightweight.py")
    print("⚠️  Using baseline model for now...")
    
    # Fallback to baseline
    class BaselineDetector:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def predict(self, image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                return {
                    "fake": round(probabilities[0][0].cpu().item(), 3),
                    "real": round(probabilities[0][1].cpu().item(), 3)
                }
    
    detector = BaselineDetector()
    model_type = "Baseline Model (Not Trained on Your Data)"
    model_status = "⚠️  Using baseline model. Train custom model for better results!"

def classify_image(image):
    """Main classification function"""
    try:
        prediction = detector.predict(image)
        
        # Enhanced feedback
        confidence = max(prediction.values())
        predicted_class = max(prediction, key=prediction.get)
        
        if confidence > 0.8:
            confidence_level = "High"
        elif confidence > 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        return prediction
        
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

# Create enhanced Gradio interface with dark theme
with gr.Blocks(theme=gr.themes.Monochrome(), title="Deepfake Detector") as iface:
    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #fff; font-size: 2.5em; margin-bottom: 10px;">🔍 Deepfake Detector</h1>
        <p style="color: #ccc; font-size: 1.1em;">AI-powered detection of fake and manipulated images</p>
    </div>
    """)
    
    # Model status indicator
    if "Custom Trained Model" in model_type:
        status_color = "#4CAF50"  # Green
        status_icon = "✅"
    else:
        status_color = "#FF9800"  # Orange
        status_icon = "⚠️"
    
    gr.HTML(f"""
    <div style="text-align: center; margin: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px; border-left: 4px solid {status_color};">
        <p style="color: {status_color}; font-size: 1.1em; margin: 0;">{status_icon} <strong>{model_type}</strong></p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image upload
            image_input = gr.Image(
                type="numpy",
                label="Upload Image",
                height=400,
                container=True
            )
            
            # Analyze button
            analyze_btn = gr.Button(
                "🔍 Analyze Image",
                variant="primary",
                size="lg",
                scale=1
            )
        
        with gr.Column(scale=1):
            # Results
            result_output = gr.Label(
                label="Detection Results",
                num_top_classes=2,
                container=True
            )
            
            # Result explanation
            gr.HTML("""
            <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px;">
                <h3 style="color: #fff; margin-top: 0;">How to Read Results:</h3>
                <ul style="color: #ccc; line-height: 1.6;">
                    <li><strong style="color: #4CAF50;">Real:</strong> Authentic photograph</li>
                    <li><strong style="color: #F44336;">Fake:</strong> AI-generated or manipulated</li>
                    <li><strong>Confidence:</strong> Higher values = more certain</li>
                </ul>
            </div>
            """)
    
    # Footer with tips
    gr.HTML("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.03); border-radius: 10px;">
        <h3 style="color: #fff; margin-bottom: 15px;">💡 Tips for Best Results</h3>
        <div style="color: #ccc; line-height: 1.8;">
            • Upload clear, high-quality images<br>
            • Works best with portraits and faces<br>
            • Larger images generally give better accuracy
        </div>
    </div>
    """)
    
    # Connect the button to the function
    analyze_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=result_output
    )
    
    # Also allow direct click on image
    image_input.change(
        fn=classify_image,
        inputs=image_input,
        outputs=result_output
    )

if __name__ == "__main__":
    print("🚀 Starting Your Custom Deepfake Detector...")
    print(f"🤖 Model Status: {model_status}")
    print("📱 Opening in browser...")
    print("⏹️  Press Ctrl+C to stop")
    
    iface.launch(inbrowser=True, share=False)
