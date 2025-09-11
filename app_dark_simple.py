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
            self.backbone = efficientnet_b0(pretrained=False)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        except ImportError:
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
        
        model = OptimizedDeepfakeModel(num_classes=2)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded - {checkpoint['best_val_acc']:.1f}% accuracy")
        return model
    
    def predict(self, image):
        """Predict using YOUR trained model"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            fake_prob = probabilities[0][0].cpu().item()
            real_prob = probabilities[0][1].cpu().item()
        
        return {
            "Fake": round(fake_prob, 3),
            "Real": round(real_prob, 3)
        }

# Initialize detector
try:
    detector = CustomDeepfakeDetector()
    model_status = "🎯 Custom Model Active"
except FileNotFoundError:
    print("⚠️ Custom model not found. Using baseline...")
    
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
                    "Fake": round(probabilities[0][0].cpu().item(), 3),
                    "Real": round(probabilities[0][1].cpu().item(), 3)
                }
    
    detector = BaselineDetector()
    model_status = "⚠️ Baseline Model"

def analyze_image(image):
    """Main detection function"""
    if image is None:
        return {"Error": "No image uploaded"}
    
    try:
        result = detector.predict(image)
        return result
    except Exception as e:
        return {"Error": f"Analysis failed: {str(e)}"}

# Create simple dark theme interface
custom_css = """
/* Dark theme styling */
.gradio-container {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.block {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

/* Custom button styling */
.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

/* Header styling */
h1 {
    color: #ffffff !important;
    text-align: center !important;
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
}

/* Status indicator */
.status-indicator {
    text-align: center;
    padding: 12px;
    margin: 20px auto;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 25px;
    border-left: 4px solid #4CAF50;
    color: #4CAF50;
    font-weight: 600;
    max-width: 300px;
}

/* Results styling */
.label-container {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome(), title="Deepfake Detector") as app:
    
    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 30px 0;">
        <h1>🔍 Deepfake Detector</h1>
        <p style="color: #cccccc; font-size: 1.1rem; margin-top: 10px;">
            Detect AI-generated and manipulated images
        </p>
    </div>
    """)
    
    # Status
    gr.HTML(f"""
    <div class="status-indicator">
        {model_status}
    </div>
    """)
    
    # Main interface
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=400):
            image_input = gr.Image(
                label="📤 Upload Image", 
                type="numpy",
                height=400,
                container=True
            )
            
            analyze_button = gr.Button(
                "🔍 Analyze Image", 
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1, min_width=400):
            result_label = gr.Label(
                label="🎯 Detection Results",
                num_top_classes=2,
                container=True
            )
            
            gr.HTML("""
            <div style="margin-top: 25px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 12px; border-left: 4px solid #667eea;">
                <h3 style="color: #ffffff; margin-top: 0; font-size: 1.2rem;">📊 Understanding Results</h3>
                <div style="color: #cccccc; line-height: 1.8;">
                    <p><strong style="color: #4CAF50;">Real:</strong> Authentic photograph</p>
                    <p><strong style="color: #f44336;">Fake:</strong> AI-generated or manipulated</p>
                    <p><em>Higher confidence = more certain result</em></p>
                </div>
            </div>
            """)
    
    # Usage tips
    gr.HTML("""
    <div style="text-align: center; margin: 30px auto; padding: 25px; background: rgba(255,255,255,0.03); border-radius: 15px; max-width: 600px;">
        <h3 style="color: #ffffff; margin-bottom: 15px;">💡 Tips</h3>
        <p style="color: #cccccc; line-height: 1.6;">
            • Use clear, high-resolution images for best results<br>
            • The model works best with portraits and faces<br>
            • Drag & drop or click to upload your image
        </p>
    </div>
    """)
    
    # Event handlers
    analyze_button.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=result_label
    )
    
    # Auto-analyze on image upload
    image_input.change(
        fn=analyze_image,
        inputs=image_input,
        outputs=result_label
    )

if __name__ == "__main__":
    print("🚀 Starting Simple Dark Deepfake Detector...")
    print("📱 Interface will open in your browser...")
    app.launch(
        inbrowser=True, 
        share=False,
        server_name="127.0.0.1",
        server_port=7861
    )
