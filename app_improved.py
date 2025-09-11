import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

class ImprovedDeepfakeDetector:
    def __init__(self):
        """Initialize with a better architecture for deepfake detection"""
        # Use ResNet50 as backbone - better for this task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _create_model(self):
        """Create improved model architecture"""
        # Use ResNet50 pretrained on ImageNet
        model = models.resnet50(pretrained=True)
        
        # Modify the final layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # fake vs real
        )
        
        return model.to(self.device)
    
    def predict(self, image):
        """Predict if image is real or fake"""
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            fake_prob = probabilities[0][0].cpu().item()
            real_prob = probabilities[0][1].cpu().item()
        
        return {
            "fake": round(fake_prob, 3),
            "real": round(real_prob, 3)
        }

# Initialize detector
detector = ImprovedDeepfakeDetector()

def classify_image(image):
    """Main classification function for Gradio"""
    try:
        prediction = detector.predict(image)
        
        # Add more detailed analysis
        confidence = max(prediction.values())
        predicted_class = max(prediction, key=prediction.get)
        
        # Enhanced feedback
        if confidence > 0.8:
            confidence_text = "High"
        elif confidence > 0.6:
            confidence_text = "Medium"
        else:
            confidence_text = "Low"
        
        return prediction
        
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

# Create Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Deepfake Detection Results"),
    title="🔍 Enhanced Deepfake Detector v1.1",
    description="""
    **Upload an image to detect if it's real or AI-generated.**
    
    - **Real**: Authentic photograph
    - **Fake**: AI-generated or manipulated image
    
    *Note: This is using a baseline model. For best results, train on your specific dataset.*
    """,
    examples=[
        # You can add example images here if you have some test images
    ],
    theme="soft"
)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Deepfake Detector...")
    print("📝 Note: This uses a baseline model. For better accuracy, consider training on your dataset.")
    iface.launch(inbrowser=True)
