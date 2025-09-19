import os
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import vit_b_16
from PIL import Image
import numpy as np
import gdown

# --- Hybrid Model Definition ---
class HybridDeepfakeDetector(nn.Module):
    def __init__(self):
        super(HybridDeepfakeDetector, self).__init__()
        # CNN Backbone (ResNet50)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # remove final layer

        # Transformer Backbone (ViT)
        self.vit = vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads.head = nn.Identity()  # remove classifier head

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 512),  # ResNet (2048) + ViT (768)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # fake vs real
        )

    def forward(self, x):
        resnet_feat = self.resnet(x)     # [batch, 2048]
        vit_feat = self.vit(x)           # [batch, 768]
        combined = torch.cat((resnet_feat, vit_feat), dim=1)
        out = self.classifier(combined)
        return out

# --- Detector Wrapper ---
class ImprovedDeepfakeDetector:
    def __init__(self, device=None):
        """Initialize hybrid CNN+ViT deepfake detector"""
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridDeepfakeDetector().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_weights(self, path, gdrive_id=None):
        """Load trained .pth weights into the model.
        Downloads from Google Drive if file not present.
        """
        if not os.path.exists(path) and gdrive_id:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"📥 Downloading model from Google Drive...")
            gdown.download(f'https://drive.google.com/uc?id={gdrive_id}', path, quiet=False)

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"✅ Loaded weights from {path}")

    def predict(self, image):
        """Predict if an image is real or fake"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze()
            return {"fake": round(probs[0].item(), 3), "real": round(probs[1].item(), 3)}

# --- Initialize and load trained model ---
detector = ImprovedDeepfakeDetector()
model_path = "hybridModel/deepfake_hybrid.pth"  # local path to store/download model
gdrive_id = "1F1QOzlsHJKpef8rUZCTLOVDR0b7uJbzN"  # Google Drive file ID
detector.load_weights(model_path, gdrive_id=gdrive_id)

# --- Gradio Interface ---
def classify_image(image):
    try:
        prediction = detector.predict(image)
        return prediction
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Deepfake Detection Results"),
    title="🔍 Hybrid CNN+ViT Deepfake Detector",
    description="Upload an image to classify whether it is real or fake using the hybrid CNN+ViT model.",
    theme="soft"
)

if __name__ == "__main__":
    print("🚀 Starting Hybrid CNN+ViT Deepfake Detector...")
    iface.launch(inbrowser=True)
