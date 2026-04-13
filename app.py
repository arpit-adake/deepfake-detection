import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- F3NET ----------------
def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class F3Net(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

        self.rgb_stream = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )

        self.fft_stream = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, rgb, fft):
        rgb_feat = self.gap(self.rgb_stream(rgb)).flatten(1)
        fft_feat = self.gap(self.fft_stream(fft)).flatten(1)
        return self.head(torch.cat([rgb_feat, fft_feat], dim=1))
    
# ---------------- DINO ----------------
class DINOModel(nn.Module):
    def __init__(self):
        super().__init__()

        # embedder
        self.embedder = nn.Module()
        self.embedder.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14"
        )

        # classifier
        self.classifier = nn.Module()
        self.classifier.net = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            emb = self.embedder.backbone(x)
        return self.classifier.net(emb)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    # EfficientNet
    effnet = timm.create_model("efficientnet_b4", num_classes=1)
    effnet.load_state_dict(torch.load("effnetb4_finetuned.pth", map_location=device))
    effnet.to(device).eval()

    # F3Net
    f3net = F3Net()
    f3net.load_state_dict(torch.load("f3net_final.pth", map_location=device))
    f3net.to(device).eval()

    # DINO
    dino = DINOModel()
    dino.load_state_dict(torch.load("dinov2_mlp_best.pth", map_location=device))
    dino.to(device).eval()

    return effnet, f3net, dino

effnet, f3net, dino = load_models()

# ---------------- TRANSFORMS ----------------
effnet_tf = transforms.Compose([
    transforms.Resize(352),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
])

f3_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

dino_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

# ---------------- FFT ----------------
def fft_img(img):
    gray = np.array(img.convert("L").resize((256,256))) / 255.0
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return torch.tensor(mag).unsqueeze(0).float()

# ---------------- PREDICTION ----------------
@torch.no_grad()
def predict(image):
    # EfficientNet
    e = effnet_tf(image).unsqueeze(0).to(device)
    p1 = torch.sigmoid(effnet(e)).item()

    # F3Net
    rgb = f3_tf(image).unsqueeze(0).to(device)
    fft = fft_img(image).unsqueeze(0).to(device)
    p2 = torch.sigmoid(f3net(rgb, fft)).item()

    # DINO
    d = dino_tf(image).unsqueeze(0).to(device)
    p3 = torch.sigmoid(dino(d)).item()

    # Ensemble
    final = 0.5 * p1 + 0.3 * p2 + 0.2 * p3

    return final, p1, p2, p3
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# -------- HEADER --------
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🧠 AI-Generated Image Detector
</h1>
<p style='text-align: center; font-size:18px;'>
Upload an image and detect whether it is <b>REAL</b> or <b>AI-GENERATED</b>
</p>
<hr>
""", unsafe_allow_html=True)

# -------- FILE UPLOAD --------
uploaded = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.markdown("### 🖼️ Preview")
    st.image(image, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- BUTTON --------
    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing... Please wait ⏳"):
            score, p1, p2, p3 = predict(image)

        st.markdown("## 🔎 Result")

        # -------- RESULT --------
        if score > 0.5:
            st.error(f"⚠️ AI GENERATED")
        else:
            st.success(f"✅ REAL IMAGE")

        # -------- CONFIDENCE BAR --------
        st.markdown("### 📊 Confidence Score")
        st.progress(int(score * 100))
        st.write(f"**Confidence:** {score*100:.2f}%")

        st.markdown("---")

