import streamlit as st
import torch
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
import os
import librosa
import numpy as np

# Load the trained model (Assuming a PyTorch model is trained and saved as endangered_species_model.pth)
class EndangeredSpeciesCNN(torch.nn.Module):
    def __init__(self):
        super(EndangeredSpeciesCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(32 * 64 * 64, 128)
        self.fc2 = torch.nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Model
model = EndangeredSpeciesCNN()
model.load_state_dict(torch.load("endangered_species_model.pth"))
model.eval()

# Transform for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to classify image
def classify_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return "Endangered" if predicted.item() == 1 else "Not Endangered"

# Function to process audio
def process_audio(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# Streamlit UI
st.title("Endangered Species Classifier")
st.write("Upload an image, enter a species name, or provide an animal sound.")

# Text Input for Animal Name
animal_name = st.text_input("Enter Animal Name")
if animal_name:
    st.write(f"Searching information for {animal_name}...")
    # Placeholder for actual endangered status check
    st.write("Status: Endangered/Not Endangered/Not Found")

# Image Upload
uploaded_file = st.file_uploader("Upload an animal image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result = classify_image(image)
    st.write(f"Classification: {result}")

# Audio Upload
audio_file = st.file_uploader("Upload an animal sound (wav/mp3)", type=["wav", "mp3"])
if audio_file:
    audio_features = process_audio(audio_file)
    st.write("Analyzing audio...")
    st.write("Status: Endangered/Not Endangered/Not Found (Placeholder)")
