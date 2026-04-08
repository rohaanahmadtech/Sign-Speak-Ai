import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
from gtts import gTTS
import os

print("=" * 50)
print("✅ All imports successful!")
print("=" * 50)
print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"MediaPipe: {mp.__version__}")
print(f"OpenCV: {cv2.__version__}")
print("=" * 50)

# Check if model files exist
files = ['signspeak_model.keras', 'hand_landmarker.task', 'label_map.npy']
for file in files:
    if os.path.exists(file):
        print(f"✅ {file} found")
    else:
        print(f"❌ {file} NOT found - make sure it's in the project folder")

print("\n🎯 Ready to run SignSpeak!")