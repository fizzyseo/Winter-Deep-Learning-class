import streamlit as st

import json
from io import BytesIO

import numpy as np
from PIL import Image

import timm
import torch
from torchvision import transforms

available_models = [
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn", "vgg19",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
]

with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    
model_name = st.selectbox('Select Model', available_models)

model = timm.create_model(model_name, pretrained=True)
model.eval()

uploaded_file = st.file_uploader("Choose a Image")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    img_for_plot = np.array(image)
    
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(tensor)[0], dim=0)
    confidences = {idx2label[i]: float(prediction[i]) for i in range(1000)}    
    confidences = sorted(confidences.items(), key=lambda x:x[1], reverse=True)
    
    st.image(img_for_plot, use_column_width=True)
    st.write(f"{confidences[0][0]}: {confidences[0][1]:.4f}")
    st.write(f"{confidences[1][0]}: {confidences[1][1]:.4f}")
    st.write(f"{confidences[2][0]}: {confidences[2][1]:.4f}")