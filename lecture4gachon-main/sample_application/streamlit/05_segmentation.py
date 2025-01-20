import streamlit as st

from io import BytesIO

import numpy as np
from PIL import Image

import torch
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.utils import  draw_segmentation_masks

# About Segmentation
segmentation_models = {
    "deeplabv3_mobilenet_v3_large": models.segmentation.deeplabv3_mobilenet_v3_large,  
    "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101, 
    "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50, 
    "fcn_resnet101": models.segmentation.fcn_resnet101, 
    "fcn_resnet50": models.segmentation.fcn_resnet50, 
    "lraspp_mobilenet_v3_large": models.segmentation.lraspp_mobilenet_v3_large
}

st.sidebar.title(f"Segmentation Model Test")

model_name = st.sidebar.selectbox("Select Model",segmentation_models)

model = segmentation_models[model_name](weights=True, progress=False)
model.eval()

uploaded_file = st.sidebar.file_uploader("Choose a Image")

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    image = np.array(image)
    image = torch.Tensor(image).type(torch.uint8).permute(2,0,1)
    tensor = image.unsqueeze(dim=0)/255.
    tensor = F.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    with torch.no_grad():
        prediction = model(tensor)
        
    normalized_masks = torch.nn.functional.softmax(prediction['out'], dim=1)
    num_classes = normalized_masks.shape[1]
    masks = normalized_masks[0]

    all_classes_masks = masks.argmax(0) == torch.arange(num_classes)[:, None, None]
    img_with_all_masks = draw_segmentation_masks(image, masks=all_classes_masks, alpha=.6)
    img_with_all_masks = img_with_all_masks.numpy().transpose(1,2,0)
    
    st.image(img_with_all_masks, use_column_width=True)