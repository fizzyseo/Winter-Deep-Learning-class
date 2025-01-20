import streamlit as st

from io import BytesIO

import numpy as np
from PIL import Image

import torch
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes

detection_models = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "fasterrcnn_mobilenet_v3_large_fpn": models.detection.fasterrcnn_mobilenet_v3_large_fpn, 
    "fasterrcnn_resnet50_fpn": models.detection.fasterrcnn_resnet50_fpn, 
    "retinanet_resnet50_fpn": models.detection.retinanet_resnet50_fpn, 
    "ssdlite320_mobilenet_v3_large": models.detection.ssdlite320_mobilenet_v3_large
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

INSTANCE_CATEGORY_COLORS = [tuple(np.random.randint(0, 254, (3))) for _ in COCO_INSTANCE_CATEGORY_NAMES]

st.sidebar.title(f"Detection Model Test")

model_name = st.sidebar.selectbox("Select Model", detection_models)

model = detection_models[model_name](weights=True, progress=False)
model.eval()

uploaded_file = st.sidebar.file_uploader("Choose a Image")

threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    image = np.array(image)
    
    image = torch.Tensor(image).type(torch.uint8).permute(2,0,1)
    tensor = image.unsqueeze(dim=0)/255.
    
    with torch.no_grad():
        prediction = model(tensor)[0]
    
    valid_box_list = (prediction["scores"] > threshold)
    labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in prediction['labels'][valid_box_list]]
    img_with_boxes = draw_bounding_boxes(image, 
                                        boxes=prediction['boxes'][valid_box_list], 
                                        labels=labels, 
                                        colors = [INSTANCE_CATEGORY_COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(label)] for label in labels], 
                                        fill=True, 
                                        width=2)
    img_with_boxes = img_with_boxes.numpy().transpose(1,2,0)    

    st.image(img_with_boxes, use_column_width=True)
