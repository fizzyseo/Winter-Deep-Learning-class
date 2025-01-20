import streamlit as st

import json
from io import BytesIO

import numpy as np
from PIL import Image

import torch
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import timm

# =====================================
# =====================================
#           GLOBAL VARIABLE
# =====================================
# =====================================

# About Classification
classification_models = {
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn", "vgg19",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
}
with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# About Detection
detection_models = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "fasterrcnn_mobilenet_v3_large_fpn": models.detection.fasterrcnn_mobilenet_v3_large_fpn, 
    "fasterrcnn_resnet50_fpn": models.detection.fasterrcnn_resnet50_fpn, 
    # "maskrcnn_resnet50_fpn": models.detection.maskrcnn_resnet50_fpn, 
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

# About Segmentation
segmentation_models = {
    "deeplabv3_mobilenet_v3_large": models.segmentation.deeplabv3_mobilenet_v3_large,  
    "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101, 
    "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50, 
    "fcn_resnet101": models.segmentation.fcn_resnet101, 
    "fcn_resnet50": models.segmentation.fcn_resnet50, 
    "lraspp_mobilenet_v3_large": models.segmentation.lraspp_mobilenet_v3_large
}
sem_classes = [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

# =====================================
# =====================================
#               FUNCTION
# =====================================
# =====================================

def mode_selector(mode):
    if mode.lower()=="classification":
        option = st.sidebar.selectbox(
                "Select Model", 
                classification_models)

    elif mode.lower()=="detection":
        option = st.sidebar.selectbox(
                "Select Model",
                detection_models)

    else:
        option = st.sidebar.selectbox(
                "Select Model",
                segmentation_models)

    return option

def set_controller(mode):
    if mode.lower()=="classification":
        option = None

    elif mode.lower()=="detection":
        class_select = st.sidebar.selectbox("Select Class", COCO_INSTANCE_CATEGORY_NAMES)
        conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        option = [class_select, conf]

    else:
        class_select = st.sidebar.selectbox("Select Class", sem_classes)
        conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        option = [class_select, conf]
        
    return option

def load_model(mode, model_name):
    
    if mode.lower()=="classification":
        model = timm.create_model(model_name, pretrained=True)
    elif mode.lower()=="detection":
        model = detection_models[model_name](weights=True, progress=False)
    else:
        model = segmentation_models[model_name](weights=True, progress=False)
    return model
    
def load_image(bytes_data, mode):
    img = Image.open(BytesIO(bytes_data)).convert("RGB")
    plot_img = np.array(img)
    img = torch.Tensor(plot_img).type(torch.uint8).permute(2,0,1)
    batch_img = img.unsqueeze(dim=0)/255.
    if mode.lower() != "detection":
        batch_img = F.normalize(batch_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img, batch_img

def inference(image, prediction, mode, option):
    result = None
    if mode.lower()=="classification":
        # Top 1~3
        prediction = prediction.squeeze(dim=0)
        predict_idx = prediction.argmax().item()
    
        prob = torch.softmax(prediction, dim=0)
        st.image(image.numpy().transpose(1, 2 ,0))
        st.text(f"{idx2label[predict_idx]}, {prob[predict_idx]}")
        result = None

    elif mode.lower()=="detection":
        prediction = prediction[0]
        valid_box_list = (prediction["scores"] > option[1]) * (prediction["labels"] == COCO_INSTANCE_CATEGORY_NAMES.index(option[0]))
        num_box = valid_box_list.sum()
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in prediction['labels'][valid_box_list]]
        img_with_boxes = draw_bounding_boxes(image, boxes=prediction['boxes'][valid_box_list], labels=labels, colors = [(126, 200, 128)] * num_box, fill=True, width=4)
        img_with_boxes = img_with_boxes.numpy().transpose(1,2,0)
        result = img_with_boxes
        st.image(result, use_column_width=True)

    else:
        normalized_masks = torch.nn.functional.softmax(prediction['out'], dim=1)
        num_classes = normalized_masks.shape[1]
        masks = normalized_masks[0]
        class_dim = 0
        all_classes_masks = masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
        img_with_all_masks = draw_segmentation_masks(image, masks=all_classes_masks[sem_class_to_idx[option[0]]], alpha=.6, colors=(240, 10, 157))
        img_with_all_masks = img_with_all_masks.numpy().transpose(1,2,0)
        result = img_with_all_masks
        st.image(result, use_column_width=True)
    return result

def main():
    # global model
    st.sidebar.title(f"Pretrained Model Test")
    mode = st.sidebar.selectbox("Choose the app mode",
            ["Classification", "Detection", "Segmentation"])

    model_name = mode_selector(mode)

    model = load_model(mode, model_name)
    model.eval()

    uploaded_file = st.sidebar.file_uploader("Choose a Image")
    option = set_controller(mode)
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        img, batch_img = load_image(bytes_data, mode)
        pred = model(batch_img)
        inference(img, pred, mode, option)
        
if __name__ == "__main__":
    main()