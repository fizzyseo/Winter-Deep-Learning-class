import gradio as gr

import os

import numpy as np

import torch
from torchvision import models
from torchvision.transforms import functional as F
from torchvision.utils import draw_segmentation_masks

# About Segmentation
segmentation_models = {
    "deeplabv3_mobilenet_v3_large": models.segmentation.deeplabv3_mobilenet_v3_large,  
    "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101, 
    "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50, 
    "fcn_resnet101": models.segmentation.fcn_resnet101, 
    "fcn_resnet50": models.segmentation.fcn_resnet50, 
    "lraspp_mobilenet_v3_large": models.segmentation.lraspp_mobilenet_v3_large
}

def predict(image, model_name):
    # Preparing About Model Inference
    model = segmentation_models[model_name](pretrained=True, progress=False)
    model.eval()
    
    image = np.array(image)
    image = torch.Tensor(image).type(torch.uint8).permute(2,0,1)
    tensor = image.unsqueeze(0)/255.
    tensor = F.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    with torch.no_grad():
        prediction = model(tensor)
        
    normalized_masks = torch.nn.functional.softmax(prediction['out'], dim=1)
    num_classes = normalized_masks.shape[1]
    masks = normalized_masks[0]
    
    all_classes_masks = masks.argmax(0) == torch.arange(num_classes)[:, None, None]
    img_with_all_masks = draw_segmentation_masks(image, masks=all_classes_masks, alpha=.6)
    img_with_all_masks = img_with_all_masks.numpy().transpose(1,2,0)
    return img_with_all_masks

img_file_format = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
example_root = "test_img"
example_img_list = [[os.path.join(example_root, file)] for file in os.listdir(example_root) if file.split(".")[-1].lower() in img_file_format]

gr.Interface(
            title="Segmentation",
            fn=predict, 
            inputs=[
                gr.Image(type="pil"),
                gr.Dropdown(list(segmentation_models.keys()), value="deeplabv3_mobilenet_v3_large"),
                ], 
            outputs=gr.Image(type="pil"),
            examples= example_img_list,
            flagging_dir="flagged/segmentation"
            ).launch()