import gradio as gr

import os
import json

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
    
def predict(image, model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(tensor)[0], dim=0)
    confidences = {idx2label[i]: float(prediction[i]) for i in range(1000)}    
    return confidences

img_file_format = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
example_root = "test_img"
example_img_list = [[os.path.join(example_root, file), ] for file in os.listdir(example_root) if file.split(".")[-1].lower() in img_file_format]

gr.Interface(
            title="Image Classification",
            fn=predict, 
            inputs=[
                gr.Image(type="pil"), 
                gr.Dropdown(available_models, value="efficientnet_b0")
                ],
            outputs=gr.Label(num_top_classes=3),
            examples= example_img_list,
            flagging_dir="flagged/classification"
            ).launch()