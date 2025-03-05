import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

main_model = models.resnet18(weights=None)  
num_ftrs = main_model.fc.in_features

main_model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  
    nn.Linear(num_ftrs, 2)  
)

main_model.load_state_dict(torch.load('RESNET-18-best_model9.pth', map_location=device, weights_only=True))  
main_model = main_model.to(device)
main_model.eval()

classes_name = ['AI-generated Image', 'Real Image']

def convert_to_rgb(image):
    if image.mode in ('P', 'RGBA'):
        return image.convert('RGB')
    return image

preprocess = transforms.Compose([
    transforms.Lambda(convert_to_rgb),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

def classify_image(image):
    image = Image.fromarray(image)
    
    input_image = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = main_model(input_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
    
    main_prediction = classes_name[predicted_class]
    main_confidence = confidence.item()
    
    return f"Image is : {main_prediction} (Confidence: {main_confidence:.4f})"

image_input = gr.Image(image_mode="RGB")  
output_text = gr.Textbox()

gr.Interface(fn=classify_image, inputs=image_input, outputs=[output_text], 
             title="Detect AI-generated Image ",
             description="Upload an image to Detected AI-generated Image .",
             theme="default").launch()
