import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import os
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the main classifier (Detector_best_model.pth)
main_model = models.resnet18(weights=None)  # Updated: weights=None
num_ftrs = main_model.fc.in_features
# main_model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: AI-generated_Image, Real_Image
main_model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # Match the training architecture
    nn.Linear(num_ftrs, 2)  # 2 classes: AI-generated Image, Real Image
)

main_model.load_state_dict(torch.load('best_model9.pth', map_location=device, weights_only=True))  # Updated: weights_only=True
main_model = main_model.to(device)
main_model.eval()

# Define class names for the classifier based on the Folder structure
classes_name = ['AI-generated Image', 'Real Image']

def convert_to_rgb(image):
    """
    Converts 'P' mode images with transparency to 'RGBA', and then to 'RGB'.
    This is to avoid transparency issues during model training.
    """
    if image.mode in ('P', 'RGBA'):
        return image.convert('RGB')
    return image

# Define preprocessing transformations (same used during training)
preprocess = transforms.Compose([
    transforms.Lambda(convert_to_rgb),
    transforms.Resize((224, 224)),  # Resize here, no need for shape argument in gr.Image
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

def classify_image(image):
    # Open the image using PIL
    image = Image.fromarray(image)
    
    # Preprocess the image
    input_image = preprocess(image).unsqueeze(0).to(device)
    
    # Perform inference with the main classifier
    with torch.no_grad():
        output = main_model(input_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
    
    # Main classifier result
    main_prediction = classes_name[predicted_class]
    main_confidence = confidence.item()
    
    return f"Image is : {main_prediction} (Confidence: {main_confidence:.4f})"

# Gradio interface (updated)
image_input = gr.Image(image_mode="RGB")  # Removed shape argument
output_text = gr.Textbox()

gr.Interface(fn=classify_image, inputs=image_input, outputs=[output_text], 
             title="Detect AI-generated Image ",
             description="Upload an image to Detected AI-generated Image .",
             theme="default").launch()
