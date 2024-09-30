import torch
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from tkinter import PhotoImage
from PIL import ImageTk

# Load the trained ResNet-18 model from a .pth file
model = models.resnet18()  # Load the architecture
num_classes = 10  # Set the number of classes to match your trained model

# Modify the fully connected layer to match the number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load('best_model.pth'))  # Load the trained weights
model.eval()  # Set the model to evaluation mode

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop the center to 224x224 (the input size of ResNet-18)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# List of class names
classes = ['Butler Hall', 'Carpenter Hall', 'Lee Hall', 'McCain Hall', 'McCool Hall', 'Old Main', 'Simrall Hall', 'Student Union', 'Swalm Hall', 'Walker Hall']

# Load and transform the image
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Make a prediction
def predict(image_path):
    image = load_image(image_path)
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the class index with the highest score
    return predicted.item()

# Function to open file dialog and predict image
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and display the image
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Make a prediction
        class_idx = predict(file_path)
        label.config(text=f"Predicted Class: {classes[class_idx]}")

# GUI Setup
root = tk.Tk()
root.title("Image Prediction")

# GUI Components
panel = Label(root)  # For displaying the image
panel.pack()

label = Label(root, text="Upload an image to predict the class")
label.pack()

btn = Button(root, text="Open Image", command=open_image)
btn.pack()

# Start the GUI event loop
root.mainloop()
