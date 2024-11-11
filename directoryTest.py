import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate a trained ResNet-18 model on a test dataset.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pth)')
parser.add_argument('--test_dir', type=str, required=True, help='Path to the directory containing test images')
args = parser.parse_args()

# Load the trained ResNet-18 model
model = models.resnet18()  # Load the architecture
num_classes = 10  # Set the number of classes to match your trained model

# Modify the fully connected layer to match the number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the trained weights with device compatibility
model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)
model = model.to(device)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

model.eval()  # Set the model to evaluation mode

# Define image transformations (resize and normalize according to model requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as needed
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Use appropriate mean and std
])

# Create dataset and dataloader
test_dataset = datasets.ImageFolder(root=args.test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Adjust batch_size as needed

# Evaluation setup
all_labels = []
all_preds = []
all_probs = []
running_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)  # Forward pass
        if torch.isnan(outputs).any():
            print("NaN detected in output! Skipping batch.")
            continue  # Skip this batch if NaNs are detected

        loss = criterion(outputs, labels)  # Calculate the loss
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)  # Get the class with the highest score
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
        
        all_labels.extend(labels.cpu().numpy())  # Store true labels
        all_preds.extend(preds.cpu().numpy())    # Store predicted labels
        all_probs.extend(probs.cpu().numpy())    # Store probabilities

avg_loss = running_loss / len(test_loader.dataset)  # Calculate average loss

# Calculate overall metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Class labels from ImageFolder (assuming they match your classes)
classes = test_dataset.classes
# Create confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
