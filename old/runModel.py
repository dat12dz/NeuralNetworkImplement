import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# Define the same CNN model used during training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 8 * 8, num_classes)  # 32x8x8 after two max pools

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
num_classes = 19  # Adjust based on the number of classes in your dataset
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),      # Resize to 32x32
    transforms.Grayscale(),           # Ensure grayscale
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Function to predict the class of an image
def predict_image(image_path, model, transform, class_names):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    return class_names[predicted_class.item()]

# Main function to run predictions
def main():
    # Define class names based on your dataset
    class_names = ['0','1','2','3','4','5','6','7','8','9','+','dec','/','eq','*','-','x','y','z']  # Adjust based on your dataset
    
    # Directory containing test images
    test_dir = 'testcase'  # Create this directory and place test images inside
    
    # List all images in the test directory
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        predicted_class = predict_image(image_path, model, transform, class_names)
        print(f'Image: {image_file} | Predicted Class: {predicted_class}')

if __name__ == '__main__':
    main()