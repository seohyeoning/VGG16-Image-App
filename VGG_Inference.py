import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models, transforms
from PIL import Image


class VGG16Inference:
    def __init__(self, model_path, classes, device="cpu"):
        self.device = torch.device(device)
        self.classes = classes

        self.model = models.vgg16(pretrained=False)
        n_inputs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n_inputs, len(classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    def preprocess_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)  
        return img_tensor

    def predict(self, img_path):
        input_tensor = self.preprocess_image(img_path).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1).squeeze().numpy() 
        _, predicted = torch.max(output, 1)
        label = self.classes[predicted.item()]
        cr_p = probabilities[0] * 100 
        uncr_p = probabilities[1] * 100  
        
        return label, cr_p, uncr_p

# TEST
if __name__ == "__main__":
    model_path = "C:/Users/user/Desktop/psh/project/KIVY/VGG_classification_app/best_model_weights.pth"
    test_dir = "C:/Users/user/Desktop/psh/project/KIVY/VGG_classification_app/test_data"
    classes = ['cracked', 'uncracked']

    vgg_inference = VGG16Inference(model_path=model_path, classes=classes)

    image_paths = [
        os.path.join(test_dir, img) for img in os.listdir(test_dir) if img.endswith(('.jpg', '.png'))
    ]

    for img_path in image_paths:
        label, cr_p, uncr_p = vgg_inference.predict(img_path)

        print(f"Image: {img_path} prediction: {label} cr_p: {cr_p}%, uncr_p: {uncr_p}%")
