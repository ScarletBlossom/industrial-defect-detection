import torch
from PIL import Image
from torchvision import transforms
from src.models import get_model, load_checkpoint
from src.config import Config


class DefectDetector:
    def __init__(self, checkpoint_path='checkpoints/best_model.pth', device=None):
        self.device = device or Config.DEVICE
        self.model = get_model(num_classes=Config.NUM_CLASSES, pretrained=False, device=self.device)
        self.model = load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['Normal', 'Defect']
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
    
    def predict_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }


if __name__ == '__main__':
    detector = DefectDetector()
    result = detector.predict('data/test/normal/normal_0000.jpg')
    print(result)