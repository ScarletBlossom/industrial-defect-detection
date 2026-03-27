import torch
import torch.nn as nn
from torchvision import models


class DefectClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DefectClassifier, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def get_model(num_classes=2, pretrained=True, device='cpu'):
    model = DefectClassifier(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model


def load_checkpoint(checkpoint_path, model, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == '__main__':
    model = get_model(num_classes=2, pretrained=True)
    print(model)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")