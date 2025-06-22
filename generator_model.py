import torch
import torch.nn as nn

class DigitGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(DigitGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

def load_generator(model_path="saved_model/generator.pt"):
    model = DigitGenerator()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_digit_images(model, digit, num_images=5):
    noise = torch.randn(num_images, 100)
    labels = torch.full((num_images,), digit, dtype=torch.long)
    with torch.no_grad():
        images = model(noise, labels)
    return images
