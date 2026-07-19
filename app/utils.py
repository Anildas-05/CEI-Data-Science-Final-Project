import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def predict_image(model, image_tensor, classes, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    return (
        classes[predicted.item()],
        confidence.item()
    )


def extract_embedding(feature_extractor, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        embedding = feature_extractor(image_tensor)

    embedding = embedding.view(-1)

    return embedding.cpu().numpy()


def calculate_similarity(embedding1, embedding2):

    tensor1 = torch.tensor(embedding1)
    tensor2 = torch.tensor(embedding2)

    similarity = F.cosine_similarity(
        tensor1.unsqueeze(0),
        tensor2.unsqueeze(0)
    )

    return similarity.item()