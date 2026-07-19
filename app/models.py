import torch
import torch.nn as nn
from torchvision import models


# -------------------------------
# Device
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# EuroSAT Classes
# -------------------------------

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


# -------------------------------
# UC Merced Classes
# -------------------------------

UCMERCED_CLASSES = [
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt"
]


# -------------------------------
# Model Loader
# -------------------------------

def load_model(dataset="EuroSAT"):

    model = models.resnet18(weights=None)

    if dataset == "EuroSAT":
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(
            torch.load(
                "models/resnet18_best.pth",
                map_location=device
            )
        )
        classes = EUROSAT_CLASSES

    else:
        model.fc = nn.Linear(model.fc.in_features, 21)
        model.load_state_dict(
            torch.load(
                "models/resnet18_ucmerced_final.pth",
                map_location=device
            )
        )
        classes = UCMERCED_CLASSES

    model.to(device)
    model.eval()

    return model, classes

def get_feature_extractor(model):

    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor