import joblib
import torch

def save_sklearn_model(model, path):
    joblib.dump(model, path)

def save_autoencoder(model, path):
    torch.save(model.state_dict(), path)
