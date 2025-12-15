import torch
import torch.nn as nn

def train_autoencoder(model, dataloader, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_log = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        loss_log.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    return model, loss_log


def compute_reconstruction_error(model, X_tensor):
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon)**2, dim=1).numpy()
    return errors
