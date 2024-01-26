import torch
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils.watertopo import WaterTopo
from utils.simulation import Simulation

from utils.utils import create_sequence

def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0

    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_val_loss = float("inf")  # Track the best validation loss
    train_losses = []
    val_losses = []

    start_time = time.time()  # Start training time

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}', f'Train Loss: {avg_train_loss:.4f}, '
                  f'Validation Loss: {avg_val_loss:.4f}', f'Best Validation Loss: {best_val_loss:.4f}')
    train_time = time.time() - start_time
    print("Training complete.")
    return train_losses, val_losses, best_val_loss, train_time


def train(
        model,
        device,
        root,
        model_name,
        channels=2,
        T=5,
        H=1,
        sim_amount=3,
        training_size=0.8,
        use_augmented_data=True,
        batch_size=4,
        num_epochs = 200,
        lr = 0.0005,
        criterion = nn.MSELoss(),
        optimizer = optim.AdamW,
        ):

    """
    This method trains a simple RNN. Given a single timestep consisting of water depth and topography (both 64*64), the RNN predicts a single step ahead. The best model state is
    saved following the save_path, and also returned by the method.
    
    Description of arguments:
    - model: the model to be trained, should be an instance of the class SimpleRNN;
    - channels (int): number of channels that the loaded model is compatible with (default 2, i.e. DEM & WED);
    - sim_amount (int): number of simulations of which the data is loaded and used for training, with a maximum of 400;
    - training_size (float): fraction of data to use for training (validation uses the fraction 1 - training_size);
    - use_augmented_data (boolean): if True, the train function will include augmented data when (randomly) selecting training and validation data;
    - batch_size (int): batch size used during training (you can modify this based on your requirements);
    - T (int): number of input time steps. Make sure this number is compatible with the defined model;
    - H (int): number of predicted time steps (default 1). Make sure this number is compatible with the defined model;
    - num_epochs (int): number of epochs used during training;
    - lr (float): learning rate used during training;
    - criterion: Loss function, default nn.MSELoss()
    - optimizer: optimizer used for training, default optim.AdamW
    - defice: the device used for training the model (i.e. CUDA [GPU] or CPU).
    - model_name (string): the best model state will be saved in ../results/trained_models/ under this name

    returns: model, train_losses, val_losses, best_val_loss, time
    """
    # load simulations to be used for training
    if channels == 2:
        sims = WaterTopo.load_simulations(str(root)+"/data/normalized_data/tra_val", sim_amount=sim_amount, number_grids=64, use_augmented_data=use_augmented_data)
    elif channels == 4:
        sims = Simulation.load_simulations(str(root)+"/data/normalized_data/tra_val", sim_amount=sim_amount, number_grids=64, use_augmented_data=use_augmented_data)

    X, Y = create_sequence(sims, T, H)

    # We keep track of indexes of train and validation.
    X_tra, X_tst, Y_tra, Y_tst, ix_tra, ix_tst = train_test_split(
        X, Y, np.arange(X.shape[0]), test_size=1-training_size, shuffle=True, random_state=42)
    
    # Split the existing test dataset into validation and test sets (50/50 split)
    X_val, X_tst, Y_val, Y_tst, ix_val, ix_tst = train_test_split(
        X_tst, Y_tst, ix_tst, test_size=0.5, shuffle=True, random_state=42)
    
    
    #create datasets and data loaders
    train_dataset = TensorDataset(torch.tensor(X_tra, dtype=torch.float32), torch.tensor(Y_tra, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # defining the optimizer
    optimizer = optimizer(model.parameters(), lr=lr)

    # defining the save path
    save_path = "../results/trained_models/" + model_name

    # training
    train_losses, val_losses, best_val_loss, time = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path)

    # Load the best model
    model.load_state_dict(torch.load(save_path))

    return model, train_losses, val_losses, best_val_loss, time