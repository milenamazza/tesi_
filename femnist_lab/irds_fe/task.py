"""irds-fe: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
import os
import json
from typing import Tuple, List
from sklearn.metrics import f1_score, precision_score, recall_score
import copy
import random

user_path = ''
def fix_random(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # slower

class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_prob=0.2):
        super(FFNN, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_prob),
        )

        self.branch1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Dropout(dropout_prob),
        )

        self.branch2 = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size//4),
            nn.Dropout(dropout_prob),
        )

        self.branch3 = nn.Sequential(
            nn.Linear(hidden_size//4, hidden_size//8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size//8),
            nn.Dropout(dropout_prob),
        )


        self.branch4 = nn.Sequential(
            nn.Linear(hidden_size//8, hidden_size//16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_size//16),
            nn.Dropout(dropout_prob),
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size//16, output_size),
        )
        
    def forward(self, x):
        x = self.input(x)
        x = self.branch1(x)
        x = self.branch2(x)
        x = self.branch3(x)
        x = self.branch4(x)
        o  = self.output(x)
        return o
    
class CI_VAE_NUMERICAL(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim=7, num_classes=10, device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")):
        super(CI_VAE_NUMERICAL, self).__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        # encoder
        self.input = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(0.2),

        )
        self.block1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.Dropout(0.2),
        )
        self.block2 = nn.Sequential(
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_size//4),
            nn.Dropout(0.2),
        )
        self.mu = nn.Linear(self.hidden_size//4, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_size//4, self.latent_dim)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_classes),
        )

        # decoder
        self.linear1 = nn.Linear(self.latent_dim, self.hidden_size//4)
        self.block3 = nn.Sequential(
            nn.Linear(self.hidden_size//4, self.hidden_size//2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.Dropout(0.2),
        )
        self.block4 = nn.Sequential(
            nn.Linear(self.hidden_size//2, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(0.2),
        )
        self.output = nn.Linear(self.hidden_size, self.input_size)


    def encoder(self, x):
        x = self.input(x)
        x = self.block1(x)
        x = self.block2(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.device)
        return 0.5 * eps * std + mu

    def latent_classifier(self, z):
        return self.classifier(z)

    def decoder(self, z):
        x = self.linear1(z)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        class_out = self.latent_classifier(z)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, class_out
    
fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    base_path = os.path.join(user_path, "femnist_processed", f"client_{partition_id}")

    train_feat_path = os.path.join(base_path, "train_features.pt")
    train_label_path = os.path.join(base_path, "train_labels.pt")
    test_feat_path = os.path.join(base_path, "test_features.pt")
    test_label_path = os.path.join(base_path, "test_labels.pt")

    # Verifica se i file esistono
    for path in [train_feat_path, train_label_path, test_feat_path, test_label_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File mancante: {path}")

    train_features = torch.load(train_feat_path)
    train_labels = torch.load(train_label_path)
    test_features = torch.load(test_feat_path)
    test_labels = torch.load(test_label_path)

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        torch.utils.data.TensorDataset(test_features, test_labels),
        batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader



def loss_fn(recon_x, x, logvar, mu, class_out, labels):
    mse_recon = nn.MSELoss()
    recon_loss = mse_recon(recon_x, x)
    class_criterion = nn.CrossEntropyLoss()
    class_loss = class_criterion(class_out, labels.long())
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    loss = recon_loss + 3 * class_loss + kld_loss
    return loss, recon_loss, class_loss, kld_loss

def train(net, trainloader, epochs, device, learning_rate=0.01, gamma=0.1, step=20):

    """Train the feed-forward neural network on the training set."""
    net.to(device)  # Move the model to the GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    rec_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    best_val_loss = float('inf')
    epochs_since_last_improvement = 0
    net.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar, class_out = net(features)
            loss, _, _, _ = loss_fn(x_hat, features, logvar, mu, class_out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Compute the average training loss
        scheduler.step()
        avg_trainloss = running_loss / len(trainloader)

        # val_loss,_,_,_,_ = test(net, valloader, device)

        # # Early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     # best_model = copy.deepcopy(net)
        #     epochs_since_last_improvement = 0
        # elif epochs_since_last_improvement >= patience:
        #     break
        # else:
        #     epochs_since_last_improvement += 1
        
    #     print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_trainloss:.4f} - Validation Loss: {val_loss:.4f}, patience: {epochs_since_last_improvement}", end="\r")
    # print("\n")
    return avg_trainloss #, best_model

def test(net, testloader, device):
    """Validate the feed-forward neural network on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    y_pred = torch.tensor([], requires_grad=True).to(device)
    y_true = torch.tensor([], requires_grad=True).to(device)
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for batch in testloader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            x_hat, mu, logvar, class_out = net(features)
            loss, _, _, _ = loss_fn(x_hat, features, logvar, mu, class_out, labels)
            total_loss += loss.item()
            _, predicted = torch.max(class_out.data, 1)
            correct += (predicted == labels).sum().item()
            y_pred = torch.cat((y_pred, predicted), 0)
            y_true = torch.cat((y_true, labels), 0)

    # Compute accuracy and average loss
    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)

    # Compute F1-score, precision, and recall
    y_pred = y_pred.squeeze().cpu().detach().numpy()
    y_true = y_true.squeeze().cpu().detach().numpy()
    f1score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    return avg_loss, accuracy, f1score, precision, recall


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


############################ new function  ###############################


def fixed_loss_fn(recon_x, x, logvar, mu, class_out, labels, alpha=1.0):
    """
    Loss function corretta con scaling appropriato
    """
    # Reconstruction loss normalizzata
    mse_recon = nn.MSELoss(reduction='mean')
    recon_loss = mse_recon(recon_x, x)
    
    # Classification loss
    class_criterion = nn.CrossEntropyLoss()
    class_loss = class_criterion(class_out, labels.long())
    
    # KL Divergence loss normalizzata
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
    
    # SCALING CORRETTO - normalizza le loss per evitare dominanza
    # Reconstruction loss tipicamente è nell'ordine di 0.1-1.0
    # Classification loss tipicamente è nell'ordine di 1.0-3.0
    # KLD loss tipicamente è nell'ordine di 0.01-0.1
    
    # Scala la reconstruction loss per essere comparabile alla classification
    scaled_recon_loss = recon_loss * 10.0  # Aumenta peso reconstruction
    
    # Formula dal paper con scaling corretto
    total_loss = scaled_recon_loss + alpha * class_loss + kld_loss
    
    return total_loss, recon_loss, class_loss, kld_loss

# =============================================================================
# 2. ARCHITETTURA CI-VAE MIGLIORATA
# =============================================================================

class ImprovedCI_VAE_NUMERICAL(nn.Module):
    def __init__(self, input_size=1280, hidden_size=512, latent_dim=7, num_classes=10, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ImprovedCI_VAE_NUMERICAL, self).__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # ENCODER MIGLIORATO con più capacità
        self.encoder_layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.BatchNorm1d(self.hidden_size//4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Variational layers
        self.mu_layer = nn.Linear(self.hidden_size//4, self.latent_dim)
        self.logvar_layer = nn.Linear(self.hidden_size//4, self.latent_dim)
        
        # CLASSIFIER più robusto
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.latent_dim * 2, self.num_classes)
        )
        
        # DECODER MIGLIORATO - simmetrico all'encoder
        self.decoder_layers = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_size//4),
            nn.BatchNorm1d(self.hidden_size//4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(self.hidden_size//4, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_size//2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(self.hidden_size, self.input_size),
            # NO activation finale per permettere valori negativi
        )

    def encoder(self, x):
        h = self.encoder_layers(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Durante inference usa solo la media

    def latent_classifier(self, z):
        return self.classifier(z)

    def decoder(self, z):
        return self.decoder_layers(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        class_out = self.latent_classifier(z)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, class_out

# =============================================================================
# 3. TRAINING FUNCTION MIGLIORATA
# =============================================================================

def improved_train(net, trainloader, epochs, device, learning_rate=0.01, gamma=0.95, step=10):
    """
    Training function migliorata con le best practices
    """
    net.to(device)
    net.train()
    
    # Optimizer migliorato
    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.999)
    )
    
    # Scheduler più aggressivo
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    
    # Gradient clipping per stabilità
    max_grad_norm = 1.0
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (features, labels) in enumerate(trainloader):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            x_hat, mu, logvar, class_out = net(features)
            
            # Loss calculation con alpha dal paper
            loss, recon_loss, class_loss, kld_loss = fixed_loss_fn(
                x_hat, features, logvar, mu, class_out, labels, alpha=3.0
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        total_loss += epoch_loss / len(trainloader)
        
        # Log ogni 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return total_loss / epochs