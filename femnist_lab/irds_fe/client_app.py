"""irds-fe: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from irds_fe.task import ImprovedCI_VAE_NUMERICAL, CI_VAE_NUMERICAL, get_weights, load_data, set_weights, test, train, fix_random, improved_train
import logging
import os

#logging.basicConfig(level=logging.INFO, format="%(asctime)s - CLIENT - %(levelname)s - %(message)s")



# Define Flower Client and client_fn
# Modifica della classe FlowerClient in client_app.py

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, rounds, latent_dim, 
                 batch_size, learning_rate, gamma, hidden_size, step, fraction, 
                 strategy_name, **strategy_params):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.rounds = rounds
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.step = step
        self.fraction = fraction
        self.strategy_name = strategy_name
        
        # Memorizza tutti i parametri specifici della strategia
        self.strategy_params = strategy_params
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        train_loss = improved_train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            self.learning_rate,
            self.gamma,
            self.step
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy, f1score, precision, recall = test(self.net, self.valloader, self.device)
        
        # Metriche base
        metrics = {
            "latent_dim": self.latent_dim,
            "rounds": self.rounds,
            "epochs": self.local_epochs,
            "batch-size": self.batch_size,
            "learning-rate": self.learning_rate,
            "gamma": self.gamma,
            "hidden_dim": self.hidden_size,
            "step": self.step,
            "fraction-fit": self.fraction,
            "strategy": self.strategy_name,
            "accuracy": accuracy,
            "f1score": f1score,
            "precision": precision,
            "recall": recall,
            "loss": float(loss)
        }
        
        # Aggiungi parametri specifici della strategia
        metrics.update(self.strategy_params)
        
        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    fix_random(42)
    input_size = 1280
    hs = context.run_config["hidden-dim"]
    ld = context.run_config["latent-dim"]
    net = ImprovedCI_VAE_NUMERICAL(input_size=input_size, hidden_size=hs, latent_dim=ld, num_classes=10)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config.get("batch-size", 32)
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size=batch_size)
    local_epochs = context.run_config["local-epochs"]
    rounds = context.run_config["num-server-rounds"]
    learning_rate = context.run_config.get("learning-rate", 0.01)
    gamma = context.run_config.get("gamma", 0.1)
    step = context.run_config.get("step", 20)
    fraction = context.run_config.get("fraction-fit", 0.5)
    strategy_name = context.run_config.get("strategy", "fedavg")

    # ===== RACCOLTA PARAMETRI SPECIFICI STRATEGIA =====
    strategy_params = {}
    
    # Parametri FedVAE
    if "reconstruction_weight" in context.run_config:
        strategy_params["reconstruction_weight"] = context.run_config["reconstruction_weight"]
    if "classification_weight" in context.run_config:
        strategy_params["classification_weight"] = context.run_config["classification_weight"]
    if "kld_weight" in context.run_config:
        strategy_params["kld_weight"] = context.run_config["kld_weight"]
    if "beta_vae" in context.run_config:
        strategy_params["beta_vae"] = context.run_config["beta_vae"]
    
    # Parametri FedProx
    if "proximal_mu" in context.run_config:
        strategy_params["proximal_mu"] = context.run_config["proximal_mu"]
    
    # Parametri FedAdagrad/FedAdam
    if "eta" in context.run_config:
        strategy_params["eta"] = context.run_config["eta"]
    if "beta_1" in context.run_config:
        strategy_params["beta_1"] = context.run_config["beta_1"]
    if "beta_2" in context.run_config:
        strategy_params["beta_2"] = context.run_config["beta_2"]
    
    # Parametri Early Stopping
    if "patience" in context.run_config:
        strategy_params["patience"] = context.run_config["patience"]
    if "min_delta" in context.run_config:
        strategy_params["min_delta"] = context.run_config["min_delta"]

    return FlowerClient(
        net, trainloader, valloader, local_epochs, rounds, ld, 
        batch_size, learning_rate, gamma, hs, step, fraction, 
        strategy_name, **strategy_params
    ).to_client()
    
# Flower ClientApp
app = ClientApp(
    client_fn,
)
