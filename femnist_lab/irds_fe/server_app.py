import flwr
import logging
import os
import csv
import torch
import numpy as np

from flwr.common import (
    Context,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    Parameters,
    Scalar,
    Metrics,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx, FedAdagrad, FedAdam
from flwr.server.client_proxy import ClientProxy
from irds_fe.task import CI_VAE_NUMERICAL, get_weights, fix_random, ImprovedCI_VAE_NUMERICAL
from typing import List, Tuple, Union, Optional, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base Strategy with Early Stopping and Model Saving
class BaseStrategyWithEarlyStopping:
    """Mixin class per aggiungere early stopping basato su accuracy a qualsiasi strategia"""
    
    def __init__(self, *args, **kwargs):
        # Estrai parametri specifici per early stopping
        self.patience = kwargs.pop('patience', 35)
        self.min_delta = kwargs.pop('min_delta', 0.001)
        super().__init__(*args, **kwargs)
        
        self.best_accuracy = -1.0  # Inizializza con un valore molto basso
        self.rounds_no_improve = 0
        self.should_stop = False

    def check_early_stopping(self, server_round: int, accuracy: float):
        """Controlla se deve fermare l'addestramento basandosi sull'accuracy"""
        # Verifica se c'è un miglioramento significativo nell'accuracy
        if accuracy > self.best_accuracy + self.min_delta:
            improvement = accuracy - self.best_accuracy
            self.best_accuracy = accuracy
            self.rounds_no_improve = 0
            logger.info(f"Round {server_round}: New best accuracy {accuracy:.4f} (improvement: +{improvement:.4f})")
        else:
            self.rounds_no_improve += 1
            logger.info(f"Round {server_round}: Accuracy {accuracy:.4f}, no improvement for {self.rounds_no_improve} rounds (best: {self.best_accuracy:.4f}, delta needed: {self.min_delta:.4f})")
            
            if self.rounds_no_improve >= self.patience:
                #logger.info(f"Early stopping triggered at round {server_round} - no improvement in accuracy for {self.patience} rounds")
                self.should_stop = True


# Enhanced FedAvg with Early Stopping
class SaveModelStrategy(BaseStrategyWithEarlyStopping, FedAvg):
    current_round = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        SaveModelStrategy.current_round = server_round
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Early stopping check - assicurati che l'accuracy sia un float
        accuracy = float(aggregated_metrics.get("accuracy", 0.0))
        self.check_early_stopping(server_round, accuracy)
        
        return aggregated_loss, aggregated_metrics

# Weighted FedAvg Strategy
class WeightedFedAvgStrategy(BaseStrategyWithEarlyStopping, FedAvg):
    current_round = 0
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_performances = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        WeightedFedAvgStrategy.current_round = server_round
        
        if not results:
            return None, {}

        # Calcola i pesi basati sulle prestazioni dei client
        weights = []
        parameters_list = []
        
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            
            # Ottieni le prestazioni del client
            if client_id in self.client_performances:
                performance = self.client_performances[client_id].get('accuracy', 0.5)
                weight = max(0.1, performance)  # Peso minimo di 0.1
            else:
                weight = 1.0  # Peso default per nuovi client
            
            weights.append(weight * fit_res.num_examples)
            parameters_list.append(parameters_to_ndarrays(fit_res.parameters))

        # Normalizza i pesi
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Aggregazione pesata
        aggregated_ndarrays = [
            np.sum([weights[i] * param_array for i, param_array in enumerate(param_arrays)], axis=0)
            for param_arrays in zip(*parameters_list)
        ]

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        aggregated_metrics = {"weighted_avg_fit": sum(weights)}
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        if not results:
            return None, {}

        # Aggiorna le prestazioni dei client
        for client_proxy, evaluate_res in results:
            client_id = client_proxy.cid
            self.client_performances[client_id] = evaluate_res.metrics

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Early stopping check
        accuracy = float(aggregated_metrics.get("accuracy", 0.0))
        self.check_early_stopping(server_round, accuracy)
        
        return aggregated_loss, aggregated_metrics

# FedVAE Strategy for VAE models
class FedVAEStrategy(BaseStrategyWithEarlyStopping, FedAvg):
    current_round = 0
    
    def __init__(self, reconstruction_weight=0.3, classification_weight=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        FedVAEStrategy.current_round = server_round
        
        if not results:
            return None, {}

        # Pesa i client basandosi sulla train_loss
        weights = []
        parameters_list = []
        
        for client_proxy, fit_res in results:
            train_loss = fit_res.metrics.get('train_loss', 1.0)
            # Inverti la loss per ottenere un peso (loss più bassa = peso più alto)
            weight = 1.0 / (1.0 + train_loss)
            
            weights.append(weight * fit_res.num_examples)
            parameters_list.append(parameters_to_ndarrays(fit_res.parameters))

        # Normalizza i pesi
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Aggregazione pesata
        aggregated_ndarrays = [
            np.sum([weights[i] * param_array for i, param_array in enumerate(param_arrays)], axis=0)
            for param_arrays in zip(*parameters_list)
        ]

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        return aggregated_parameters, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Early stopping basato SOLO sull'accuracy
        accuracy = float(aggregated_metrics.get("accuracy", 0.0))
        self.check_early_stopping(server_round, accuracy)

        return aggregated_loss, aggregated_metrics

# Adaptive FedAvg Strategy
class AdaptiveFedAvgStrategy(BaseStrategyWithEarlyStopping, FedAvg):
    current_round = 0
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_performances = []
        self.initial_fraction_fit = self.fraction_fit

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        AdaptiveFedAvgStrategy.current_round = server_round
        
        # Adatta la frazione di client in base alle prestazioni passate
        if len(self.round_performances) > 3:
            recent_trend = np.mean(self.round_performances[-3:]) - np.mean(self.round_performances[-6:-3])
            
            if recent_trend < 0:  # Prestazioni in calo
                self.fraction_fit = min(1.0, self.fraction_fit * 1.1)
                #logger.info(f"Round {server_round}: Increasing client fraction to {self.fraction_fit:.2f}")
            else:  # Prestazioni in miglioramento
                self.fraction_fit = max(0.3, self.fraction_fit * 0.95)
                #logger.info(f"Round {server_round}: Adjusting client fraction to {self.fraction_fit:.2f}")

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Registra le prestazioni del round
        accuracy = float(aggregated_metrics.get("accuracy", 0.0))
        self.round_performances.append(accuracy)
        
        # Mantieni solo le ultime 10 prestazioni
        if len(self.round_performances) > 10:
            self.round_performances.pop(0)

        # Early stopping check
        self.check_early_stopping(server_round, accuracy)
        
        return aggregated_loss, aggregated_metrics

# Enhanced FedProx with Early Stopping
class FedProxWithEarlyStopping(BaseStrategyWithEarlyStopping, FedProx):
    current_round = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        FedProxWithEarlyStopping.current_round = server_round
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        accuracy = float(aggregated_metrics.get("accuracy", 0.0))
        self.check_early_stopping(server_round, accuracy)
        
        return aggregated_loss, aggregated_metrics

# Enhanced FedAdagrad with Early Stopping
class FedAdagradWithEarlyStopping(BaseStrategyWithEarlyStopping, FedAdagrad):
    current_round = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        FedAdagradWithEarlyStopping.current_round = server_round
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        accuracy = float(aggregated_metrics.get("accuracy", 0.0))
        self.check_early_stopping(server_round, accuracy)
        
        return aggregated_loss, aggregated_metrics

# Enhanced FedAdam with Early Stopping
class FedAdamWithEarlyStopping(BaseStrategyWithEarlyStopping, FedAdam):
    current_round = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        FedAdamWithEarlyStopping.current_round = server_round
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Metrics]],
        failures: List[Union[Tuple[ClientProxy, Metrics], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        accuracy = float(aggregated_metrics.get("accuracy", 0.0))
        self.check_early_stopping(server_round, accuracy)
        
        return aggregated_loss, aggregated_metrics

# Metric aggregation function
def simple_average(server_round: int, metrics: List[Tuple[int, Metrics]]) -> Metrics:
    clients_num = len(metrics)
    round_number = server_round
    
    if not metrics:
        return {}
    
    first = metrics[0][1]

    # Parametri comuni a tutte le strategie
    num_rounds = first.get("rounds", round_number)
    epochs = first.get("epochs", 5)
    latent_dim = first.get("latent_dim", 7)
    hidden_dim = first.get("hidden_dim", 512)
    learning_rate = first.get("learning-rate", 0.01)
    batch_size = first.get("batch-size", 32)
    gamma = first.get("gamma", 0.1)
    step = first.get("step", 20)
    fraction = first.get("fraction-fit", 0.5)
    strategy = first.get("strategy", "fedavg")

    # ===== GESTIONE INTELLIGENTE PARAMETRI SPECIFICI =====
    
    # Parametri FedVAE (attivi solo se strategy è "fedvae")
    if strategy.lower() == "fedvae":
        reconstruction_weight = first.get("reconstruction_weight", 0.3)
        classification_weight = first.get("classification_weight", 0.7)
        kld_weight = first.get("kld_weight", 1.0)
        beta_vae = first.get("beta_vae", 1.0)
    else:
        reconstruction_weight = 0.0
        classification_weight = 0.0
        kld_weight = 0.0
        beta_vae = 0.0

    # Parametri FedProx (attivi solo se strategy è "fedprox")
    if strategy.lower() == "fedprox":
        proximal_mu = first.get("proximal_mu", 0.1)
    else:
        proximal_mu = 0.0

    # Parametri FedAdagrad (attivi solo se strategy è "fedadagrad")
    if strategy.lower() == "fedadagrad":
        eta_adagrad = first.get("eta", 0.1)
    else:
        eta_adagrad = 0.0

    # Parametri FedAdam (attivi solo se strategy è "fedadam")
    if strategy.lower() == "fedadam":
        eta_adam = first.get("eta", 0.001)
        beta_1 = first.get("beta_1", 0.9)
        beta_2 = first.get("beta_2", 0.99)
    else:
        eta_adam = 0.0
        beta_1 = 0.0
        beta_2 = 0.0

    # Parametri Early Stopping (comuni ma opzionali)
    patience = first.get("patience", 35)
    min_delta = first.get("min_delta", 0.001)

    def avg(key):
        return sum([m.get(key, 0.0) for _, m in metrics]) / clients_num

    accuracy = avg("accuracy")
    f1score = avg("f1score")
    precision = avg("precision")
    recall = avg("recall")
    train_loss = avg("loss")

    filename = "ris-7-adam.csv"
    write_header = not os.path.exists(filename)
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Round", "Clients", "Latent Dim", "Hidden Dim", "Rounds", "Epochs",
                "Batch Size", "Learning Rate", "Gamma", "Step", "fraction-fit",
                "Strategy", 
                # Parametri FedVAE
                "Recon Weight", "Class Weight", "KLD Weight", "Beta VAE",
                # Parametri FedProx
                "Proximal Mu",
                # Parametri FedAdagrad
                "Eta Adagrad",
                # Parametri FedAdam
                "Eta Adam", "Beta 1", "Beta 2",
                # Parametri Early Stopping
                "Patience", "Min Delta",
                # Metriche
                "Accuracy", "F1 Score", "Precision", "Recall", "Train Loss"
            ])
        writer.writerow([
            round_number, clients_num, latent_dim, hidden_dim, num_rounds, epochs,
            batch_size, learning_rate, gamma, step, fraction,
            strategy,
            # Parametri FedVAE
            reconstruction_weight, classification_weight, kld_weight, beta_vae,
            # Parametri FedProx
            proximal_mu,
            # Parametri FedAdagrad
            eta_adagrad,
            # Parametri FedAdam
            eta_adam, beta_1, beta_2,
            # Parametri Early Stopping
            patience, min_delta,
            # Metriche
            accuracy, f1score, precision, recall, train_loss
        ])

    logger.info(
        f"[Round {round_number}] Strategy={strategy} | Acc={accuracy:.4f}, F1={f1score:.4f}, Prec={precision:.4f}, Recall={recall:.4f}, Loss={train_loss:.4f}"
    )

    return {
        "accuracy": accuracy,
        "f1score": f1score,
        "precision": precision,
        "recall": recall,
        "train_loss": train_loss,
    }


# Strategy-specific metric aggregators
def make_metrics_aggregator(strategy_class):
    def aggregator(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        return simple_average(strategy_class.current_round, metrics)
    return aggregator

# Model saving function
def save_final_model(parameters):
    model = ImprovedCI_VAE_NUMERICAL()
    weights = parameters_to_ndarrays(parameters)
    model.load_state_dict({name: torch.tensor(param) for name, param in zip(model.state_dict().keys(), weights)})
    torch.save(model.state_dict(), "final_model.pth")
    #logger.info("Final model saved to final_model.pth")

# Strategy factory function
def create_strategy(strategy_name: str, fraction_fit: float, parameters: Parameters, **kwargs):
    """Factory function per creare la strategia richiesta"""
    
    common_params = {
        'fraction_fit': fraction_fit,
        'fraction_evaluate': 1.0,
        'min_fit_clients': 2,
        'min_available_clients': 2,
        'initial_parameters': parameters,
        'patience': kwargs.get('patience', 35),
        'min_delta': kwargs.get('min_delta', 0.001),
    }
    
    if strategy_name.lower() == 'fedavg':
        strategy = SaveModelStrategy(**common_params)
        strategy.evaluate_metrics_aggregation_fn = make_metrics_aggregator(SaveModelStrategy)
        
    elif strategy_name.lower() == 'weighted_fedavg':
        strategy = WeightedFedAvgStrategy(**common_params)
        strategy.evaluate_metrics_aggregation_fn = make_metrics_aggregator(WeightedFedAvgStrategy)
        
    elif strategy_name.lower() == 'fedvae':
        vae_params = common_params.copy()
        vae_params.update({
            'reconstruction_weight': kwargs.get('reconstruction_weight', 0.3),
            'classification_weight': kwargs.get('classification_weight', 0.7),
        })
        strategy = FedVAEStrategy(**vae_params)
        strategy.evaluate_metrics_aggregation_fn = make_metrics_aggregator(FedVAEStrategy)
        
    elif strategy_name.lower() == 'adaptive_fedavg':
        strategy = AdaptiveFedAvgStrategy(**common_params)
        strategy.evaluate_metrics_aggregation_fn = make_metrics_aggregator(AdaptiveFedAvgStrategy)
        
    elif strategy_name.lower() == 'fedprox':
        prox_params = common_params.copy()
        prox_params['proximal_mu'] = kwargs.get('proximal_mu', 0.1)
        strategy = FedProxWithEarlyStopping(**prox_params)
        strategy.evaluate_metrics_aggregation_fn = make_metrics_aggregator(FedProxWithEarlyStopping)
        
    elif strategy_name.lower() == 'fedadagrad':
        ada_params = common_params.copy()
        ada_params['eta'] = kwargs.get('eta', 1e-1)
        strategy = FedAdagradWithEarlyStopping(**ada_params)
        strategy.evaluate_metrics_aggregation_fn = make_metrics_aggregator(FedAdagradWithEarlyStopping)
        
    elif strategy_name.lower() == 'fedadam':
        adam_params = common_params.copy()
        adam_params.update({
            'eta': kwargs.get('eta', 1e-3),
            'beta_1': kwargs.get('beta_1', 0.9),
            'beta_2': kwargs.get('beta_2', 0.99),
        })
        strategy = FedAdamWithEarlyStopping(**adam_params)
        strategy.evaluate_metrics_aggregation_fn = make_metrics_aggregator(FedAdamWithEarlyStopping)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: "
                        "fedavg, weighted_fedavg, fedvae, adaptive_fedavg, fedprox, fedadagrad, fedadam")
    
    return strategy

# ServerApp function
def server_fn(context: Context):
    fix_random(42)

    # Get configuration parameters
    num_rounds = context.run_config["num-server-rounds"]
    epochs = context.run_config["local-epochs"]
    fraction_fit = context.run_config["fraction-fit"]
    hs = context.run_config["hidden-dim"]
    ld = context.run_config["latent-dim"]
    input_size = 1280
    
    # Get strategy configuration
    strategy_name = context.run_config.get("strategy", "fedavg")
    patience = context.run_config.get("patience", 35)
    min_delta = context.run_config.get("min_delta", 0.001)
    
    # Strategy-specific parameters
    strategy_params = {
        'patience': patience,
        'min_delta': min_delta,
        'proximal_mu': context.run_config.get("proximal_mu", 0.1),
        'eta': context.run_config.get("eta", 1e-1),
        'beta_1': context.run_config.get("beta_1", 0.9),
        'beta_2': context.run_config.get("beta_2", 0.99),
        'reconstruction_weight': context.run_config.get("reconstruction_weight", 0.3),
        'classification_weight': context.run_config.get("classification_weight", 0.7),
    }

    # Initialize model and get parameters
    net = ImprovedCI_VAE_NUMERICAL(input_size=input_size, hidden_size=hs, latent_dim=ld, num_classes=10)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Create strategy based on configuration
    #logger.info(f"Creating strategy: {strategy_name} with patience={patience}, min_delta={min_delta}")
    strategy = create_strategy(strategy_name, fraction_fit, parameters, **strategy_params)

    # Configure server
    config = ServerConfig(num_rounds=num_rounds)
    
    # Set callback to save final model
    context.run_results_callback = lambda: save_final_model(strategy.parameters)

    # Create components and wrap to check for early stopping
    components = ServerAppComponents(strategy=strategy, config=config)
    original_evaluate = components.strategy.aggregate_evaluate

    def wrapped_evaluate(*args, **kwargs):
        res = original_evaluate(*args, **kwargs)
        if hasattr(strategy, 'should_stop') and strategy.should_stop:
            #logger.info("Early stopping triggered: exiting simulation")
            raise SystemExit("Early stopping triggered: exiting simulation")
        return res

    components.strategy.aggregate_evaluate = wrapped_evaluate
    return components

# Launch app
app = ServerApp(server_fn=server_fn)