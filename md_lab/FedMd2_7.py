import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Dict, Optional
import copy
import random
import csv
from sklearn.cluster import KMeans
from datetime import datetime
import itertools

# ========== CONFIG ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

user_path1 = '/scratch.hpc/mazza/femnist/femnist_lab/'
user_path2 = '/scratch.hpc/mazza/femnist/irds_lab/irds_dataset/'
ds = 'irds'  # 'femnist' or 'irds'
num_classes = 4 if ds=='irds' else 10
ris_path = f"fedmd_results_all_{ds}_7.csv"

def fix_random(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ========== DIVERSIFIED MODEL ARCHITECTURES ==========
class MLPClassifier_Shallow(nn.Module):
    """Modello MLP poco profondo"""
    def __init__(self, input_dim=1280, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class MLPClassifier_Deep(nn.Module):
    """Modello MLP profondo"""
    def __init__(self, input_dim=1280, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class MLPClassifier_Wide(nn.Module):
    """Modello MLP largo"""
    def __init__(self, input_dim=1280, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class CNNClassifier(nn.Module):
    """Modello CNN adattato per features 1D"""
    def __init__(self, input_dim=1280, num_classes=10):
        super().__init__()
        # Reshape input per CNN 1D
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Calcola dimensione dopo conv layers
        conv_output_size = input_dim // 4 * 64
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Reshape per CNN: (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# ========== ADVANCED SYNTHETIC DATASET GENERATION ==========
class SyntheticDatasetGenerator:
    """Generatore avanzato di dataset sintetici per FedMD"""
    
    def __init__(self, input_dim: int, num_classes: int, num_clients: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.real_data_stats = self._collect_real_data_statistics()
    
    def _collect_real_data_statistics(self) -> Dict:
        """Raccoglie statistiche dai dati reali per generazione più realistica"""
        print("📊 Collecting real data statistics for synthetic generation...")
        
        all_features = []
        all_labels = []
        
        for client_id in range(min(self.num_clients, 3)):  # Limita per efficienza
            try:
                train_loader, _ = load_data(client_id, self.num_clients, batch_size=64)
                for batch_x, batch_y in train_loader:
                    all_features.append(batch_x)
                    all_labels.append(batch_y)
            except Exception as e:
                print(f"⚠️ Could not load data for client {client_id}: {e}")
                continue
        
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Statistiche globali
            global_mean = torch.mean(all_features, dim=0)
            global_std = torch.std(all_features, dim=0)
            
            # Statistiche per classe
            class_stats = {}
            for c in range(self.num_classes):
                mask = all_labels == c
                if mask.sum() > 0:
                    class_features = all_features[mask]
                    class_stats[c] = {
                        'mean': torch.mean(class_features, dim=0),
                        'std': torch.std(class_features, dim=0),
                        'count': mask.sum().item()
                    }
            
            return {
                'global_mean': global_mean,
                'global_std': global_std,
                'class_stats': class_stats,
                'feature_ranges': {
                    'min': torch.min(all_features, dim=0)[0],
                    'max': torch.max(all_features, dim=0)[0]
                }
            }
        
        # Fallback se non riusciamo a caricare dati reali
        return {
            'global_mean': torch.zeros(self.input_dim),
            'global_std': torch.ones(self.input_dim),
            'class_stats': {},
            'feature_ranges': {
                'min': torch.ones(self.input_dim) * -2,
                'max': torch.ones(self.input_dim) * 2
            }
        }
    
    def generate_gaussian_mixture_data(self, num_samples: int) -> TensorDataset:
        """Genera dati usando Gaussian Mixture Model basato sui dati reali"""
        print(f"🎲 Generating {num_samples} synthetic samples using GMM...")
        
        features_list = []
        labels_list = []
        
        samples_per_class = num_samples // self.num_classes
        
        for class_id in range(self.num_classes):
            if class_id in self.real_data_stats['class_stats']:
                # Usa statistiche della classe reale
                mean = self.real_data_stats['class_stats'][class_id]['mean']
                std = self.real_data_stats['class_stats'][class_id]['std']
                # Aggiungi un po' di rumore per diversificare
                std = std + torch.randn_like(std) * 0.1
                std = torch.clamp(std, min=1e-6)
            else:
                # Fallback a statistiche globali con offset per classe
                mean = self.real_data_stats['global_mean'] + torch.randn(self.input_dim) * 0.5
                std = self.real_data_stats['global_std'] * (0.8 + class_id * 0.1)
                std = torch.clamp(std, min=1e-6)

            # Genera campioni per questa classe
            class_features = torch.normal(
                mean.unsqueeze(0).repeat(samples_per_class, 1),
                std.unsqueeze(0).repeat(samples_per_class, 1)
            )
            
            # Clamp ai range osservati nei dati reali
            feature_min = self.real_data_stats['feature_ranges']['min']
            feature_max = self.real_data_stats['feature_ranges']['max']
            class_features = torch.clamp(class_features, feature_min, feature_max)
            
            features_list.append(class_features)
            labels_list.append(torch.full((samples_per_class,), class_id, dtype=torch.long))
        
        # Aggiungi campioni rimanenti
        remaining = num_samples - (samples_per_class * self.num_classes)
        if remaining > 0:
            # Distribuzione uniforme tra le classi
            for i in range(remaining):
                class_id = i % self.num_classes
                if class_id in self.real_data_stats['class_stats']:
                    mean = self.real_data_stats['class_stats'][class_id]['mean']
                    std = self.real_data_stats['class_stats'][class_id]['std']
                else:
                    mean = self.real_data_stats['global_mean']
                    std = self.real_data_stats['global_std']
                
                extra_sample = torch.normal(mean.unsqueeze(0), std.unsqueeze(0))
                features_list.append(extra_sample)
                labels_list.append(torch.tensor([class_id], dtype=torch.long))
        
        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        
        # Shuffle
        indices = torch.randperm(all_features.size(0))
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        
        # Normalizzazione finale
        all_features = F.normalize(all_features, p=2, dim=1)
        
        return TensorDataset(all_features, all_labels)

# ========== FEDMD IMPLEMENTATION ==========
class FedMDSystem:
    """
    Implementazione completa di FedMD (Li & Wang, 2019)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.input_dim = config['input_dim']
        self.num_classes = config['num_classes']
        self.num_clients = config['num_clients']
        
        # Inizializza generatore dataset sintetico
        self.dataset_generator = SyntheticDatasetGenerator(
            self.input_dim, self.num_classes, self.num_clients
        )
        
        # Inizializza client con modelli diversificati
        self.clients = self._initialize_clients()
        
        # Genera dataset pubblico
        self.public_dataset = self.dataset_generator.generate_gaussian_mixture_data(
            config['public_samples']
        )
        
        # Setup logging
        self.results_log = []
        self.setup_csv_logging()
    
    def _initialize_clients(self) -> List[Dict]:
        """Inizializza client con architetture diverse"""
        model_types = [
            #MLPClassifier_Shallow,
            #MLPClassifier_Deep, 
            #MLPClassifier_Wide,
            #CNNClassifier,
            MLPClassifier
        ]
        
        clients = []
        for i in range(self.num_clients):
            # Assegna tipo di modello in modo ciclico
            model_type = model_types[i % len(model_types)]
            model = model_type(self.input_dim, self.num_classes).to(device)
            
            clients.append({
                'id': i,
                'model': model,
                'model_type': model_type.__name__,
                'dataset_size': 0
            })
            
            print(f"🤖 Client {i}: {model_type.__name__}")
        
        return clients
    
    def setup_csv_logging(self):
        """Setup del logging CSV"""
        self.csv_filename = ris_path

        
        # Header del CSV
        fieldnames = [
            'experiment_id', 'round', 'lr', 'temperature', 'consensus_epochs', 
            'local_epochs', 'alpha', 'public_samples', 'batch_size', 'seed',
            'client_id', 'model_type', 'self_accuracy', 'avg_cross_accuracy',
            'loss_consensus', 'dataset_size'
        ]
        
        file_exists = os.path.isfile(self.csv_filename)

        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

        print(f"📝 Results will be logged to: {self.csv_filename}")
    
    def log_round_results(self, round_num: int, metrics: Dict):
        """Salva solo le metriche aggregate del round nel CSV"""
        with open(self.csv_filename, 'a', newline='') as csvfile:
            fieldnames = [
                'experiment_id', 'round', 'lr', 'temperature', 'consensus_epochs', 
                'local_epochs', 'alpha', 'public_samples', 'batch_size', 'seed',
                'avg_self_accuracy', 'avg_cross_accuracy', 'avg_loss_consensus'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            avg_self = np.mean([client['self_accuracy'] for client in metrics['clients']])
            avg_cross = np.mean([client['avg_cross_accuracy'] for client in metrics['clients']])
            avg_loss = np.mean([client['loss_consensus'] for client in metrics['clients']])

            row = {
                'experiment_id': self.config.get('experiment_id', 'default'),
                'round': round_num,
                'lr': self.config['lr'],
                'temperature': self.config['temperature'],
                'consensus_epochs': self.config['consensus_epochs'],
                'local_epochs': self.config['local_epochs'],
                'alpha': self.config.get('alpha', 0.5),
                'public_samples': self.config['public_samples'],
                'batch_size': self.config['batch_size'],
                'seed': self.config.get('seed', 42),
                'avg_self_accuracy': avg_self,
                'avg_cross_accuracy': avg_cross,
                'avg_loss_consensus': avg_loss
            }

            writer.writerow(row)

    
    def local_training_phase(self) -> None:
        """Fase di training locale per tutti i client"""
        print("🏠 Local Training Phase")
        
        for client in self.clients:
            print(f"   Training Client {client['id']} ({client['model_type']})")
            
            # Carica dati del client  
            train_loader, _ = load_data(
                client['id'], self.num_clients, 
                batch_size=self.config['batch_size']
            )
            client['dataset_size'] = len(train_loader.dataset)
            
            # Training locale
            self._train_local_model(
                client['model'], train_loader, 
                epochs=self.config['local_epochs'],
                lr=self.config['lr']
            )
    
    def _train_local_model(self, model: nn.Module, train_loader: DataLoader, 
                          epochs: int, lr: float) -> float:
        """Training di un singolo modello locale"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(train_loader)
        
        return total_loss / epochs
    
    def consensus_phase(self) -> torch.Tensor:
        """Fase di consensus: calcolo delle predizioni consenso"""
        print("🤝 Consensus Phase")
        
        public_loader = DataLoader(
            self.public_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # Raccogli predizioni da tutti i client
        all_predictions = []
        weights = []
        
        for client in self.clients:
            print(f"   Getting predictions from Client {client['id']}")
            
            preds = self._get_soft_predictions(
                client['model'], public_loader, 
                temperature=self.config['temperature']
            )
            all_predictions.append(preds)
            weights.append(client['dataset_size'])
        
        # Calcola consensus (media pesata)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        consensus_preds = torch.zeros_like(all_predictions[0])
        for preds, weight in zip(all_predictions, normalized_weights):
            consensus_preds += weight * preds
        
        print(f"   📊 Consensus shape: {consensus_preds.shape}")
        return consensus_preds
    
    def _get_soft_predictions(self, model: nn.Module, dataloader: DataLoader, 
                             temperature: float) -> torch.Tensor:
        """Ottieni predizioni soft da un modello"""
        model.eval()
        all_preds = []
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                logits = model(x)
                soft_preds = F.softmax(logits / temperature, dim=1)
                all_preds.append(soft_preds.cpu())
        
        return torch.cat(all_preds, dim=0)
    
    def distillation_phase(self, consensus_preds: torch.Tensor) -> Dict:
        """Fase di distillazione: align verso consensus"""
        print("🎓 Distillation Phase")
        
        distillation_losses = {}
        
        for client in self.clients:
            print(f"   Distilling Client {client['id']} towards consensus")
            
            loss = self._distill_towards_consensus(
                client['model'], self.public_dataset, consensus_preds,
                epochs=self.config['consensus_epochs'],
                lr=self.config['lr'],
                temperature=self.config['temperature']
            )
            distillation_losses[client['id']] = loss
        
        return distillation_losses
    
    def _distill_towards_consensus(self, model: nn.Module, public_dataset: TensorDataset,
                                consensus_preds: torch.Tensor, epochs: int, 
                                lr: float, temperature: float, alpha: float = None) -> float:
        """Distilla un modello verso le predizioni consensus usando soft + hard labels"""
        if alpha is None:
            alpha = self.config.get('alpha', 0.5)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        dataloader = DataLoader(public_dataset, batch_size=self.config['batch_size'], shuffle=False)

        model.train()
        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)

                # Ottieni predizioni consensus per questo batch
                batch_start = batch_idx * dataloader.batch_size
                batch_end = min(batch_start + x.size(0), consensus_preds.size(0))

                if batch_start >= consensus_preds.size(0):
                    break

                target_preds = consensus_preds[batch_start:batch_end].to(device)

                optimizer.zero_grad()
                outputs = model(x)

                # Soft targets: KLDivLoss tra log softmax dello student e softmax consensus
                student_soft = F.log_softmax(outputs / temperature, dim=1)
                soft_loss = F.kl_div(student_soft, target_preds, reduction='batchmean')

                # Hard targets: CrossEntropyLoss sulle etichette sintetiche
                hard_loss = F.cross_entropy(outputs, y)

                # Loss combinata con bilanciamento e T^2
                loss = alpha * hard_loss + (1 - alpha) * (temperature ** 2) * soft_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            total_loss += epoch_loss / len(dataloader)

        return total_loss / epochs


    
    def evaluation_phase(self, round_num: int, distillation_losses: Dict) -> Dict:
        """Fase di valutazione completa"""
        print("📊 Evaluation Phase")
        
        # Matrice di accuracy cross-client
        acc_matrix = np.zeros((self.num_clients, self.num_clients))
        
        for i, client_i in enumerate(self.clients):
            for j in range(self.num_clients):
                _, test_loader = load_data(j, self.num_clients, batch_size=self.config['batch_size'])
                acc = self._evaluate_model(client_i['model'], test_loader)
                acc_matrix[i, j] = acc
        
        # Calcola metriche
        client_metrics = []
        for i, client in enumerate(self.clients):
            self_acc = acc_matrix[i, i]
            cross_accs = [acc_matrix[i, j] for j in range(self.num_clients) if j != i]
            avg_cross_acc = np.mean(cross_accs) if cross_accs else 0.0
            
            client_metrics.append({
                'client_id': client['id'],
                'model_type': client['model_type'],
                'self_accuracy': self_acc,
                'avg_cross_accuracy': avg_cross_acc,
                'loss_consensus': distillation_losses.get(client['id'], 0.0),
                'dataset_size': client['dataset_size']
            })
        
        # Stampa risultati in formato tabellare
        print("\n🔍 Cross-Client Accuracy Matrix:")
        header = "         " + "  ".join([f"C{j}" for j in range(self.num_clients)])
        print(header)
        for i in range(self.num_clients):
            row = f"Model C{i} " + "  ".join([f"{acc_matrix[i,j]*100:5.1f}%" for j in range(self.num_clients)])
            print(row)
        
        # Metriche aggregate
        avg_self = np.mean([acc_matrix[i, i] for i in range(self.num_clients)])
        avg_cross = np.mean([acc_matrix[i, j] for i in range(self.num_clients) for j in range(self.num_clients) if i != j])
        
        print(f"\n📈 Round {round_num} Summary:")
        print(f"   Avg self accuracy:   {avg_self*100:.2f}%")
        print(f"   Avg cross accuracy:  {avg_cross*100:.2f}%")
        
        metrics = {
            'round': round_num,
            'clients': client_metrics,
            'avg_self_accuracy': avg_self,
            'avg_cross_accuracy': avg_cross,
            'accuracy_matrix': acc_matrix.tolist()
        }
        
        return metrics
    
    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> float:
        """Valuta accuracy di un modello"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def run_federated_training(self) -> Dict:
        """Esegue l'intero processo FedMD con early stopping"""
        print(f"🚀 Starting FedMD Training")
        print(f"Config: {self.config}")
        
        all_results = []
        best_accuracy = -float('inf')
        patience = self.config.get('early_stopping_patience', 5)
        no_improve_counter = 0

        for round_num in range(self.config['rounds']):
            print(f"\n{'='*60}")
            print(f"🔄 Round {round_num + 1}/{self.config['rounds']}")
            print(f"{'='*60}")
            
            # FASE 1: Training locale
            self.local_training_phase()
            
            # FASE 2: Consensus
            consensus_preds = self.consensus_phase()
            
            # FASE 3: Distillazione 
            distillation_losses = self.distillation_phase(consensus_preds)
            
            # FASE 4: Valutazione
            if round_num % self.config.get('eval_frequency', 1) == 0:
                metrics = self.evaluation_phase(round_num + 1, distillation_losses)
                all_results.append(metrics)
                
                # Log nel CSV
                self.log_round_results(round_num + 1, metrics)

                current_accuracy = metrics['avg_self_accuracy']
                if current_accuracy > best_accuracy + 1e-4:
                    best_accuracy = current_accuracy
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                    print(f"⏳ No improvement for {no_improve_counter} round(s)")

                if no_improve_counter >= patience:
                    print(f"🛑 Early stopping triggered at round {round_num + 1}")
                    break

        return {
            'config': self.config,
            'results': all_results,
            'final_models': {i: client['model'] for i, client in enumerate(self.clients)}
        }

# ========== HYPERPARAMETER GRID SEARCH ==========
class FedMDHyperparameterSearch:
    """Sistema per grid search degli iperparametri FedMD"""
    
    def __init__(self):
        self.base_config = {
            'input_dim': 1280,
            'num_classes': num_classes,
            'num_clients': 7,
            'rounds': 40,
            'batch_size': 128,
            'eval_frequency': 1,
            'seed': 42,
            'early_stopping_patience': 10  # oppure altro numero
        }
        
        # Griglia iperparametri
        self.param_grid = {
            'lr': [0.001, 0.0006, 0.0001],
            'temperature': [3.0, 5.0],
            'consensus_epochs': [3, 5, 10],
            'local_epochs': [3, 5],
            'public_samples': [200],
            'alpha': [0.0, 0.3, 0.5, 0.7]

        }
    
    def get_all_configurations(self) -> List[Dict]:
        """Genera tutte le combinazioni di iperparametri"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        configurations = []
        for i, combination in enumerate(itertools.product(*param_values)):
            config = self.base_config.copy()
            config['experiment_id'] = f"exp_{i:03d}"
            
            for param_name, param_value in zip(param_names, combination):
                config[param_name] = param_value
            
            configurations.append(config)
        
        return configurations
    
    def run_grid_search(self, max_experiments: Optional[int] = None) -> List[Dict]:
        """Esegue grid search completa"""
        configurations = self.get_all_configurations()
        
        if max_experiments:
            configurations = configurations[:max_experiments]
        
        print(f"🔬 Starting Grid Search with {len(configurations)} configurations")
        
        all_results = []
        
        for i, config in enumerate(configurations):
            print(f"\n{'🧪 EXPERIMENT'} {i+1}/{len(configurations)}")
            print(f"Config: {config}")
            
            # Fix seed per reproducibilità
            fix_random(config['seed'])
            
            try:
                # Esegui esperimento
                fedmd_system = FedMDSystem(config)
                results = fedmd_system.run_federated_training()
                all_results.append(results)
                
                print(f"✅ Experiment {i+1} completed successfully")
                
            except Exception as e:
                print(f"❌ Experiment {i+1} failed: {e}")
                continue
        
        return all_results
    
    def run_single_experiment(self, custom_config: Optional[Dict] = None) -> Dict:
        """Esegue un singolo esperimento con configurazione custom"""
        if custom_config:
            config = {**self.base_config, **custom_config}
        else:
            # Configurazione di default ottimizzata
            config = {
                **self.base_config,
                'lr': 0.001,
                'temperature': 3.0,
                'consensus_epochs': 5,
                'local_epochs': 3,
                'public_samples': 500,
                'experiment_id': 'single_run'
            }
        
        fix_random(config['seed'])
        
        fedmd_system = FedMDSystem(config)
        return fedmd_system.run_federated_training()

# ========== DATA LOADER (dal tuo codice originale) ==========
def load_data(partition_id: int, num_partitions: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    if ds == 'femnist':
        base_path = os.path.join(user_path1, "femnist_processed", f"client_{partition_id}")
        
        train_feat_path = os.path.join(base_path, "train_features.pt")
        train_label_path = os.path.join(base_path, "train_labels.pt")
        test_feat_path = os.path.join(base_path, "test_features.pt")
        test_label_path = os.path.join(base_path, "test_labels.pt")
        
        for path in [train_feat_path, train_label_path, test_feat_path, test_label_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File mancante: {path}")
        
        train_features = torch.load(train_feat_path)
        train_labels = torch.load(train_label_path)
        test_features = torch.load(test_feat_path)
        test_labels = torch.load(test_label_path)
        
        train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    else:  # irds dataset
        map_dirs = {
            '0': '101', '1': '102', '2': '103', 
            '3': '104', '4': '105', '5': '106', '6': '107'
        }

        path = f'{user_path2}/{map_dirs[str(partition_id)]}'

        train_dir_len = len(os.listdir(f'{path}/train'))
        validation_dir_len = len(os.listdir(f'{path}/validation'))
        test_dir_len = len(os.listdir(f'{path}/test'))
        
        train_map_file = json.load(open(f'{path}/quantized_train_files_map.json'))
        validation_map_file = json.load(open(f'{path}/quantized_validation_files_map.json'))
        test_map_file = json.load(open(f'{path}/quantized_test_files_map.json'))

        # Load training data
        train_labels = np.array([])
        for i in range(train_dir_len):
            train_data_tensor = torch.load(f'{path}/train/{i}_{map_dirs[str(partition_id)]}_train_quantized_features.pt')
            train_labels = np.append(train_labels, int(train_map_file[f'{i}_{map_dirs[str(partition_id)]}_train_quantized_features.pt']))
            if i == 0:
                train_data = train_data_tensor
            else:
                train_data = torch.cat((train_data, train_data_tensor), 0)
        
        # Load validation data
        validation_labels = np.array([])
        for i in range(validation_dir_len):
            validation_data_tensor = torch.load(f'{path}/validation/{i}_{map_dirs[str(partition_id)]}_validation_quantized_features.pt')
            validation_labels = np.append(validation_labels, int(validation_map_file[f'{i}_{map_dirs[str(partition_id)]}_validation_quantized_features.pt']))
            if i == 0:
                validation_data = validation_data_tensor
            else:
                validation_data = torch.cat((validation_data, validation_data_tensor), 0)
        
        # Load test data
        test_labels = np.array([])
        for i in range(test_dir_len):
            test_data_tensor = torch.load(f'{path}/test/{i}_{map_dirs[str(partition_id)]}_test_quantized_features.pt')
            test_labels = np.append(test_labels, int(test_map_file[f'{i}_{map_dirs[str(partition_id)]}_test_quantized_features.pt']))
            if i == 0:
                test_data = test_data_tensor
            else:
                test_data = torch.cat((test_data, test_data_tensor), 0)
        
        # Combine train and validation for training
        combined_train_data = torch.cat((train_data, validation_data), 0)
        combined_train_labels = np.append(train_labels, validation_labels)
        
        train_loader = DataLoader(
            TensorDataset(combined_train_data, torch.tensor(combined_train_labels, dtype=torch.long)),
            batch_size=batch_size, shuffle=True
        )
        
        test_loader = DataLoader(
            TensorDataset(test_data, torch.tensor(test_labels, dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader


#!/usr/bin/env python3
"""
FedMD Main Runner - Script principale per eseguire esperimenti FedMD
"""

import argparse
import json
import os
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def run_single_experiment(custom_config: Optional[Dict] = None) -> Dict:
    """
    Esegue un singolo esperimento FedMD
    
    Args:
        custom_config: Configurazione custom, se None usa quella di default
        
    Returns:
        Risultati dell'esperimento
    """
    print("🚀 Running Single FedMD Experiment")
    
    search_system = FedMDHyperparameterSearch()
    results = search_system.run_single_experiment(custom_config)
    
    print("✅ Single experiment completed!")
    return results

def run_grid_search_runner(max_experiments: Optional[int] = None) -> List[Dict]:
    """Esegue grid search completa evitando esperimenti già eseguiti"""
    search_system = FedMDHyperparameterSearch()
    configurations = search_system.get_all_configurations()

    # Carica ID esperimenti già eseguiti dal CSV
    existing_ids = set()
    if os.path.exists(ris_path):
        with open(ris_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row['experiment_id'])

    # Filtra configurazioni
    filtered_configurations = [conf for conf in configurations if conf['experiment_id'] not in existing_ids]

    if max_experiments:
        filtered_configurations = filtered_configurations[:max_experiments]

    print(f"🔬 Starting Grid Search with {len(filtered_configurations)} new configurations (skipping {len(configurations) - len(filtered_configurations)} already run)")

    all_results = []
    for i, config in enumerate(filtered_configurations):
        print(f"\n🧪 EXPERIMENT {i+1}/{len(filtered_configurations)} — {config['experiment_id']}")
        print(f"Config: {config}")

        fix_random(config['seed'])

        try:
            fedmd_system = FedMDSystem(config)
            results = fedmd_system.run_federated_training()
            all_results.append(results)
            print(f"✅ Experiment {config['experiment_id']} completed")
        except Exception as e:
            print(f"❌ Experiment {config['experiment_id']} failed: {e}")
            continue

    return all_results

def run_quick_test() -> Dict:
    """
    Esegue un test rapido con configurazione minima per debug
    """
    print("⚡ Running Quick Test")
    
    quick_config = {
        'rounds': 3,
        'consensus_epochs': 2,
        'local_epochs': 2,
        'public_samples': 100,
        'lr': 0.001,
        'temperature': 3.0,
        'experiment_id': 'quick_test'
    }
    
    return run_single_experiment(quick_config)

def run_optimized_experiment() -> Dict:
    """
    Esegue esperimento con configurazione ottimizzata (basata su ricerca precedente)
    """
    print("🎯 Running Optimized Experiment")
    
    optimized_config = {
        'rounds': 20,
        'lr': 0.001,
        'temperature': 3.0,
        'consensus_epochs': 5,
        'local_epochs': 3,
        'public_samples': 500,
        'batch_size': 32,
        'experiment_id': 'optimized_run'
    }
    
    return run_single_experiment(optimized_config)

def run_mini_grid_search() -> List[Dict]:
    """
    Esegue una grid search ridotta per test rapidi
    """
    print("🔬 Running Mini Grid Search")
    
    mini_grid = {
        'lr': [0.0005],
        'temperature': [1.0, 3.0],
        'consensus_epochs': [3, 5],
        'local_epochs': [2, 3],
        'public_samples': [200, 500]
    }
    
    return run_grid_search(max_experiments=8, custom_grid=mini_grid)

def analyze_results(csv_file: str) -> None:
    """
    Analizza i risultati da file CSV e genera visualizzazioni
    
    Args:
        csv_file: Path al file CSV con i risultati
    """
    print(f"📊 Analyzing results from {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found!")
        return
    
    # Carica dati
    df = pd.read_csv(csv_file)
    
    # Statistiche base
    print("\n📈 Basic Statistics:")
    print(f"Total experiments: {df['experiment_id'].nunique()}")
    print(f"Total rounds: {df['round'].nunique()}")
    print(f"Avg self accuracy: {df['self_accuracy'].mean()*100:.2f}%")
    print(f"Avg cross accuracy: {df['avg_cross_accuracy'].mean()*100:.2f}%")
    
    # Migliori configurazioni
    print("\n🏆 Top 5 Configurations (by avg cross accuracy):")
    best_configs = df.groupby('experiment_id').agg({
        'avg_cross_accuracy': 'mean',
        'self_accuracy': 'mean',
        'lr': 'first',
        'temperature': 'first',
        'consensus_epochs': 'first',
        'local_epochs': 'first',
        'public_samples': 'first'
    }).sort_values('avg_cross_accuracy', ascending=False).head()
    
    print(best_configs)
    
    # Salva report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"fedmd_analysis_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("FedMD Experiment Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source File: {csv_file}\n\n")
        f.write("Basic Statistics:\n")
        f.write(f"Total experiments: {df['experiment_id'].nunique()}\n")
        f.write(f"Total rounds: {df['round'].nunique()}\n")
        f.write(f"Avg self accuracy: {df['self_accuracy'].mean()*100:.2f}%\n")
        f.write(f"Avg cross accuracy: {df['avg_cross_accuracy'].mean()*100:.2f}%\n\n")
        f.write("Top 5 Configurations:\n")
        f.write(best_configs.to_string())
    
    print(f"📝 Analysis report saved to: {report_file}")

def main():
    """Funzione main con CLI per diversi tipi di esecuzione"""
    parser = argparse.ArgumentParser(description='FedMD Experiment Runner')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'grid', 'quick', 'optimized', 'mini_grid', 'analyze'],
                       help='Modalità di esecuzione')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='Numero massimo di esperimenti per grid search')
    parser.add_argument('--csv_file', type=str, default=None,
                       help='File CSV da analizzare (solo per mode=analyze)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='File JSON con configurazione custom')
    
    args = parser.parse_args()
    
    # Carica configurazione custom se fornita
    custom_config = None
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
        print(f"📖 Loaded custom config from {args.config_file}")
    
    # Esegui in base alla modalità
    if args.mode == 'single':
        results = run_single_experiment(custom_config)
        print("🎉 Single experiment completed!")
        
    elif args.mode == 'grid':
        results = run_grid_search_runner(args.max_experiments)
        print("🎉 Grid search completed!")
        
    elif args.mode == 'quick':
        results = run_quick_test()
        print("🎉 Quick test completed!")
        
    elif args.mode == 'optimized':
        results = run_optimized_experiment()
        print("🎉 Optimized experiment completed!")
        
    elif args.mode == 'mini_grid':
        results = run_mini_grid_search()
        print("🎉 Mini grid search completed!")
        
    elif args.mode == 'analyze':
        if not args.csv_file:
            # Trova il file CSV più recente
            csv_files = [f for f in os.listdir('.') if f.startswith('fedmd_results_') and f.endswith('.csv')]
            if csv_files:
                args.csv_file = max(csv_files, key=os.path.getctime)
                print(f"📁 Using most recent CSV file: {args.csv_file}")
            else:
                print("❌ No CSV file found! Please specify --csv_file")
                return
        
        analyze_results(args.csv_file)
        print("🎉 Analysis completed!")
    
    print("\n✨ All done! Check the generated files for results.")

# Funzioni di convenienza per uso programmatico
def quick_run():
    """Esecuzione rapida per test"""
    return run_quick_test()

def full_grid_run(max_exp: int = None):
    """Esecuzione grid search completa"""
    return run_grid_search(max_exp)

def optimized_run():
    """Esecuzione con configurazione ottimizzata"""
    return run_optimized_experiment()

if __name__ == "__main__":
    main()