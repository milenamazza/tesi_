import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Dict, Optional
import copy
import os
import json
import random
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import csv
import pandas as pd


# ========== CONFIG ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

user_path1 = '/scratch.hpc/mazza/femnist/femnist_lab/'
user_path2 = '/scratch.hpc/mazza/femnist/irds_lab/irds_dataset/'
ds = 'irds'  # 'femnist' or 'irds'
path = f"fed_kd_results_{ds}_3.csv"
num_classes = 4 if ds=='irds' else 10

def fix_random(seed: int) -> None:
    """Fix all random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class StudentModel(nn.Module):
    """Student model architecture - following FedKD literature standards"""
    def __init__(self, input_dim: int = 1280, num_classes: int = 10, hidden_dim: int = 256):
        super(StudentModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract intermediate features - useful for feature-based distillation"""
        return self.feature_extractor(x)

# ========== DATA LOADING (unchanged) ==========
def load_data(partition_id: int, num_partitions: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Load federated data partitions"""
    if ds == 'femnist':
        base_path = os.path.join(user_path1, "femnist_processed", f"client_{partition_id}")
        
        train_feat_path = os.path.join(base_path, "train_features.pt")
        train_label_path = os.path.join(base_path, "train_labels.pt")
        test_feat_path = os.path.join(base_path, "test_features.pt")
        test_label_path = os.path.join(base_path, "test_labels.pt")
        
        for path in [train_feat_path, train_label_path, test_feat_path, test_label_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
        
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

class SyntheticDataGenerator:
    """
    Synthetic data generator following FedKD literature
    Based on "Federated Learning with Only Positive Labels" and similar works
    """
    
    def __init__(self, input_dim: int, num_classes: int, num_clients: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.client_statistics = None
        self.global_statistics = None
        
    def collect_client_statistics(self) -> Dict:
        """Collect statistical moments from client data without sharing raw data"""
        print("📊 Collecting client statistics for synthetic data generation...")
        
        client_stats = []
        all_means = []
        all_stds = []
        total_samples = 0
        
        for client_id in range(self.num_clients):
            try:
                train_loader, _ = load_data(client_id, self.num_clients, batch_size=128)
                
                # Collect all features from this client
                client_features = []
                client_labels = []
                
                for batch_x, batch_y in train_loader:
                    client_features.append(batch_x)
                    client_labels.append(batch_y)
                
                if client_features:
                    features_tensor = torch.cat(client_features, dim=0)
                    labels_tensor = torch.cat(client_labels, dim=0)
                    
                    # Compute statistical moments
                    client_mean = torch.mean(features_tensor, dim=0)
                    client_std = torch.std(features_tensor, dim=0) + 1e-8  # Add small epsilon
                    client_min = torch.min(features_tensor, dim=0)[0]
                    client_max = torch.max(features_tensor, dim=0)[0]
                    
                    # Class-wise statistics
                    class_stats = {}
                    for c in range(self.num_classes):
                        class_mask = labels_tensor == c
                        if class_mask.sum() > 0:
                            class_features = features_tensor[class_mask]
                            class_stats[c] = {
                                'mean': torch.mean(class_features, dim=0),
                                'std': torch.std(class_features, dim=0) + 1e-8,
                                'count': class_mask.sum().item()
                            }
                    
                    client_stat = {
                        'id': client_id,
                        'mean': client_mean,
                        'std': client_std,
                        'min': client_min,
                        'max': client_max,
                        'class_stats': class_stats,
                        'total_samples': features_tensor.size(0)
                    }
                    
                    client_stats.append(client_stat)
                    all_means.append(client_mean)
                    all_stds.append(client_std)
                    total_samples += features_tensor.size(0)
                    
                    print(f"   Client {client_id}: {features_tensor.size(0)} samples")
                    
            except Exception as e:
                print(f"   Warning: Failed to load client {client_id}: {e}")
                continue
        
        # Compute global statistics
        if all_means:
            global_mean = torch.stack(all_means).mean(dim=0)
            global_std = torch.stack(all_stds).mean(dim=0)
            
            self.client_statistics = client_stats
            self.global_statistics = {
                'mean': global_mean,
                'std': global_std,
                'total_samples': total_samples
            }
            
            print(f"   Statistics collected from {len(client_stats)} clients")
            return {'client_stats': client_stats, 'global_stats': self.global_statistics}
        
        return None
    
    def generate_synthetic_dataset(self, num_samples: int) -> TensorDataset:
        """
        Generate synthetic dataset using collected statistics
        Following the approach from FedKD papers
        """
        if self.client_statistics is None:
            self.collect_client_statistics()
        
        if self.client_statistics is None:
            print("⚠️ Using basic Gaussian generation")
            return self._generate_basic_gaussian(num_samples)
        
        print(f"🎯 Generating {num_samples} synthetic samples")
        
        features_list = []
        labels_list = []
        
        # Strategy 1: Sample from global distribution (40%)
        global_samples = int(num_samples * 0.4)
        if global_samples > 0:
            global_features = torch.normal(
                self.global_statistics['mean'].unsqueeze(0).repeat(global_samples, 1),
                self.global_statistics['std'].unsqueeze(0).repeat(global_samples, 1)
            )
            features_list.append(global_features)
            global_labels = torch.randint(0, self.num_classes, (global_samples,))
            labels_list.append(global_labels)
        
        # Strategy 2: Sample from client-specific distributions (40%)
        client_samples = int(num_samples * 0.4)
        samples_per_client = max(1, client_samples // len(self.client_statistics))
        
        for client_stat in self.client_statistics:
            client_features = torch.normal(
                client_stat['mean'].unsqueeze(0).repeat(samples_per_client, 1),
                client_stat['std'].unsqueeze(0).repeat(samples_per_client, 1) * 0.8
            )
            features_list.append(client_features)
            
            # Sample labels based on client's class distribution
            class_weights = []
            for c in range(self.num_classes):
                if c in client_stat['class_stats']:
                    class_weights.append(client_stat['class_stats'][c]['count'])
                else:
                    class_weights.append(1)  # Small weight for unseen classes
            
            class_weights = np.array(class_weights, dtype=float)
            class_weights = class_weights / class_weights.sum()
            
            client_labels = torch.tensor(
                np.random.choice(self.num_classes, samples_per_client, p=class_weights)
            )
            labels_list.append(client_labels)
        
        # Strategy 3: Class-specific generation (20%)
        remaining_samples = num_samples - sum(f.size(0) for f in features_list)
        if remaining_samples > 0:
            samples_per_class = max(1, remaining_samples // self.num_classes)
            
            for class_idx in range(self.num_classes):
                # Find clients that have this class
                class_means = []
                class_stds = []
                
                for client_stat in self.client_statistics:
                    if class_idx in client_stat['class_stats']:
                        class_means.append(client_stat['class_stats'][class_idx]['mean'])
                        class_stds.append(client_stat['class_stats'][class_idx]['std'])
                
                if class_means:
                    # Average across clients for this class
                    avg_class_mean = torch.stack(class_means).mean(dim=0)
                    avg_class_std = torch.stack(class_stds).mean(dim=0)
                    
                    class_features = torch.normal(
                        avg_class_mean.unsqueeze(0).repeat(samples_per_class, 1),
                        avg_class_std.unsqueeze(0).repeat(samples_per_class, 1) * 0.7
                    )
                else:
                    # Fallback to global statistics
                    class_features = torch.normal(
                        self.global_statistics['mean'].unsqueeze(0).repeat(samples_per_class, 1),
                        self.global_statistics['std'].unsqueeze(0).repeat(samples_per_class, 1)
                    )
                
                features_list.append(class_features)
                class_labels = torch.full((samples_per_class,), class_idx)
                labels_list.append(class_labels)
        
        # Combine all synthetic data
        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        
        # Post-processing: ensure realistic ranges
        if self.client_statistics:
            # Compute overall min/max from all clients
            all_mins = torch.stack([cs['min'] for cs in self.client_statistics])
            all_maxs = torch.stack([cs['max'] for cs in self.client_statistics])
            global_min = torch.min(all_mins, dim=0)[0]
            global_max = torch.max(all_maxs, dim=0)[0]
            
            # Clamp to realistic ranges
            all_features = torch.clamp(all_features, global_min, global_max)
        
        print(f"   Generated {all_features.size(0)} synthetic samples")
        return TensorDataset(all_features, all_labels)
    
    def _generate_basic_gaussian(self, num_samples: int) -> TensorDataset:
        """Fallback Gaussian generation"""
        features = torch.randn(num_samples, self.input_dim) * 2.0
        labels = torch.randint(0, self.num_classes, (num_samples,))
        return TensorDataset(features, labels)

class FedKDServer:
    """
    Federated Knowledge Distillation Server
    Following FedKD algorithm from literature
    """
    
    def __init__(self, input_dim: int, num_classes: int, num_clients: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.global_model = StudentModel(input_dim, num_classes).to(device)
        self.synthetic_data_generator = SyntheticDataGenerator(input_dim, num_classes, num_clients)
        
        # Initialize global model weights
        self._initialize_global_model()
        
    def _initialize_global_model(self):
        """Initialize global model with Xavier initialization"""
        for m in self.global_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def aggregate_knowledge(self, client_outputs: List[torch.Tensor], 
                          client_weights: List[float],
                          aggregation_method: str = 'weighted_avg') -> torch.Tensor:
        """
        Aggregate client knowledge (soft predictions)
        Methods: 'weighted_avg', 'entropy_weighted', 'confidence_weighted'
        """
        if aggregation_method == 'weighted_avg':
            # Standard weighted averaging
            total_weight = sum(client_weights)
            normalized_weights = [w / total_weight for w in client_weights]
            
            aggregated = torch.zeros_like(client_outputs[0])
            for output, weight in zip(client_outputs, normalized_weights):
                aggregated += weight * output
                
        elif aggregation_method == 'entropy_weighted':
            # Weight by inverse entropy (more confident clients get higher weight)
            entropy_weights = []
            for output in client_outputs:
                # Compute average entropy
                entropy = -torch.sum(output * torch.log(output + 1e-8), dim=1)
                avg_entropy = torch.mean(entropy).item()
                # Inverse entropy as weight
                entropy_weights.append(1.0 / (1.0 + avg_entropy))
            
            # Combine with data size weights
            combined_weights = [ew * dw for ew, dw in zip(entropy_weights, client_weights)]
            total_weight = sum(combined_weights)
            normalized_weights = [w / total_weight for w in combined_weights]
            
            aggregated = torch.zeros_like(client_outputs[0])
            for output, weight in zip(client_outputs, normalized_weights):
                aggregated += weight * output
                
        elif aggregation_method == 'confidence_weighted':
            # Weight by prediction confidence (max probability)
            conf_weights = []
            for output in client_outputs:
                max_probs = torch.max(output, dim=1)[0]
                avg_confidence = torch.mean(max_probs).item()
                conf_weights.append(avg_confidence)
            
            # Combine with data size weights
            combined_weights = [cw * dw for cw, dw in zip(conf_weights, client_weights)]
            total_weight = sum(combined_weights)
            normalized_weights = [w / total_weight for w in combined_weights]
            
            aggregated = torch.zeros_like(client_outputs[0])
            for output, weight in zip(client_outputs, normalized_weights):
                aggregated += weight * output
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        return aggregated
    
    def distill_knowledge(self, teacher_outputs: torch.Tensor,
                         synthetic_dataset: TensorDataset,
                         temperature: float = 3.0,
                         alpha: float = 0.5,
                         distill_epochs: int = 10,
                         lr: float = 0.001) -> Dict:
        """
        Distill aggregated knowledge into global model
        Following standard KD formulation: L = α * L_KD + (1-α) * L_CE
        """
        print(f"🎓 Knowledge distillation: T={temperature}, α={alpha}, epochs={distill_epochs}")
        
        dataloader = DataLoader(synthetic_dataset, batch_size=64, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=distill_epochs)
        
        self.global_model.train()
        
        epoch_losses = {'total': [], 'kd': [], 'ce': []}
        
        for epoch in range(distill_epochs):
            total_loss = 0
            kd_loss_sum = 0
            ce_loss_sum = 0
            num_batches = 0
            
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                
                # Handle batch size mismatch
                batch_size = x.size(0)
                start_idx = (batch_idx * dataloader.batch_size) % teacher_outputs.size(0)
                end_idx = min(start_idx + batch_size, teacher_outputs.size(0))
                
                if start_idx >= end_idx:
                    continue
                    
                # Get teacher outputs for this batch
                teacher_batch = teacher_outputs[start_idx:end_idx].to(device)
                
                # Ensure matching sizes
                min_size = min(x.size(0), teacher_batch.size(0))
                x, y, teacher_batch = x[:min_size], y[:min_size], teacher_batch[:min_size]
                
                optimizer.zero_grad()
                
                # Student forward pass
                student_logits = self.global_model(x)
                
                # Knowledge Distillation Loss
                student_soft = F.log_softmax(student_logits / temperature, dim=1)
                teacher_soft = F.softmax(teacher_batch / temperature, dim=1)
                kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
                
                # Cross Entropy Loss
                ce_loss = F.cross_entropy(student_logits, y)
                
                # Combined Loss
                loss = alpha * kd_loss + (1 - alpha) * ce_loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                kd_loss_sum += kd_loss.item()
                ce_loss_sum += ce_loss.item()
                num_batches += 1
            
            scheduler.step()
            
            # Record epoch losses
            if num_batches > 0:
                epoch_losses['total'].append(total_loss / num_batches)
                epoch_losses['kd'].append(kd_loss_sum / num_batches)
                epoch_losses['ce'].append(ce_loss_sum / num_batches)
                
                if (epoch + 1) % max(1, distill_epochs // 3) == 0:
                    print(f"   Epoch {epoch+1}: Total={epoch_losses['total'][-1]:.4f}, "
                          f"KD={epoch_losses['kd'][-1]:.4f}, CE={epoch_losses['ce'][-1]:.4f}")
        
        return epoch_losses
    
    def get_global_model_copy(self) -> nn.Module:
        """Return a copy of the global model"""
        return copy.deepcopy(self.global_model)
    
    def evaluate_global_model(self, test_loaders: List[DataLoader]) -> Dict:
        """Evaluate global model on all clients"""
        self.global_model.eval()
        
        client_accuracies = []
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for client_id, test_loader in enumerate(test_loaders):
                correct = 0
                samples = 0
                
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = self.global_model(x)
                    _, predicted = torch.max(outputs, 1)
                    
                    correct += (predicted == y).sum().item()
                    samples += y.size(0)
                
                client_acc = correct / samples if samples > 0 else 0
                client_accuracies.append(client_acc)
                total_correct += correct
                total_samples += samples
        
        global_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'global_accuracy': global_accuracy,
            'client_accuracies': client_accuracies,
            'avg_client_accuracy': np.mean(client_accuracies),
            'std_client_accuracy': np.std(client_accuracies)
        }

class FedKDClient:
    """
    Federated Knowledge Distillation Client
    Following standard federated learning client operations
    """
    
    def __init__(self, client_id: int, input_dim: int, num_classes: int):
        self.client_id = client_id
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.local_model = StudentModel(input_dim, num_classes).to(device)
        
    def local_training(self, train_loader: DataLoader, 
                      global_model: nn.Module,
                      epochs: int = 5, 
                      lr: float = 0.01) -> Dict:
        """
        Local training phase
        Standard federated learning local update
        """
        print(f"🔧 Client {self.client_id}: Local training ({epochs} epochs)")
        
        # Copy global model parameters
        self.local_model.load_state_dict(global_model.state_dict())
        
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        
        self.local_model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                outputs = self.local_model(x)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            epoch_losses.append(avg_loss)
        
        return {
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'epoch_losses': epoch_losses
        }
    
    def generate_soft_predictions(self, synthetic_dataset: TensorDataset) -> torch.Tensor:
        """Generate soft predictions on synthetic dataset"""
        dataloader = DataLoader(synthetic_dataset, batch_size=64, shuffle=False)
        
        self.local_model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                logits = self.local_model(x)
                predictions = F.softmax(logits, dim=1)
                all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions, dim=0)
    
    def get_dataset_size(self, train_loader: DataLoader) -> int:
        """Get training dataset size"""
        return len(train_loader.dataset)

def run_federated_knowledge_distillation(num_clients: int = 7,
                                        num_rounds: int = 20,
                                        local_epochs: int = 5,
                                        distill_epochs: int = 10,
                                        base_lr: float = 0.01,
                                        temperature: float = 3.0,
                                        alpha: float = 0.5,
                                        aggregation_method: str = 'weighted_avg',
                                        synthetic_data_size: int = 1000,
                                        seed: int = 42,
                                        # Early stopping parameters
                                        early_stopping: bool = True,
                                        patience: int = 5,
                                        min_delta: float = 0.001,
                                        min_rounds: int = 10) -> Dict:
    """
    Federated Knowledge Distillation with Early Stopping
    
    Args:
        early_stopping: Whether to use early stopping
        patience: Number of rounds to wait without improvement
        min_delta: Minimum change to qualify as improvement
        min_rounds: Minimum number of rounds before early stopping can trigger
    """
    
    print("🚀 Federated Knowledge Distillation (Literature Standard)")
    print(f"Configuration: {num_clients} clients, {num_rounds} rounds")
    print(f"Local training: {local_epochs} epochs, Distillation: {distill_epochs} epochs")
    print(f"Temperature: {temperature}, Alpha: {alpha}, Synthetic data: {synthetic_data_size}")
    
    if early_stopping:
        print(f"🛑 Early Stopping: patience={patience}, min_delta={min_delta}, min_rounds={min_rounds}")
    
    # Fix random seeds for reproducibility
    fix_random(seed)
    
    # Initialize components
    input_dim = 1280  # MobileNet feature dimension
    server = FedKDServer(input_dim, num_classes, num_clients)
    clients = [FedKDClient(i, input_dim, num_classes) for i in range(num_clients)]
    
    # Load client datasets
    print("📂 Loading federated datasets...")
    client_train_loaders = []
    client_test_loaders = []
    
    for client_id in range(num_clients):
        try:
            train_loader, test_loader = load_data(client_id, num_clients, batch_size=32)
            client_train_loaders.append(train_loader)
            client_test_loaders.append(test_loader)
            print(f"   Client {client_id}: {len(train_loader.dataset)} training samples")
        except Exception as e:
            print(f"   Error loading client {client_id}: {e}")
            return {"error": f"Failed to load data for client {client_id}"}
    
    # Performance tracking
    performance_history = []
    
    # Early stopping variables
    best_global_accuracy = 0.0
    best_avg_client_accuracy = 0.0
    rounds_without_improvement = 0
    early_stopped = False
    stopped_at_round = num_rounds
    
    # Main Federated Learning Loop
    for round_num in range(num_rounds):
        print(f"\n🔄 ROUND {round_num + 1}/{num_rounds}")
        
        # Step 1: Local Training Phase
        print("📚 Phase 1: Local Training")
        local_models = []
        client_dataset_sizes = []
        
        for client_id, (client, train_loader) in enumerate(zip(clients, client_train_loaders)):
            # Local training with current global model
            local_stats = client.local_training(
                train_loader=train_loader,
                global_model=server.get_global_model_copy(),
                epochs=local_epochs,
                lr=base_lr * (0.99 ** round_num)  # Learning rate decay
            )
            
            # Store client information
            dataset_size = client.get_dataset_size(train_loader)
            client_dataset_sizes.append(dataset_size)
            
            print(f"   Client {client_id}: Local loss = {local_stats['final_loss']:.4f}, "
                  f"Dataset size = {dataset_size}")
        
        # Step 2: Synthetic Data Generation Phase
        print("🎯 Phase 2: Synthetic Data Generation")
        synthetic_dataset = server.synthetic_data_generator.generate_synthetic_dataset(
            num_samples=synthetic_data_size
        )
        
        # Step 3: Knowledge Extraction Phase  
        print("🧠 Phase 3: Knowledge Extraction")
        client_soft_predictions = []
        
        for client_id, client in enumerate(clients):
            # Generate soft predictions on synthetic data
            soft_preds = client.generate_soft_predictions(synthetic_dataset)
            client_soft_predictions.append(soft_preds)
            
            avg_confidence = torch.mean(torch.max(soft_preds, dim=1)[0]).item()
            avg_entropy = torch.mean(-torch.sum(soft_preds * torch.log(soft_preds + 1e-8), dim=1)).item()
            
            print(f"   Client {client_id}: Avg confidence = {avg_confidence:.3f}, "
                  f"Avg entropy = {avg_entropy:.3f}")
        
        # Step 4: Knowledge Aggregation Phase
        print("🔗 Phase 4: Knowledge Aggregation")
        aggregated_knowledge = server.aggregate_knowledge(
            client_outputs=client_soft_predictions,
            client_weights=client_dataset_sizes,
            aggregation_method=aggregation_method
        )
        
        # Compute aggregation statistics
        agg_confidence = torch.mean(torch.max(aggregated_knowledge, dim=1)[0]).item()
        agg_entropy = torch.mean(-torch.sum(aggregated_knowledge * torch.log(aggregated_knowledge + 1e-8), dim=1)).item()
        print(f"   Aggregated knowledge: Confidence = {agg_confidence:.3f}, Entropy = {agg_entropy:.3f}")
        
        # Step 5: Global Model Update via Knowledge Distillation
        print("🎓 Phase 5: Knowledge Distillation")
        distill_stats = server.distill_knowledge(
            teacher_outputs=aggregated_knowledge,
            synthetic_dataset=synthetic_dataset,
            temperature=temperature,
            alpha=alpha,
            distill_epochs=distill_epochs,
            lr=base_lr * 0.1  # Lower LR for distillation
        )
        
        # Step 6: Evaluation
        print("📊 Phase 6: Global Model Evaluation")
        eval_results = server.evaluate_global_model(client_test_loaders)
        
        # Store performance metrics
        round_metrics = {
            'round': round_num + 1,
            'global_accuracy': eval_results['global_accuracy'],
            'avg_client_accuracy': eval_results['avg_client_accuracy'],
            'std_client_accuracy': eval_results['std_client_accuracy'],
            'client_accuracies': eval_results['client_accuracies'],
            'distill_loss': distill_stats['total'][-1] if distill_stats['total'] else 0,
            'kd_loss': distill_stats['kd'][-1] if distill_stats['kd'] else 0,
            'ce_loss': distill_stats['ce'][-1] if distill_stats['ce'] else 0,
            'agg_confidence': agg_confidence,
            'agg_entropy': agg_entropy
        }
        performance_history.append(round_metrics)
        
        current_global_acc = eval_results['global_accuracy']
        current_avg_client_acc = eval_results['avg_client_accuracy']
        
        print(f"   📈 Global Accuracy: {current_global_acc:.4f}")
        print(f"   📈 Avg Client Accuracy: {current_avg_client_acc:.4f} "
              f"(±{eval_results['std_client_accuracy']:.4f})")
        print(f"   📈 Distillation Loss: {round_metrics['distill_loss']:.4f}")
        
        # Early Stopping Logic
        if early_stopping and round_num >= min_rounds - 1:  # -1 because round_num starts from 0
            # Check for improvement (using global accuracy as primary metric)
            improved = False
            
            if current_global_acc > best_global_accuracy + min_delta:
                best_global_accuracy = current_global_acc
                best_avg_client_accuracy = current_avg_client_acc
                rounds_without_improvement = 0
                improved = True
                print(f"   🎯 New best global accuracy: {best_global_accuracy:.4f}")
            else:
                rounds_without_improvement += 1
                print(f"   ⏳ No improvement for {rounds_without_improvement}/{patience} rounds")
            
            # Check if we should stop
            if rounds_without_improvement >= patience:
                print(f"   🛑 Early stopping triggered after {patience} rounds without improvement")
                print(f"   🏆 Best global accuracy: {best_global_accuracy:.4f}")
                early_stopped = True
                stopped_at_round = round_num + 1
                break
        else:
            # Always update best metrics in initial rounds
            if current_global_acc > best_global_accuracy:
                best_global_accuracy = current_global_acc
                best_avg_client_accuracy = current_avg_client_acc
    
    # Final Results Summary
    final_results = {
        'algorithm': 'FedKD',
        'dataset': ds,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'actual_rounds': stopped_at_round,
        'local_epochs': local_epochs,
        'distill_epochs': distill_epochs,
        'temperature': temperature,
        'alpha': alpha,
        'base_lr': base_lr,
        'synthetic_data_size': synthetic_data_size,
        'aggregation_method': aggregation_method,
        'early_stopping': early_stopping,
        'early_stopped': early_stopped,
        'patience': patience if early_stopping else None,
        'min_delta': min_delta if early_stopping else None,
        'min_rounds': min_rounds if early_stopping else None,
        'performance_history': performance_history,
        'final_global_accuracy': performance_history[-1]['global_accuracy'] if performance_history else 0,
        'final_avg_client_accuracy': performance_history[-1]['avg_client_accuracy'] if performance_history else 0,
        'final_std_client_accuracy': performance_history[-1]['std_client_accuracy'] if performance_history else 0,
        'best_global_accuracy': best_global_accuracy,
        'best_avg_client_accuracy': best_avg_client_accuracy
    }
    
    # Log results to CSV
    log_fedkd_results_to_csv(path, final_results)
    
    status_msg = "EARLY STOPPED" if early_stopped else "COMPLETED"
    print(f"\n🎯 FEDERATED KNOWLEDGE DISTILLATION {status_msg}")
    print(f"   Rounds: {stopped_at_round}/{num_rounds}")
    print(f"   Final Global Accuracy: {final_results['final_global_accuracy']:.4f}")
    print(f"   Best Global Accuracy: {best_global_accuracy:.4f}")
    print(f"   Final Avg Client Accuracy: {final_results['final_avg_client_accuracy']:.4f}")
    print(f"   Results saved to: {path}")
    
    return final_results

def log_fedkd_results_to_csv(filepath: str, results: Dict):
    """Log FedKD results to CSV file with early stopping info"""
    import csv
    
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode='a', newline='') as csv_file:
        fieldnames = [
            'algorithm', 'dataset', 'num_clients', 'num_rounds', 'actual_rounds',
            'local_epochs', 'distill_epochs', 'temperature', 'alpha',
            'base_lr', 'synthetic_data_size', 'aggregation_method',
            'early_stopping', 'early_stopped', 'patience', 'min_delta', 'min_rounds',
            'round', 'global_accuracy', 'avg_client_accuracy', 'std_client_accuracy',
            'distill_loss', 'kd_loss', 'ce_loss', 'agg_confidence', 'agg_entropy',
            'best_global_accuracy', 'best_avg_client_accuracy'
        ]
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Write results for each evaluation round
        for perf in results['performance_history']:
            row = {
                'algorithm': results['algorithm'],
                'dataset': results['dataset'],
                'num_clients': results['num_clients'],
                'num_rounds': results['num_rounds'],
                'actual_rounds': results['actual_rounds'],
                'local_epochs': results['local_epochs'],
                'distill_epochs': results['distill_epochs'],
                'temperature': results['temperature'],
                'alpha': results['alpha'],
                'base_lr': results['base_lr'],
                'synthetic_data_size': results['synthetic_data_size'],
                'aggregation_method': results['aggregation_method'],
                'early_stopping': results['early_stopping'],
                'early_stopped': results['early_stopped'],
                'patience': results['patience'],
                'min_delta': results['min_delta'],
                'min_rounds': results['min_rounds'],
                'round': perf['round'],
                'global_accuracy': perf['global_accuracy'],
                'avg_client_accuracy': perf['avg_client_accuracy'],
                'std_client_accuracy': perf['std_client_accuracy'],
                'distill_loss': perf['distill_loss'],
                'kd_loss': perf['kd_loss'],
                'ce_loss': perf['ce_loss'],
                'agg_confidence': perf['agg_confidence'],
                'agg_entropy': perf['agg_entropy'],
                'best_global_accuracy': results['best_global_accuracy'],
                'best_avg_client_accuracy': results['best_avg_client_accuracy']
            }
            writer.writerow(row)

# Aggiorna anche la funzione load_completed_configs per includere i nuovi parametri
def load_completed_configs(filepath: str) -> set:
    if not os.path.exists(filepath):
        return set()

    df = pd.read_csv(filepath)

    # Colonne richieste per identificare una configurazione unica
    required_cols = ['temperature', 'alpha', 'base_lr', 'local_epochs', 'distill_epochs', 'num_clients', 'num_rounds']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("CSV is missing one or more required columns.")

    # Ottieni solo le configurazioni completate (prendi l'ultimo round per ogni configurazione)
    completed_configs = set()
    
    # Raggruppa per configurazione e prendi solo quelle che hanno completato tutti i round o sono early stopped
    for config_group in df.groupby(['temperature', 'alpha', 'base_lr', 'local_epochs', 'distill_epochs', 'num_clients', 'num_rounds']):
        config_params = config_group[0]
        config_data = config_group[1]
        
        # Verifica se la configurazione è completata
        max_round = config_data['round'].max()
        expected_rounds = config_params[6]  # num_rounds
        
        # Se ha raggiunto il numero massimo di round O se è early stopped
        if max_round == expected_rounds or config_data['early_stopped'].any():
            completed_configs.add(config_params)
    
    return completed_configs

from itertools import product

# Main execution con early stopping
if __name__ == "__main__":
    temperatures = [1.0, 3.0, 5.0]
    alphas = [0.0, 0.3, 0.7]
    base_lrs = [0.001, 0.01, 0.1]
    local_epochs_list = [3, 5, 10]
    distill_epochs_list = [5, 10]
    num_clients_list = [3]

    num_rounds = 70
    
    # Early stopping parameters
    use_early_stopping = True
    patience = 15  # Wait 5 rounds without improvement
    min_delta = 0.00  # Minimum improvement of 0.1%
    min_rounds = 10  # Don't stop before round 10

    completed_configs = load_completed_configs(path)
    all_configs = list(product(temperatures, alphas, base_lrs, local_epochs_list, distill_epochs_list, num_clients_list))

    print(f"🔍 Total configurations: {len(all_configs)}")
    print(f"⏭️ Skipping {len(completed_configs)} already completed")

    for i, (T, alpha, lr, local_ep, distill_ep, num_clients) in enumerate(all_configs):
        config_key = (
            T, alpha, lr,
            local_ep, distill_ep,
            num_clients, num_rounds
        )

        if config_key in completed_configs:
            print(f"⏩ Skipping already completed config {config_key}")
            continue

        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(all_configs)}")
        print(f"T={T}, α={alpha}, LR={lr}, Local={local_ep}, Distill={distill_ep}, Clients={num_clients}, Rounds={num_rounds}")
        print(f"{'='*60}")

        results = run_federated_knowledge_distillation(
            num_clients=num_clients,
            num_rounds=num_rounds,
            local_epochs=local_ep,
            distill_epochs=distill_ep,
            temperature=T,
            alpha=alpha,
            base_lr=lr,
            synthetic_data_size=1000,
            aggregation_method='weighted_avg',
            seed=42,
            # Early stopping parameters
            early_stopping=use_early_stopping,
            patience=patience,
            min_delta=min_delta,
            min_rounds=min_rounds
        )

        if 'error' not in results:
            status = "EARLY STOPPED" if results.get('early_stopped', False) else "COMPLETED"
            rounds_info = f"{results.get('actual_rounds', num_rounds)}/{num_rounds}"
            print(f"✅ {status} — Rounds: {rounds_info} — Global Accuracy: {results['final_global_accuracy']:.4f} (Best: {results['best_global_accuracy']:.4f})")
        else:
            print(f"❌ Failed: {results['error']}")
