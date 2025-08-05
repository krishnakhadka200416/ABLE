"""
Model definitions and training functionality for ABLE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

# ART (Adversarial Robustness Toolbox)
from art.estimators.classification import PyTorchClassifier

# TabNet and TabTransformer
from pytorch_tabnet.tab_model import TabNetClassifier as PytorchTabNetClassifier
from tab_transformer_pytorch import TabTransformer

###############################################################################
# MODEL DEFINITIONS
###############################################################################
class MLP(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(input_size)
        
    def forward(self, x):
        x = self.bn1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class CustomTabNetClassifier(PyTorchClassifier):
    def __init__(self, model, clip_values, device_type, input_dim, num_classes=2):
        class TabNetWrapper(nn.Module):
            def __init__(self, tabnet_network):
                super(TabNetWrapper, self).__init__()
                self.tabnet = tabnet_network
            
            def forward(self, x):
                out, _ = self.tabnet(x)
                return out

        wrapped_model = TabNetWrapper(model.network.to(device_type))

        super(CustomTabNetClassifier, self).__init__(
            model=wrapped_model,
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(wrapped_model.parameters(), lr=0.001),
            input_shape=(input_dim,),
            nb_classes=num_classes,
            clip_values=clip_values,
            device_type=device_type
        )
        self.tabnet_model = model

    def predict(self, x, **kwargs):
        x_t = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            probs = self.tabnet_model.predict_proba(x_t)
        return probs

class TabTransformerWrapper(nn.Module):
    def __init__(self, num_continuous, num_classes=2, device='cpu'):
        super().__init__()
        self.transformer = TabTransformer(
            categories=[1],  # Add a dummy categorical feature
            num_continuous=num_continuous,
            dim=32,
            dim_out=num_classes,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2)
        ).to(device)
        
    def forward(self, x):
        # Create a dummy categorical tensor with zeros
        batch_size = x.shape[0]
        x_categ = torch.zeros((batch_size, 1), dtype=torch.long, device=x.device)
        return self.transformer(x_categ=x_categ, x_cont=x)

class CustomTabTransformerClassifier(PyTorchClassifier):
    def __init__(self, model, loss, optimizer, input_shape, nb_classes, clip_values, device_type):
        super(CustomTabTransformerClassifier, self).__init__(
            model=model,
            loss=loss,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values,
            device_type=device_type
        )
    
    def predict(self, x, **kwargs):
        x_t = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            out = self.model(x_t)
            out = out.cpu().numpy()
        return out

###############################################################################
# MODEL TRAINING
###############################################################################
def create_and_train_model(model_name, X_train, y_train, X_test, y_test, input_dim, num_classes, device):
    """Create and train a specified model."""
    
    if model_name.upper() == 'MLP':
        model = MLP(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(15):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # Test accuracy
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        accuracy = accuracy_score(y_test, predictions)
        
        # Wrap with ART
        classifier = PyTorchClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(input_dim,),
            nb_classes=num_classes,
            clip_values=(X_train.min(), X_train.max()),
            device_type=device.type
        )
        
    elif model_name.upper() == 'TABNET':
        tabnet_model = PytorchTabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            lambda_sparse=1e-3, momentum=0.7, clip_value=2.,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            device_name=device.type
        )
        
        tabnet_model.fit(X_train, y_train, max_epochs=15, batch_size=64, virtual_batch_size=128)
        
        # Test accuracy
        predictions = tabnet_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        classifier = CustomTabNetClassifier(
            model=tabnet_model,
            clip_values=(X_train.min(), X_train.max()),
            device_type=device.type,
            input_dim=input_dim,
            num_classes=num_classes
        )
        
    elif model_name.upper() == 'TABTRANSFORMER':
        model = TabTransformerWrapper(input_dim, num_classes, device).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(15):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # Test accuracy
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        accuracy = accuracy_score(y_test, predictions)
        
        classifier = CustomTabTransformerClassifier(
            model=model,
            loss=criterion,
            optimizer=optimizer,
            input_shape=(input_dim,),
            nb_classes=num_classes,
            clip_values=(X_train.min(), X_train.max()),
            device_type=device.type
        )
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return classifier 