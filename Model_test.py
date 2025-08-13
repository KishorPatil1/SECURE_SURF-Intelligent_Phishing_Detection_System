import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
# Function to clean data
def clean_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df

# Load datasets
phishing_data = pd.read_csv('original_new_phish_25k.csv', dtype=str, low_memory=False)
legitimate_data = pd.read_csv('legit_data.csv', dtype=str, low_memory=False)
phishing_data['Label'] = 1
legitimate_data['Label'] = 0

# Combine and clean
dataset = pd.concat([phishing_data, legitimate_data])
dataset = dataset.drop(['url', 'NonStdPort', 'GoogleIndex', 'double_slash_redirecting', 'https_token'], axis=1)
dataset = clean_data(dataset)

X = dataset.drop('Label', axis=1)
y = dataset['Label'].astype(int)
# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

input_dim = X_train.shape[1]
# All model classes

class BasicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class DropoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class BatchNormMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x))

class TanhMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 128)
        self.fc_res1 = nn.Linear(128, 128)
        self.fc_res2 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.relu(self.fc_in(x))
        res = self.fc_res1(x1)
        res = self.fc_res2(res)
        x2 = self.relu(x1 + res)
        return self.sigmoid(self.out(x2))

# class CNNModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.input_size = int(np.sqrt(input_dim))
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * (self.input_size//2)**2, 128)
#         self.fc2 = nn.Linear(128, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = x.view(-1, 1, self.input_size, self.input_size)
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64 * (self.input_size//2)**2)
#         x = self.relu(self.fc1(x))
#         return self.sigmoid(self.fc2(x))
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions and pooling
        # After first conv: input_dim
        # After first pool: input_dim/2
        # After second conv: input_dim/2
        # After second pool: input_dim/4
        self.fc1_input_size = 64 * (input_dim // 4)
        
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input to (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))
# Model dictionary
deep_classifiers = {
    'CNN Model': CNNModel,
    'Basic MLP': BasicMLP,
    'Dropout MLP': DropoutMLP,
    'Deep MLP': DeepMLP,
    'BatchNorm MLP': BatchNormMLP,
    'Tanh MLP': TanhMLP,
    'Residual MLP': ResidualMLP,
    
}

# Train function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

# Evaluate function
def evaluate_model(model, test_loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.append(outputs.cpu().numpy())
            true_labels.append(labels.numpy())
    preds = np.vstack(preds)
    true_labels = np.vstack(true_labels)
    pred_classes = (preds > 0.5).astype(int)
    return accuracy_score(true_labels, pred_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []
best_overall_acc = 0
best_model_name = None
best_weights = None

for name, ModelClass in deep_classifiers.items():
    print(f"Training {name}...")
    model = ModelClass().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    patience = 3
    patience_counter = 0

    for epoch in range(20):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_model_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_model_weights)
    results.append((name, best_acc))
    
    if best_acc > best_overall_acc:
        best_overall_acc = best_acc
        best_model_name = name
        best_weights = best_model_weights
# DataFrame
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy'])
print(results_df)

# Bonferroni-Dunn
try:
    dunn_results = sp.posthoc_dunn(results_df, val_col='Accuracy', group_col='Classifier', p_adjust='bonferroni')
    print(dunn_results)
except Exception as e:
    print("Bonferroni-Dunn test failed:", e)

# Save best model
final_model = deep_classifiers[best_model_name]().to(device)
final_model.load_state_dict(best_weights)
joblib.dump(final_model.state_dict(), f"{best_model_name.replace(' ', '_')}_best_model_weights.pkl")
print(f"Best model: {best_model_name} with accuracy {best_overall_acc:.4f}")

# Accuracy Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Classifier', y='Accuracy', data=results_df)
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("classifier_accuracy_comparison.png")
plt.show()
