import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURES_FOLDER = 'features'


def load_data():
    all_data = []
    all_labels = []

    for file_name in os.listdir(FEATURES_FOLDER):
        if file_name.endswith('.csv'):
            label = os.path.splitext(file_name)[0]
            data_csv = pd.read_csv(os.path.join(FEATURES_FOLDER, file_name))
            data_csv.drop(columns=['file'], inplace=True)
            all_data.append(data_csv)
            all_labels.extend([label] * len(data_csv))

    data_df = pd.concat(all_data, ignore_index=True)
    labels = pd.Series(all_labels, name='label')

    # Convert labels to categorical codes and one-hot encode
    label_codes = labels.astype('category').cat.codes
    target = pd.get_dummies(label_codes)

    # Convert data to float32
    data_df = data_df.astype(np.float32)

    scaler = StandardScaler()
    data_df = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)

    return data_df, target


# Neural network model - 2 fully connected layers - ReLU activation for hidden layers
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 26)
        self.fc2 = nn.Linear(26, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# fucntion to train model for an epoch
def train_nn_model(data_loader, model, criterion, optimizer, device):
    model.train()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model


# fucntion to test model after an epoch
def test_nn_model(data_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            running_acc += (outputs.argmax(1) == labels.argmax(1)).float().mean().item()

    return running_loss / len(data_loader), 100 * running_acc / len(data_loader)


def save_metric_plots(metrics, epochs):
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(np.arange(1, epochs+1, 1))
    plt.savefig('models/nn_loss.png')

    plt.figure(figsize=(12, 6))
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(np.arange(1, epochs+1, 1))
    plt.savefig('models/nn_acc.png')


def main(epochs):
    logging.info(f'Training neural network\n')

    logging.info(f'Loading data')
    data, target = load_data()
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    train_data = torch.tensor(train_data.values).float()
    train_target = torch.tensor(train_target.values).float()
    test_data = torch.tensor(test_data.values).float()
    test_target = torch.tensor(test_target.values).float()

    # Create datasets and data loaders
    train_dataset = TensorDataset(train_data, train_target)
    test_dataset = TensorDataset(test_data, test_target)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    logging.info('Data loaded successfully\n')

    model = NeuralNetwork(data.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.0001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    metrics = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    logging.info('Training neural network\n')
    # training the model
    for epoch in range(epochs):
        model = train_nn_model(train_loader, model, criterion, optimizer, device)
        train_loss, train_acc = test_nn_model(train_loader, model, criterion, device)
        test_loss, test_acc = test_nn_model(test_loader, model, criterion, device)
        # print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        # print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')
        logging.info(f'Epoch {epoch + 1} :: Train Loss: {train_loss}, Test Acc: {test_acc}')
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)

    logging.info('\nFinished training neural network\n')

    logging.info('Saving metrics')
    save_metric_plots(metrics, epochs)
    logging.info('Metrics saved successfully in models folder\n')

    logging.info('Saving model')
    torch.save(model.state_dict(), f'models/nn.pth')
    logging.info('Model saved successfully\n')



# if __name__ == '__main__':
#     main()