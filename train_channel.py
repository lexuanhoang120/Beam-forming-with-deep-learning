import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
from torch.utils.tensorboard import SummaryWriter
from utils import mat_load, trans_Vrf, Rate_func 

# Set random seed
torch.manual_seed(2020)
np.random.seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Nt = 64  # number of antennas
P = 1    # normalized transmit power

# -----------------------
# define the training function
# -----------------------
def train_on_setup(setup_path, model_save_path, Nt, epochs=100, learning_rate=1e-3,activation='relu'):

    H, H_est = mat_load(setup_path)
    H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
    H = np.squeeze(H)
    SNR = np.power(10, np.random.randint(-20, 20, [H.shape[0], 1]) / 10)

    H_input_tensor = torch.tensor(H_input, dtype=torch.float32).to(device)
    H_tensor = torch.tensor(H, dtype=torch.complex64).to(device)
    SNR_tensor = torch.tensor(SNR, dtype=torch.float32).to(device)

    # Define the model
    class BFNN(nn.Module):
        def __init__(self, Nt):
            super(BFNN, self).__init__()
            self.flatten = nn.Flatten()
            self.bn1 = nn.BatchNorm1d(H_input.shape[2] * H_input.shape[3])
            self.fc1 = nn.Linear(H_input.shape[2] * H_input.shape[3], 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, Nt)
            

        def forward(self, imperfect_CSI, perfect_CSI, SNR_input):
            x = self.flatten(imperfect_CSI)
            x = self.bn1(x)

            if activation == 'relu':
                x = torch.relu(self.fc1(x))
            elif activation == 'tanh':
                x = torch.tanh(self.fc1(x))
            else:
                x = torch.sigmoid(self.fc1(x))
                
            x = self.bn2(x)

            if activation == 'relu':
                x = torch.relu(self.fc2(x))
            elif activation == 'tanh':
                x = torch.tanh(self.fc2(x))
            else:
                x = torch.sigmoid(self.fc2(x))

            x = self.bn3(x)
            phase = self.fc3(x)
            V_RF = trans_Vrf(phase)  # Custom PyTorch version
            rate = Rate_func(perfect_CSI, V_RF, SNR_input,Nt)  # Custom PyTorch version
            return rate
        
        
    # Custom Dataset
    class BFNN_Dataset(torch.utils.data.Dataset):
        def __init__(self, H_input, H, SNR):
            self.H_input = H_input
            self.H = H
            self.SNR = SNR

        def __len__(self):
            return len(self.SNR)

        def __getitem__(self, idx):
            return self.H_input[idx], self.H[idx], self.SNR[idx]

    # Create dataset and dataloaders
    dataset = BFNN_Dataset(H_input_tensor, H_tensor, SNR_tensor)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)

    model = BFNN(Nt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=20, min_lr=5e-5)

    def loss_fn(y_pred):
        return y_pred.mean()
    
    best_val_loss = float('inf')
    best_model_path = model_save_path.replace('.pth', '_best.pth')
    latest_model_path = model_save_path.replace('.pth', '_latest.pth')
    writer = SummaryWriter(log_dir=f'runs/{model_name}_{activation}_lr{learning_rate}_ep{epochs}')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imperfect_CSI, perfect_CSI, SNR_input in train_loader:

            imperfect_CSI = imperfect_CSI.to(device)
            perfect_CSI = perfect_CSI.to(device)
            SNR_input = SNR_input.to(device)

            optimizer.zero_grad()
            rate = model(imperfect_CSI, perfect_CSI, SNR_input)
            loss = loss_fn(rate)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)



        model.eval()
        with torch.no_grad():
            val_loss = 0
            for imperfect_CSI, perfect_CSI, SNR_input in val_loader:

                imperfect_CSI = imperfect_CSI.to(device)
                perfect_CSI = perfect_CSI.to(device)
                SNR_input = SNR_input.to(device)

                rate = model(imperfect_CSI, perfect_CSI, SNR_input)
                loss = loss_fn(rate)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

        print(f"[{setup_path}] Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        # Save latest model
        torch.save(model.state_dict(), latest_model_path)

        # Log to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', learning_rate, epoch)
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved new best model: {best_model_path}")

        print(f"[{setup_path}] Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")


    writer.close()
    # print(f"Model saved to {model_save_path}")

# Paths to the training datasets
# ------------------------------------------
setup_paths = [
    "data_sets/-20db/train",
    "data_sets/0db/train",
    "data_sets/20db/train",
    "data_sets/Lest1/train",
    "data_sets/Lest2/train",
]

lst_learning_rate = [1e-3]  # learning rates 5e-5 , 1e-4,1e-5
lst_epochs = [1000]         # number of epochs 100,500,
lst_activation = ['relu']  # activation functions , 'tanh', 'sigmoid'

for setup in setup_paths:
    model_name = setup.split('/')[-2]  # e.g., "0db"
    for epochs in lst_epochs:
        for act in lst_activation:
            for lr in lst_learning_rate:
            
                save_path = f"models_channel/model_{model_name}_{act}_lr{lr}_ep{epochs}.pth"
                print(f"Training on {setup} with {act} activation, lr={lr}, epochs={epochs}")
                train_on_setup(
                    setup_path=setup,
                    model_save_path=save_path,
                    Nt=Nt,
                    epochs=epochs,
                    activation=act,
                    learning_rate=lr
                )
