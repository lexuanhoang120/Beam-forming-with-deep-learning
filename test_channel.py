import numpy as np
import torch
from utils import mat_load, trans_Vrf, Rate_func
import matplotlib.pyplot as plt
import os
import pandas as pd


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
Nt = 64

# -----------------------
# Model Definition
# -----------------------
class BFNN(torch.nn.Module):
    def __init__(self, Nt, input_shape, activation='relu'):
        super(BFNN, self).__init__()
        self.activation = activation
        flat_dim = input_shape[2] * input_shape[3]

        self.flatten = torch.nn.Flatten()
        self.bn1 = torch.nn.BatchNorm1d(flat_dim)
        self.fc1 = torch.nn.Linear(flat_dim, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, Nt)

    def forward(self, imperfect_CSI, perfect_CSI, SNR_input):
        x = self.flatten(imperfect_CSI)
        x = self.bn1(x)

        if self.activation == 'relu':
            x = torch.relu(self.fc1(x))
        elif self.activation == 'tanh':
            x = torch.tanh(self.fc1(x))
        else:
            x = torch.sigmoid(self.fc1(x))

        x = self.bn2(x)

        if self.activation == 'relu':
            x = torch.relu(self.fc2(x))
        elif self.activation == 'tanh':
            x = torch.tanh(self.fc2(x))
        else:
            x = torch.sigmoid(self.fc2(x))

        x = self.bn3(x)
        phase = self.fc3(x)
        V_RF = trans_Vrf(phase)
        rate = Rate_func(perfect_CSI, V_RF, SNR_input,Nt)
        return rate

# -----------------------
# Evaluation Function
# -----------------------
def evaluate_model(model_path, activation, test_path, snr_range=range(-20, 25, 5)):
    # Load test data
    H, H_est = mat_load(test_path)
    H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], axis=1), axis=1)
    H = np.squeeze(H)
    H_input = torch.tensor(H_input, dtype=torch.float32)
    H_tensor = torch.tensor(H, dtype=torch.complex64)

    Nt = H.shape[1]
    model = BFNN(Nt, H_input.shape, activation=activation)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    results = []
    with torch.no_grad():
        for snr in snr_range:
            SNR = np.power(10, np.ones([H.shape[0], 1]) * snr / 10)
            SNR_tensor = torch.tensor(SNR, dtype=torch.float32)

            y_pred = model(H_input, H_tensor, SNR_tensor)
            avg_rate = -y_pred.mean().item()
            results.append((snr, avg_rate))

    return results

# -----------------------
# Define Paths and Parameters
# -----------------------

setup_paths = [
    "train_set/data_sets/-20db/test",
    "train_set/data_sets/0db/test",
    "train_set/data_sets/20db/test",
    "train_set/data_sets/Lest1/test",
    "train_set/data_sets/Lest2/test",
]

lst_learning_rate = [1e-3]  # learning rates 5e-5 , 1e-4,1e-5
lst_epochs = [1000]         # number of epochs 100,500,
lst_activation = ['relu']  # activation functions , 'tanh', 'sigmoid'
all_results = []


# -----------------------
# Evaluate Models
# -----------------------
for setup in setup_paths:
    model_name = setup.split('/')[-2]

    for act in lst_activation:
        for lr in lst_learning_rate:
            for epochs in lst_epochs:
                model_path = f"models_2/model_{model_name}_{act}_lr{lr}_ep{epochs}_best.pth"

                if not os.path.exists(model_path):
                    print(f"Skipping missing model: {model_path}")
                    continue

                print(f"Evaluating: {model_path}")
                metrics = evaluate_model(model_path, act, setup)

                for snr, rate in metrics:
                    all_results.append({
                        "Setup": model_name,
                        "Activation": act,
                        "LearningRate": lr,
                        "Epochs": epochs,
                        "SNR(dB)": snr,
                        "SpectralEfficiency": rate
                    })

# -----------------------
# Save Results to DataFrame
# -----------------------
df = pd.DataFrame(all_results)
df.to_csv("evaluation_results_channel.csv", index=False)
print("Evaluation results saved to 'evaluation_results_channel.csv'")



# test_path = "train_set/data_sets/0db/test"  # Common test set
