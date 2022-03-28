import os.path
import numpy as np
import pandas as pd
import torch as T
import matplotlib.pyplot as plt

from os.path import exists
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

T.set_default_dtype(T.float64)

device = T.device('cpu')
independent_data_col = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]
dependent_data_col = [46]
derived_data_col = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
key_fitting_col = [5, 6, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]
l_brk_col = [47]

derived_names = ['VA', 'Beta', 'Beta_para', 'Beta_perp', 'Omega_i', 'omega_pi', 'omega_pe', 'rho_i', 'rho_s', 'rho_c',
                 'd_i', 'd_e', 'sigma_i', 'Lperp', 'lambda_r']

# Data examination
# How do independent variables relate to the dependent variable?
# Split up data based on .pdf file
independent_data = pd.read_csv('Terres_Li_Sample_Data_Set.csv', usecols=independent_data_col)
dependent_data = pd.read_csv('Terres_Li_Sample_Data_Set.csv', usecols=dependent_data_col)
derived_data = pd.read_csv('Terres_Li_Sample_Data_Set.csv', usecols=derived_data_col)
key_fitting_data = pd.read_csv('Terres_Li_Sample_Data_Set.csv', usecols=key_fitting_col)
l_brk_data = pd.read_csv('Terres_Li_Sample_Data_Set.csv', usecols=l_brk_col)

# Combine data we are training aka independent and dependent variables
data = key_fitting_data.join(dependent_data)

# What does the correlation matrix look like?
correlation_matrix = data.corr().abs()
s = correlation_matrix.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
print("Correlation Matrix for independent fitting variables in descending order")

# Loop over data and print all k_brk data
for name, val in so.iteritems():
    if name[0] == 'k_brk':
        print(name, val)

# Check if any other independent variables are better
data_two = independent_data.join(dependent_data)
correlation_matrix = data_two.corr().abs()
s = correlation_matrix.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
print("\nCorrelation Matrix for all independent data in descending order")

# Loop over data and print all k_brk data
for name, val in so.iteritems():
    if name[0] == 'k_brk':
        print(name, val)


# temp = input("Press any key to continue...")

# Everything is weakly correlated with VSW being the strongest at 0.35

# Create three different possible models:
# First an unnormalized model
class TerresLiData(T.utils.data.Dataset):
    def __init__(self, source):
        all_data = np.loadtxt(source, delimiter=',', dtype=np.float64)

        temp_x = all_data[:, [i for i in range(all_data.shape[1] - 1)]]
        temp_y = all_data[:, [-1]]

        self.x = T.tensor(temp_x, dtype=T.float64).to(device)
        self.y = T.tensor(temp_y, dtype=T.float64).to(device)

    def __getitem__(self, idx):
        preds = self.x[idx, :]
        k_brk = self.y[idx, :]
        return preds, k_brk

    def __len__(self):
        return len(self.x)


# Define a class for the MLP
class Network(T.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = T.nn.Sequential(
            T.nn.Linear(size, 20),
            T.nn.ReLU(),
            T.nn.Linear(20, 10),
            T.nn.Sigmoid(),
            T.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Define a class for the MLP
class Network_BNormalized(T.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = T.nn.Sequential(
            T.nn.Linear(size, 20),
            T.nn.BatchNorm1d(20),
            T.nn.ReLU(),
            T.nn.Linear(20, 10),
            T.nn.BatchNorm1d(10),
            T.nn.Sigmoid(),
            T.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x)


# Define a function to initialize weights and biases
def init_weights_bias(network):
    if isinstance(network, T.nn.Linear):
        T.nn.init.xavier_uniform_(network.weight)
        T.nn.init.zeros_(network.bias)


# Define a function to split the data into test/train/validation. Save in .csv format
def split_data(data_col, target_col, normalize = False):
    # Columns for test are only key fitting and dependent variables
    use_cols = data_col + target_col
    # Check if data already exists. If true, print and
    if os.path.exists('Terres_Li_Train.csv') and os.path.exists('Terres_Li_Test.csv') and os.path.exists(
            'Terres_Li_val.csv'):
        print("Files exist. Removing to remake files")
        os.remove('Terres_Li_Train.csv')
        os.remove('Terres_Li_Test.csv')
        os.remove('Terres_Li_val.csv')
    print("Creating train, test, and validation data")
    # load data from .csv, skipping the header
    all_data = np.loadtxt('Terres_Li_Sample_Data_Set.csv', dtype=np.float64, usecols=use_cols, delimiter=',', skiprows=1)
    # Split data into data and target
    temp_x = all_data[:, [i for i in range(len(data_col))]]
    temp_y = all_data[:, -1]
    # Create test train data with test size = 20%
    X_train, X_test, y_train, y_test = train_test_split(temp_x, temp_y, test_size=0.2, shuffle=True)
    # Create the validation data with validation size = ~10% of total data (13% * 80% = 10%)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.13, shuffle=False)

    # normalize if needed:
    if normalize:
        normalizer = MinMaxScaler()
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)
        #X_val = normalizer.transform(X_val)

    # Combine data for saving to .csv file
    XY_train = np.column_stack((X_train, y_train))
    XY_test = np.column_stack((X_test, y_test))
    #XY_val = np.column_stack((X_val, y_val))

    np.savetxt("Terres_Li_Train.csv", XY_train, delimiter=',')
    np.savetxt("Terres_Li_Test.csv", XY_test, delimiter=',')
    #np.savetxt("Terres_Li_val.csv", XY_val, delimiter=',')


def main():
    split_data(key_fitting_col, dependent_data_col)
    loss = []

    # Load each dataset
    train_ds = TerresLiData('Terres_LI_Train.csv')
    test_ds = TerresLiData('Terres_LI_Test.csv')
    #val_ds = TerresLiData('Terres_LI_val.csv')
    batch_size = 10

    # Load training dataset using DataLoader
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Create MLP which uses CPU and apply weight/bias
    net = Network(train_ds.x.shape[1]).to(device)
    net.apply(init_weights_bias)

    # Use 500 training cycles at a very low learning rate for best results
    max_epochs = 500
    learning_rate = 0.0001
    r2 = []

    # Optimizer is Adam and Loss is MSE
    loss_func = T.nn.MSELoss()
    optimizer = T.optim.Adam(net.parameters(), lr=learning_rate)

    print("\nStart of training")
    net.train()
    for epochs in range(max_epochs):
        epoch_loss = 0

        # Feed forward and backward
        for (batch_idx, batch) in enumerate(train_ldr):
            (X, Y) = batch
            optimizer.zero_grad()
            y_hat = net(X)
            loss_val = loss_func(y_hat, Y)
            epoch_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()

        loss.append(epoch_loss)
        print("epoch = %4d   loss = %0.16f" % (epochs + 1, epoch_loss))

        if epochs % 50 == 0:
            pred = test_ds.x
            with T.no_grad():
                pred_kbrk = net(pred)
            prediction_array = pred_kbrk.detach().cpu().numpy()
            target = test_ds.y
            target_array = target.detach().cpu().numpy()
            r2.append(r2_score(target_array, prediction_array))
            print(r2[-1])

    # After training, set MLP to evaluation and perform a prediction
    net.eval()
    pred = test_ds.x

    with T.no_grad():
        pred_kbrk = net(pred)

    # Convert values to numpy for R2 score
    prediction_array = pred_kbrk.detach().cpu().numpy()
    target = test_ds.y
    target_array = target.detach().cpu().numpy()
    r2.append(r2_score(target_array, prediction_array))
    print(r2[-1])

    # Plot the loss
    plt.plot(loss, label='Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    # Typically loss starts too high to see detail. Try with only values less than 0.01
    i = 0
    while i < len(loss):
        if loss[i] >= 0.01:
            loss.pop(i)
        else:
            i += 1

    # Plot the loss
    plt.plot(loss, label='Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    # Plot the r2 score
    plt.plot(r2, label='R2 Score')
    plt.title("R2 Score Every 50 Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.legend(loc='best')
    plt.show()

    # The R2 score is typically a negative number. Various testing can reduce the R2 score to a double-digit negative
    # but this is still poor. Next attempt to narrow down the best derived variables


if __name__ == "__main__":
    main()
