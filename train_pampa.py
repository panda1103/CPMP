import os
import time
import numpy as np
import pandas as pd
import torch
import itertools
import torch.nn as nn
from sklearn import metrics
from featurization.data_utils import load_data_from_df, construct_loader
from model.transformer import make_model

import torch.optim as optim

def train(model, data_loader_train, criterion, lr, device):
    sample_size = 0
    total_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for batch in data_loader_train:
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        adjacency_matrix = adjacency_matrix.to(device)
        node_features = node_features.to(device)
        distance_matrix = distance_matrix.to(device)
        batch_mask = batch_mask.to(device)
        y = y.to(device)
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        sample_size += len(y)
    return total_loss/sample_size

def evaluate(model, data_loader_train, criterion, lr, device):
    outputs = []
    targets = []
    model = model.eval()
    sample_size = 0
    total_loss = 0
    for batch in data_loader_train:
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        adjacency_matrix = adjacency_matrix.to(device)
        node_features = node_features.to(device)
        distance_matrix = distance_matrix.to(device)
        batch_mask = batch_mask.to(device)
        targets.extend(y.view(1,-1)[0].numpy().tolist())
        y = y.to(device)
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        loss = criterion(output, y)
        total_loss += loss.item()
        outputs.extend(output.view(1,-1)[0].detach().cpu().numpy().tolist())
        sample_size += len(y)
    r2 = metrics.r2_score(targets, outputs)
    mse = metrics.mean_squared_error(outputs, targets)
    mae = metrics.mean_absolute_error(outputs, targets)
    return total_loss/sample_size, r2, mse, mae

#one_hot_formal_charge=True
def main():
    epochs = 600
    dataset = 'pampa'
    outdir = f"train_{dataset}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    repeat_list = [0, 1, 2]
    h_list = [64]
    N_list = [6]
    N_dense_list = [2]
    slope_list = [0.16]

    parameter_combinations = list(itertools.product(repeat_list, h_list, N_list, N_dense_list, slope_list))
    for repeat, h, N, N_dense, slope in parameter_combinations:
        ff = 'uff'
        ig = 'true'
        indir = f'{dataset}_{ff}_ig_{ig}'
        X_train = pd.read_pickle(f'./data/{indir}/X_train.pkl').values.tolist()
        X_val = pd.read_pickle(f'./data/{indir}/X_val.pkl').values.tolist()
        X_test = pd.read_pickle(f'./data/{indir}/X_test.pkl').values.tolist()
        y_train = pd.read_pickle(f'./data/{indir}/y_train.pkl').values.tolist()
        y_val = pd.read_pickle(f'./data/{indir}/y_val.pkl').values.tolist()
        y_test = pd.read_pickle(f'./data/{indir}/y_test.pkl').values.tolist()

        '''
        After the hyperparameters are determined, we merge the training set and the validation set to train the model.
        '''
        X_train.extend(X_val)
        y_train.extend(y_val)

        d_atom = X_train[0][0].shape[1]
        d_model = 64
        h = h
        N = N
        N_dense = N_dense
        slope = slope
        drop = 0.1
        lambda_attention = 0.1
        lambda_distance = 0.6
        aggregation = 'dummy_node'
        gpu = "cuda:2"
        total_start_time = time.time()
        model_params = {
            'd_atom': d_atom,
            'd_model': d_model,
            'N': N,
            'h': h,
            'N_dense': N_dense,
            'lambda_attention': lambda_attention, 
            'lambda_distance': lambda_distance,
            'leaky_relu_slope': slope, 
            'dense_output_nonlinearity': 'relu', 
            'distance_matrix_kernel': 'exp', 
            'dropout': drop,
            'aggregation_type': aggregation
        }
        batch_size = 32
        lr = 1e-3
        best_model_saver = f"{outdir}/repeat_{repeat}_{ff}_ig_{ig}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.best_wegiht.pth"
        LOG_FILE = f"{outdir}/repeat_{repeat}_{ff}_ig_{ig}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.result.csv"
        best_loss = None
        use_cuda = torch.cuda.is_available()
        device = torch.device(gpu if use_cuda else "cpu")
        model = make_model(**model_params)
        model = model.to(device)
        for epoch in range(1, epochs+1):
            data_loader_train = construct_loader(X_train, y_train, batch_size)
            data_loader_test = construct_loader(X_test, y_test, batch_size)
            criterion = nn.MSELoss(reduction='sum')
            train_loss = train(model, data_loader_train, criterion, lr, device)
            test_loss, r2, mse, mae = evaluate(model, data_loader_test, criterion, lr, device)
            if not best_loss or best_loss > test_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), best_model_saver)

            print(f"epoch={epoch}, train_loss={train_loss}, test_loss={test_loss}, r2={r2}, mse={mse}, mae={mae}, best_loss={best_loss}")
            #print("-"*100)
            #print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
            results = pd.DataFrame({"epoch":[epoch], "train_loss":[train_loss], "test_loss":[test_loss], "r2":[r2], "mse":[mse], "mae":[mae], "best_loss":[best_loss]})
            results.to_csv(LOG_FILE,
                           mode='a',
                           index=False,
                           header=False if os.path.isfile(LOG_FILE) else True)


if __name__ == '__main__':
    main()

