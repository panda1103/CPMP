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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, confusion_matrix, classification_report

import torch.optim as optim

def evaluate(model, data_loader_train, device):
    outputs = []
    model = model.eval()
    for batch in data_loader_train:
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        adjacency_matrix = adjacency_matrix.to(device)
        node_features = node_features.to(device)
        distance_matrix = distance_matrix.to(device)
        batch_mask = batch_mask.to(device)
        y = y.to(device)
        output = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
        outputs.extend(output.view(1,-1)[0].detach().cpu().numpy().tolist())
    return outputs

def predict(data_loader, saved_model):
    d_atom = 28
    ff = 'uff'
    ig = 'true'
    d_model = 64
    h = 64
    N = 6
    N_dense = 2
    slope = 0.16
    drop = 0.1
    lambda_attention = 0.1
    lambda_distance = 0.6
    aggregation = 'dummy_node'
    gpu = "cuda:0"
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
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(gpu)
        pretrained_state_dict = torch.load(saved_model, weights_only=True, map_location=torch.device(gpu))
    else:
        device = torch.device("cpu")
        pretrained_state_dict = torch.load(saved_model, weights_only=True, map_location=torch.device('cpu'))
    model = make_model(**model_params)
    model_state_dict = model.state_dict()
    for name, param in pretrained_state_dict.items():
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)

    model = model.to(device)
    y_predict = evaluate(model, data_loader, device)
    return  y_predict

#one_hot_formal_charge=True
import argparse
def main():
    parser = argparse.ArgumentParser(description="Predict PAMPA, Caco-2, RRCK and MDCK Membrane Permeability")
    
    parser.add_argument("--input_file", type=str, help=".csv file")
    parser.add_argument("--result_file", type=str, help="out file")
    
    args = parser.parse_args()
    
    INPUT_CSV = args.input_file
    RESULT_FILE = args.result_file
    DATASETS = ['pampa', 'caco2', 'rrck', 'mdck']
    df = pd.read_csv(INPUT_CSV)
    X, y = load_data_from_df(INPUT_CSV, ff='uff', ignoreInterfragInteractions=True, one_hot_formal_charge=True, use_data_saving=False)
    data_loader = construct_loader(X, y, batch_size=2, shuffle=False)
    result_dict = {}
    result_dict['smiles'] = df['smiles'].values
    for dataset in DATASETS:
        saved_model = f"saved_model/{dataset}.best_wegiht.pth"
        y = predict(data_loader, saved_model)
        result_dict[dataset] = y
    df = pd.DataFrame(result_dict)
    df.to_csv(RESULT_FILE, mode='w', index=False, header=True)

if __name__ == '__main__':
    main()

