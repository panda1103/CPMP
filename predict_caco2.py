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

def evaluate(model, data_loader_train, criterion, lr, device):
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

#one_hot_formal_charge=True
def main():
    dataset = 'caco2'
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
        X_test = pd.read_pickle(f'./data/{indir}/X_test.pkl').values.tolist()
        y_test = pd.read_pickle(f'./data/{indir}/y_test.pkl').values.tolist()

        d_atom = X_test[0][0].shape[1]
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
        batch_size = 64
        lr = 1e-3
        best_model_saver = f"{outdir}/repeat_{repeat}_{ff}_ig_{ig}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.best_wegiht.pth"
        LOG_FILE = f"{outdir}/repeat_{repeat}_{ff}_ig_{ig}_h{h}_N{N}_N_dense{N_dense}_slope{slope}_batch_size{batch_size}.predict.csv"
        best_loss = None
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = torch.device(gpu)
            pretrained_state_dict = torch.load(best_model_saver, weights_only=True, map_location=torch.device(gpu))
        else:
            device = torch.device("cpu")
            pretrained_state_dict = torch.load(best_model_saver, weights_only=True, map_location=torch.device('cpu'))
        model = make_model(**model_params)
        model_state_dict = model.state_dict()
        for name, param in pretrained_state_dict.items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

        model = model.to(device)
        data_loader_test = construct_loader(X_test, y_test, batch_size, shuffle=False)
        criterion = nn.MSELoss()
        y_predict = evaluate(model, data_loader_test, criterion, lr, device)
        y = [item for sublist in y_test for item in sublist]
        df = pd.DataFrame({"y": y})
        df['predict'] = y_predict
        df.to_csv(LOG_FILE,
                       mode='w',
                       index=False,
                       header=True)

        threshold = -6

        df['y_binary'] = (df['y'] > threshold).astype(int)

        # 计算 AUC
        auc_score = roc_auc_score(df['y_binary'], df['predict'])
        print(f"ROC-AUC: {auc_score}")

        # 计算 PRC
        precision, recall, _ = precision_recall_curve(df['y_binary'], df['predict'])
        prc_auc = auc(recall, precision)
        print(f"PR-AUC: {prc_auc}")

        # 计算精度
        df['predict_binary'] = (df['predict'] > threshold).astype(int)
        accuracy = accuracy_score(df['y_binary'], df['predict_binary'])
        print(f"Accuracy: {accuracy}")

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(df['y_binary'], df['predict_binary'])
        print("Confusion Matrix:")
        print(conf_matrix)

        # 打印分类报告
        class_report = classification_report(df['y_binary'], df['predict_binary'])
        print("Classification Report:")
        print(class_report)



if __name__ == '__main__':
    main()

