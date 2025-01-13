import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from featurization.data_utils import load_data_from_df, construct_loader

df = pd.read_csv('CycPeptMPDB_Peptide_Assay_RRCK.csv', low_memory=False)[['SMILES', 'RRCK']]

df.columns = ['smiles', 'y']

df_select = df[df['y']>-10.0]

df_select.to_csv('rrck.csv', index=False)

train_df, test_df = train_test_split(df_select, test_size=0.3, random_state=12)

train_df.to_csv('rrck_train.csv', index=False)

test_df.to_csv('rrck_test.csv', index=False)

X_train, y_train = load_data_from_df('rrck_train.csv', ff='uff', ignoreInterfragInteractions=True, one_hot_formal_charge=True, use_data_saving=True)

X_test, y_test = load_data_from_df('rrck_test.csv', ff='uff', ignoreInterfragInteractions=True, one_hot_formal_charge=True, use_data_saving=True)


X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)

X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

X_train.to_pickle('./X_train.pkl')
y_train.to_pickle('./y_train.pkl')

X_test.to_pickle('./X_test.pkl')
y_test.to_pickle('./y_test.pkl')
