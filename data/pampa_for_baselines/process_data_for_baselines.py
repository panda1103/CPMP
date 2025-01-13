import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def smiles_to_ecfp(smiles: list[str], radius: int = 6, nbits: int = 1024, silent: bool = True, to_array: bool = True) \
        -> np.ndarray:
    """ Get a Numpy array of ECFPs from a list of SMILES strings """
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    from rdkit.Chem import MolFromSmiles
    from rdkit.DataStructs import ConvertToNumpyArray

    if type(smiles) is str:
        smiles = [smiles]

    fp = [
        GetMorganFingerprintAsBitVect(MolFromSmiles(s), radius, nBits=nbits)
        for s in tqdm(smiles, disable=silent)
    ]

    if not to_array:
        return fp

    output = []
    for f in fp:
        arr = np.zeros((1, ))
        ConvertToNumpyArray(f, arr)
        output.append(arr)

    return np.asarray(output)

def process(df):
    smiles = np.array(df.smiles.tolist())
    fingerprints = smiles_to_ecfp(smiles, silent=False)
    return fingerprints

df_train = pd.read_csv('pampa_train.csv')

X_train = process(df_train)

torch.save(X_train, './X_train_fg.pt')

df_val = pd.read_csv('pampa_val.csv')

X_val = process(df_val)

torch.save(X_val, './X_val_fg.pt')

df_test = pd.read_csv('pampa_test.csv')

X_test = process(df_test)

torch.save(X_test, './X_test_fg.pt')
