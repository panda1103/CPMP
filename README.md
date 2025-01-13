# CPMP: Cyclic Peptide Membrane Permeability Prediction Using Deep Learning Model Based on Molecular Attention Transformer  
## Abstract  
The CPMP model is a deep learning approach for predicting the membrane permeability of cyclic peptides. Built on the Molecular Attention Transformer (MAT) neural network, CPMP achieves high determination coefficients (R²) of 0.67 for PAMPA, 0.75 for Caco-2, 0.62 for RRCK, and 0.73 for MDCK permeability predictions.
<img src="plot/pampa.png" alt="PAMPA" width="100" height="100">
![Caco2](plot/caco2.png)

## Requirements  
* PyTorch (2.0.1) 
* RDKit (2023.3.2) 
* Scikit-learn (1.3.0) 

## Usage  
### 1. Data
Before running CPMP on the same data used in the paper, the original data needs to be processed.  
For example, if you want to preprocess the PAMPA data, run:
```
cd data/pampa_uff_ig_true
python process_data.py
```
This takes around five hour on a regular computer.
Alternatively, pre-processed data (~6.2GB) can be found [here](https://zenodo.org/records/14638776). You can directly download and replace the `data` directory.

### 2. Train the CPMP model
For example, if you want to train CPMP mpdel with PAMPA data, run:
```
python train_pampa.py
```

### 3. Use the trained model to predict the test set
For example, if you want to predict  PAMPA test data, run:
```
python predict_pampa.py
```

### 4. Baseline
The implementation scripts for the baseline methods can be found at this path: `baselines/`

## MAT
The CPMP model is built based on the MAT framework. You can find more information about MAT [here](https://github.com/ardigen/MAT).

## License

CPMP is released under an [Apache v2.0 license](LICENSE).

