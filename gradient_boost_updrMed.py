import pandas as pd
from sklearn.calibration import column_or_1d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Function to prepare dataset with all the steps mentioned above:
def prepare_dataset(train_proteins, train_peptides):
    # Step 1: Grouping 
    df_protein_grouped = train_proteins.groupby(['visit_id','UniProt'])['NPX'].mean().reset_index()
    df_peptide_grouped = train_peptides.groupby(['visit_id','Peptide'])['PeptideAbundance'].mean().reset_index()
    
    # Step 2: Pivoting
    df_protein = df_protein_grouped.pivot(index='visit_id',columns = 'UniProt', values = 'NPX').rename_axis(columns=None).reset_index()
    df_peptide = df_peptide_grouped.pivot(index='visit_id',columns = 'Peptide', values = 'PeptideAbundance').rename_axis(columns=None).reset_index()
    
    # Step 3: Merging
    pro_pep_df = df_protein.merge(df_peptide, on = ['visit_id'], how = 'left')
    
    return pro_pep_df


def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


data_patients = pd.read_csv('train_clinical_data.csv')
data_proteins = pd.read_csv('train_proteins.csv')
data_peptides = pd.read_csv('train_peptides.csv')

data_pp = prepare_dataset(data_proteins, data_peptides)


dataset_df = data_pp.merge(data_patients[['visit_id', 'patient_id', 'visit_month', 'updrs_1']], on = ['visit_id'], how = 'left')
dataset_df = dataset_df.dropna(subset=['updrs_1'])

# merge two dataframes and remove rows where visit_id do not occur in both datasets
data = pd.merge(data_patients, data_pp, on='visit_id')

# drop visit_id column, patient_id column, visit month and upd23b medication column
data = data.drop(['visit_id', 'patient_id', 'visit_month'], axis=1)

# handle missing values by replacing with mean of the column values 
med = data['upd23b_clinical_state_on_medication']
data.drop(labels=['upd23b_clinical_state_on_medication'], axis=1,inplace = True)
data = data.fillna(data.mean())

#p put back the upd23b_clinical_state_on_medication column
data = pd.concat([data, med], axis=1)


# create a target matrix as the updrs columns
y = data[["updrs_1", "updrs_2", "updrs_3", "updrs_4"]]

# drop updrs columns from data
X = data.drop(['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'], axis=1)


# List of target labels to loop through and train models
target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
def grad_comb(X, Y, ne, lr, md, ss):
    
    # get the index of upd23b_clinical_state_on_medication column
    col = X.columns.get_loc("upd23b_clinical_state_on_medication")
    
    # split data on upd23b_clinical_state_on_medication into two datasets for on and off
    X_off = X[X['upd23b_clinical_state_on_medication'] == 'On']
    X_on = X[X['upd23b_clinical_state_on_medication'] == 'Off']
    
    # remove upd23b_clinical_state_on_medication column from both datasets using col index
    X_off = np.delete(X_off, col, axis=1)
    X_on = np.delete(X_on, col, axis=1)
    
    # split Y into two datasets for on and off
    Y_off = Y[X['upd23b_clinical_state_on_medication'] == 'On']
    Y_on = Y[X['upd23b_clinical_state_on_medication'] == 'Off']
    
    Y_train_off, Y_test_off, X_train_off, X_test_off = train_test_split(Y_off, X_off, test_size=0.2, random_state=0, shuffle=False)
    Y_train_on, Y_test_on, X_train_on, X_test_on = train_test_split(Y_on, X_on, test_size=0.2, random_state=0, shuffle=False)
    
    dt = GradientBoostingRegressor(n_estimators= ne, learning_rate= lr, max_depth= md, subsample= ss, random_state= 0)
    dt.fit(X_train_off,Y_train_off)
    Y_pred_off = dt.predict(X_test_off)
    
    dt = GradientBoostingRegressor(n_estimators= ne, learning_rate= lr, max_depth= md, subsample= ss, random_state= 0)
    dt.fit(X_train_on,Y_train_on)
    Y_pred_on = dt.predict(X_test_on)

    print("Mean Squared Error for medication on:", mean_squared_error(Y_test_on, Y_pred_on))
    print("Mean Squared Error for medication off:", mean_squared_error(Y_test_off, Y_pred_off))

n_estimators = [100, 300, 200, 500]
learning_rate = [0.01, 0.001, 0.001, 0.001]
max_depth = [1, 3, 3, 1]
sample_size = [0.75, 0.25, 0.25, 1]


for i, tar in enumerate(target):
    print(tar)
    grad_comb(X, y[tar], n_estimators[i], learning_rate[i], max_depth[i], sample_size[i])
