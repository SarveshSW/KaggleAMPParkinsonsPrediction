import pandas as pd
from sklearn.calibration import column_or_1d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def grad_boost(x, y, search_space):
    #get indexes of values in y that are nan
    nan_indexes = np.argwhere(np.isnan(y))
    #remove nan values from y and rows in x
    y = np.delete(y, nan_indexes)
    x = np.delete(x, nan_indexes, axis=0)
    
    clf = GradientBoostingRegressor(random_state= 0)
    GS = GridSearchCV(estimator= clf, param_grid= search_space, cv= 2, n_jobs= -1, verbose= 1, scoring= 'neg_mean_squared_error')
    GS.fit(x,y)

    # Evaluate the Model (MSE per iteration)
    return GS.best_params_

def prepare_dataset(train_proteins, train_peptides):
    # Step 1: Grouping 
    df_protein_grouped = train_proteins.groupby(['visit_id','UniProt'])['NPX'].mean().reset_index()
    df_peptide_grouped = train_peptides.groupby(['visit_id','Peptide'])['PeptideAbundance'].mean().reset_index()
    
    # Step 2: Pivoting
    df_protein = df_protein_grouped.pivot(index='visit_id',columns = 'UniProt', values = 'NPX').rename_axis(columns=None).reset_index()
    df_peptide = df_peptide_grouped.pivot(index='visit_id',columns = 'Peptide', values = 'PeptideAbundance').rename_axis(columns=None).reset_index()
    
    # Step 3: Merging
    protein_peptide_data = df_protein.merge(df_peptide, on = ['visit_id'], how = 'left')
    
    return protein_peptide_data


# loss function that is half of the predicted - actual squared
def loss(a, b):
    return 1/2*(a-b)**2

# Read the CSV file into a pandas DataFrame from Main file
data_patients = pd.read_csv('train_clinical_data.csv')
data_proteins = pd.read_csv('train_proteins.csv')
data_peptides = pd.read_csv('train_peptides.csv')

data_pp = prepare_dataset(data_proteins, data_peptides)
# sort data_pp and data_patients on visit_id
data_patients = data_patients.sort_values(by=['visit_id'])
data_pp = data_pp.sort_values(by=['visit_id'])


# merge two dataframes and remove rows where visit_id do not occur in both datasets
data = pd.merge(data_patients, data_pp, on='visit_id')

# drop visit_id column, patient_id column, visit month and upd23b medication column
data = data.drop(['visit_id', 'patient_id', 'visit_month', 'upd23b_clinical_state_on_medication'], axis=1)

# handle missing values by replacing with mean of the column values 

# cretae a target matrix as the updrs columns
Y = data[['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']]

# drop updrs columns from data
X = data.drop(['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'], axis=1)
X = X.fillna(data.mean())

# standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Center the target variable
Y = Y - Y.mean()
Y = Y.to_numpy()

# hyperparameters set
search_space = {
    'subsample': [0.25, 0.5, 0.75, 1.0],
    'max_depth': [1, 3, 5, 8, 10],
    'n_estimators': [100, 200, 300, 400, 500], 
    'learning_rate': [0.1, 0.01, 0.001, 0.0001]
}

#print("updrs_1")
#print(grad_boost(X, Y[:,0], search_space))
#print("updrs_2")
#print(grad_boost(X, Y[:,1], search_space))
print("updrs_3")
print(grad_boost(X, Y[:,2], search_space))
print("updrs_4")
print(grad_boost(X, Y[:,3], search_space))

