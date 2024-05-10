import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

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
data = data.drop(['visit_id', 'patient_id', 'visit_month', 'upd23b_clinical_state_on_medication'], axis=1)

# handle missing values by replacing with mean of the column values 
data = data.fillna(data.mean())


# create a target matrix as the updrs columns
y = data[["updrs_1", "updrs_2", "updrs_3", "updrs_4"]]

# drop updrs columns from data
X = data.drop(['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'], axis=1)


# List of target labels to loop through and train models
target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
def grad_comb(X, Y):
    Y_train, Y_test, X_train, X_test = train_test_split(Y, X, test_size=0.2, random_state=0, shuffle=False)
    dt = DecisionTreeRegressor(random_state= 0)

    # hyperparameters set
    search_space = {
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
    
    GS = GridSearchCV(dt, search_space, cv=5, scoring="neg_mean_squared_error", n_jobs= -1)
    
    GS.fit(X_train,Y_train)

    return GS.best_params_

for tar in target:
    print(tar)
    print(grad_comb(X, y[tar]))





