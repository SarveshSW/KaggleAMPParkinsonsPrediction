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
data = data.drop(['visit_id', 'patient_id', 'visit_month', 'upd23b_clinical_state_on_medication'], axis=1)

# handle missing values by replacing with mean of the column values 
data = data.fillna(data.mean())


# create a target matrix as the updrs columns
y = data[["updrs_1", "updrs_2", "updrs_3", "updrs_4"]]

# drop updrs columns from data
X = data.drop(['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'], axis=1)


# List of target labels to loop through and train models
target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]
def grad_comb(X, Y, ne, lr, md, ss):
    Y_train, Y_test, X_train, X_test = train_test_split(Y, X, test_size=0.2, random_state=0, shuffle=False)
    dt = GradientBoostingRegressor(n_estimators= ne, learning_rate= lr, max_depth= md, subsample= ss, random_state= 0)
    dt.fit(X_train,Y_train)

    Y_pred_original = dt.predict(X_test)
    
    mse_per_iteration = mean_squared_error(Y_test, Y_pred_original, multioutput='raw_values')

    print("Mean Squared Error:", mse_per_iteration)
    
    feature_importances = []
    for i in range(len(dt.feature_importances_)):
        feature_importances.append([i, dt.feature_importances_[i]])
    
    # remove features with importance less than 0.01
    feature_importances = [x for x in feature_importances if x[1] > 0.001]
    # get indexes of features to keep
    feature_indexes = [x[0] for x in feature_importances]
    # remove features from X and X_test
    X_train = X_train.iloc[:, feature_indexes]
    X_test = X_test.iloc[:, feature_indexes]
    
    dt2 = GradientBoostingRegressor(n_estimators= ne, learning_rate= lr, max_depth= md, subsample= ss, random_state= 0)
    dt2.fit(X_train,Y_train)
    Y_pred = dt2.predict(X_test)
    
    mse_per_iteration = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')

    print("Mean Squared Error:", mse_per_iteration)

n_estimators = [100, 300, 200, 500]
learning_rate = [0.01, 0.001, 0.001, 0.001]
max_depth = [1, 3, 3, 1]
sample_size = [0.75, 0.25, 0.25, 1]


for i, tar in enumerate(target):
    print(tar)
    grad_comb(X, y[tar], n_estimators[i], learning_rate[i], max_depth[i], sample_size[i])
