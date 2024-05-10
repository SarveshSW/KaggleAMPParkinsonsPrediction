import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import column_or_1d
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

protein_peptide_df = prepare_dataset(data_proteins, data_peptides)
FEATURES_LIST = [i for i in protein_peptide_df.columns if i not in ["visit_id"]]
FEATURES_LIST.append("visit_month")

updrs_list = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

for updrs in updrs_list: 
    print(updrs, "report: ")
    dataset_df = protein_peptide_df.merge(data_patients[['visit_id', 'patient_id', 'visit_month', updrs]], on = ['visit_id'], how = 'left')
    dataset_df = dataset_df.dropna(subset=[updrs])
    X = dataset_df[FEATURES_LIST]
    X = X.fillna(X.mean())
    Y = column_or_1d(dataset_df[updrs], warn=True)
    X_train, X_test,Y_train, Y_test, = train_test_split(X,Y, test_size=0.2, random_state=0)
    
    search_space = {
      'n_estimators': [100, 200, 300, 400, 500],
    }
    
    # changed to {1, 3, 5, 7, 9, 11}
    md = 11
    
    model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state = 0, max_depth = md), random_state=0)
    
    GS = GridSearchCV(model, search_space, cv=2, scoring="neg_mean_squared_error", verbose=0, n_jobs=-1)
    GS.fit(X_train, Y_train)
    print("Best parameters for max_depth = ", md)
    print(GS.best_params_)
    print(GS.best_score_)

