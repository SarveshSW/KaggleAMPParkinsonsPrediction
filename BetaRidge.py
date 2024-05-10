import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def batch_gradient_descent_diff(x, y, epochs, alpha, updrs):
    beta = np.ones(shape=(1196,1))
    phi = 0.5
    diff = np.zeros(shape=(epochs,1))
    ## loop for epoch
    for j in range (epochs):
        ## loop for each row
        gradient = 1/854 * (-2*x.T @ y[:, updrs - 1].reshape(854,1) + 2*x.T @ x @ beta ) + 2 * phi * beta
        beta = beta - alpha * gradient
        diff[j] = loss(Y_test[:, updrs - 1].reshape(214,1), X_test, beta)
    return diff

def batch_gradient_descent(x, y, epochs, alpha):
    beta = np.ones(shape=(1196,1))
    phi = 0.5
    ## loop for epoch
    for j in range (epochs):
        ## loop for each row
        gradient = 1/854 * (-2*x.T @ y + 2*x.T @ x @ beta ) + 2 * phi * beta
        beta = beta - alpha * gradient
    return beta



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


# func that finds the MSE of the model given the test data and beta values
def loss(Y_test, X_test, beta):
    return 1/854 * LA.norm(Y_test - X_test @ beta)**2

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
data = data.fillna(data.mean())

# cretae a target matrix as the updrs columns
Y = data[['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']]

# drop updrs columns from data
X = data.drop(['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'], axis=1)

# standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Center the target variable
Y = Y - Y.mean()


# split the data into training and testing sets
Y_train, Y_test, X_train, X_test = train_test_split(Y, X, test_size=0.2, random_state=0, shuffle=False)

# convert to numpy arrays
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()
X_train = np.hstack((np.ones(shape= (854,1)), X_train))
X_test = np.hstack((np.ones(shape= (214,1)), X_test))
print(X_train.shape)

# plot with differences of closed form and batch gradient descent
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), sharex=True, sharey=True)

# alpha array of 0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01
alpha_array = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]

# loop through each alpha and call batch gradient descent function
# obtain the beta values and plot the differences with the test data
for i in range(4):
    axes[i].set_title('updrs_' + str(i + 1))
    axes[i].plot(batch_gradient_descent_diff(X_train, Y_train, 100, 0.001, i + 1))

plt.show()

print(batch_gradient_descent(X_train, Y_train, 100, 0.001))

