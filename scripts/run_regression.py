#!/usr/bin/env python3 -u

# To run this script:


################ imports #####################
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

import argparse

#################### Loading data ##########################




###################### Define Functions #######################

def load_mean_embeds(dir_path):
    '''Iterate over the layers and fit and predict using grid search'''
     # list of all files containing the embeddings and created a dictionary
    embeddings = {}
    for file in os.listdir(dir_path):
        if file.endswith('.pt'):
            file_path = os.path.join(dir_path, file)
            label = file.split('.pt')[0]
            embeddings[label] = np.array(torch.load(file_path)['mean_representations'])

    # convert to array type sequences_embeddings to a data frame
    df=pd.DataFrame.from_dict(embeddings, orient='index').reset_index()
    df.rename(columns={'index':'ID'}, inplace=True)
    return df


def run_regression(features, target, target_name, model, layer, n_comp):
    
    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        features, target, range(len(target)), test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA only on the training set
    pca = PCA(n_comp)
    X_train_PCAs = pca.fit_transform(X_train_scaled)
    
    # Apply the same PCA transformation to the test set
    X_test_PCAs = pca.transform(X_test_scaled)
    
    # Create a list of models to test
    if model == 'linear':
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
    elif model == 'lasso':
        model = Lasso()
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1],'max_iter': [10000]}
    elif model == 'svr':
        model = SVR()
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 0.1, 0.01, 0.001]}
    else:
        print('Model not found')
        print('Model options: linear, lasso, svr')
    
    # search best hyperparameter 
    grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=10, n_jobs=30)
    
    # train in the whole training set
    grid_search.fit(X_train_PCAs, y_train)
    
    # make predictions on the PCA-transformed test set
    y_pred = grid_search.predict(X_test_PCAs)
    train_pred = grid_search.predict(X_train_PCAs)
    
    # Model evaluation on test data set
    test_r2 = metrics.r2_score(y_test, y_pred)
    test_mae = metrics.mean_absolute_error(y_test, y_pred)
    test_mse = metrics.mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_spearman_Coef = scipy.stats.spearmanr(y_test, y_pred)
   
    model_name = grid_search.best_estimator_.__class__.__name__
    best_param = str(grid_search.best_params_)
    
    # Saving results
    res_dict = {}
    res_dict["Target_name"]= target_name
    res_dict["Layer"]= layer
    res_dict["N_components"]=  n_comp 
    res_dict["Model"]= model_name 
    res_dict["parameters"]= best_param
    
    res_dict["test_R2_score"] = test_r2
    res_dict["test_MAE_score"] = test_mae
    res_dict["test_RMSE_score"] = test_rmse
    res_dict["test_spearman_Coef"] = test_spearman_Coef
    
    # updating data frame with results
    results  = pd.DataFrame(res_dict).reset_index(drop=True)

    # Print the mean scores
    print(f"Layer: {layer}, Num componemts: {n_comp}, R2: {test_r2:.2f}, best params: {best_param}")

    # Plot the results
    filename = f"{target_name}_{layer}_{n_comp}_{model_name}.png"
    plt.scatter(train_pred, y_train, label="Train samples", c='#d95f02')
    plt.scatter(y_pred, y_test, label="Test samples", c='#7570b3')
    plt.title(f"Model: {model_name}")
    plt.xlabel("Predicted Stability")
    plt.ylabel("True Stability")
    
    global_min = np.floor(min(min(train_pred), min(y_train), min(y_test), min(y_pred)))
    global_max = np.ceil(max(max(train_pred), max(y_train), max(y_test), max(y_pred)))
   
    plt.plot([global_min, global_max], [global_min, global_max], c='k', zorder=0) 
    plt.text(global_min, global_max-5, f'Train R$^2$: {grid_search.score(X_train_PCAs, y_train).round(2)}', fontsize = 9)
    plt.text(global_min, global_max-4, f'Test R$^2$: {grid_search.score(X_test_PCAs, y_test).round(2)}', fontsize = 9)
    
    plt.legend()
    plt.savefig(filename, format='png', dpi=300)
    plt.show()
    plt.close()      

    #return results
    return results, test_indices, y_pred


############################# Run Predictions #############################
layer = '10'
target_name = 'stability'
model = 'lasso'
n_comp = 500
embed_dir = f'/stor/work/Wilke/luiz/tail_stability/embeddings/esm2_15B_embeds_per_layer/{layer}/'
meta_data_dir = '/stor/work/Wilke/luiz/tail_stability/data/tail_stability_metadata.csv'
output_dir= f'/stor/work/Wilke/luiz/tail_stability/results/tail_stability_regression_{model}_layer{layer}_pca{n_comp}.csv'

# Load meta data and embeddings
meta_data = pd.read_csv(meta_data_dir)

print(f'Loading embeddings from layer {layer}')
embeds = load_mean_embeds(embed_dir)
data = meta_data.merge(embeds, how='left', left_on='ID', right_on='ID')

# Define target and features
target = data[target_name]
# define features
features = data.iloc[:, meta_data.shape[1]:]

print(f'Running regression for {target_name} on layer {layer}')
#res = run_regression(features, target, target_name, model, layer, n_comp)
res, test_indices, y_pred = run_regression(features, target, target_name, model, layer, n_comp)

# Create a dataframe for y_test and y_pred
df_pred = pd.DataFrame({
    'ID': data['ID'].iloc[test_indices],
    'y_test': y_test.values,
    'y_pred': y_pred
})

df.to_csv(output_dir, index=False)

print('Process Finished')        
