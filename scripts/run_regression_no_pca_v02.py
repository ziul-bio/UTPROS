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


# Split the data into training and testing sets
def split_data(features, target):
    '''Split the data into training and testing sets'''
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    '''Scale the data'''
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def dimention_reduction(X_train_scaled, X_test_scaled, n_components=100):
    '''Perform dimention reduction using PCA'''
    pca = PCA(n_components, random_state=42)
    # Apply PCA only on the training set
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca


def fit_regression_model(X, y, model='linear'):
    '''Fit the regression model'''
    if model == 'linear':
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False]}
    elif model == 'lasso':
        model = Lasso()
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'max_iter': [100000]}
    elif model == 'svr':
        model = SVR()
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 0.1, 0.01, 0.001]}
    else:
        print('Model not found')
        print('Model options: linear, lasso, svr')
        return None, None  # early exit

    # search best hyperparameters 
    grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=10, n_jobs=10, refit=True)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    
    # Explicitly refitting the best estimator on the entire dataset
    best_model = grid_search.best_estimator_
    #best_model.fit(X, y)

    return best_model, best_params




def make_predictions(model, X):
    '''Make predictions'''
    pred = model.predict(X)
    return pred



def compute_metrics(pred, test):
    '''Compute metrics'''
    r2 = metrics.r2_score(test, pred)
    mae = metrics.mean_absolute_error(test, pred)
    mse = metrics.mean_squared_error(test, pred)
    rmse = np.sqrt(mse)
    spearman_Coef = scipy.stats.spearmanr(test, pred)

    return {'R2':r2, 'MAE': mae, 'RMSE' : rmse, 'SpearmanCoef': spearman_Coef[0], 'SpearmanPvalue': spearman_Coef[1]}




############################# Run Predictions #############################
layer = '10'
target_name = 'stability'
model_name = 'lasso'
embed_dir = f'/stor/work/Wilke/luiz/tail_stability/embeddings/esm2_15B_embeds_per_layer/{layer}/'
meta_data_dir = '/stor/work/Wilke/luiz/tail_stability/data/tail_stability_metadata.csv'
output_dir= f'/stor/work/Wilke/luiz/tail_stability/results/tail_stability_regression_{model_name}_layer{layer}_v02'

# Load meta data and embeddings
meta_data = pd.read_csv(meta_data_dir)

print(f'Loading embeddings from layer {layer}')
embeds = load_mean_embeds(embed_dir)
data = meta_data.merge(embeds, how='inner', left_on='ID', right_on='ID')

# Define target and features
target = data[target_name]
# define features
features = data.iloc[:, meta_data.shape[1]:]

print(f'Running regression for {target_name} on layer {layer}')
X_train, X_test, y_train, y_test = split_data(features, target)

X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
print(f'Features scaled shape: {X_train_scaled.shape}')

model, best_params = fit_regression_model(X_train_scaled, y_train, model='lasso')
print(best_params)

test_pred = make_predictions(model, X_test_scaled)
train_pred = make_predictions(model, X_train_scaled)


scores = compute_metrics(test_pred, y_test)
scores.update({'BestParameters': best_params})
print(f'Metrics: {scores}')

pd.DataFrame(scores, index=[0]).to_csv(f'{output_dir}_scores.csv', index=False)


res = pd.DataFrame({'Target': y_test, 'Predictions': test_pred})
res_df = data.iloc[:, :3].merge(res, how='inner', left_index=True, right_index=True)
res_df.to_csv(f'{output_dir}_prediction.csv', index=False)

print('Process Finished')        
