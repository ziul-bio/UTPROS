#!/usr/bin/env python3 -u

# To run this script:
# time python run_regression_all_layers.py -n 'fitness' -i data/BLAT_ECOLX_Ostermeier2014_metadata.csv -I embeddings/blat_ostermeier2014_esm2_15B_mean -o results/regression/v03/res_reg_Blac_ostermeir2014.csv
# python run_regression_all_layers.py -n 'normalized_fitness' -i data/amiE_metadata.csv -I embeddings/amiE_esm2_15B_mean -o results/regression/v03/res_reg_amiE.csv

################ imports #####################
import os
import numpy as np
import pandas as pd

import scipy
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

import argparse

#################### Loading data ##########################


parser = argparse.ArgumentParser(description="A script that accepts inputs and outputs arguments.")
parser.add_argument("-n", "--target", help="Target name")
parser.add_argument("-i", "--metadata", help="Input meta data file path")
parser.add_argument("-I", "--embeds", help="Embedding directory data")
parser.add_argument("-o", "--output", help="Output file path")

args = parser.parse_args()

target_name = args.target
meta_data = pd.read_csv(args.metadata)
embed_dir = args.embeds
output_dir = args.output

rep_layer = 48

# target_name = 'Activity'
# meta_data = pd.read_csv('/stor/work/Wilke/luiz/multi_layers_analysis/data/P62593_meta_data.csv')
# embed_dir = '/stor/work/Wilke/luiz/multi_layers_analysis/embeddings/P62593_esm2_15B_mean'
# output_dir = '/stor/work/Wilke/luiz/multi_layers_analysis/results/regression/v03/P62593_all_layers.csv'


###################### Define Functions #######################

def load_mean_embeds(dir_path, layer):
    '''Iterate over the layers and fit and predict using grid search'''
     # list of all files containing the embeddings and created a dictionary
    embeddings = {}
    for file in os.listdir(dir_path):
        if file.endswith('.pt'):
            file_path = os.path.join(dir_path, file)
            label = file.split('.pt')[0]
            embeddings[label] = np.array(torch.load(file_path)['mean_representations'][layer])

    # convert to array type sequences_embeddings to a data frame
    df=pd.DataFrame.from_dict(embeddings, orient='index').reset_index()
    df.rename(columns={'index':'ID'}, inplace=True)
    return df



def run_regression(features, target, target_name, layer):
    # lists to store the results
    r2s = []
    maes = []
    mses = []
    rmses = []
    spearman_Coefs = []
    random_states = [21, 32, 42, 64, 84, 128, 166, 256, 332, 512]
    results = pd.DataFrame()
    
    for i in random_states:

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=i)

        # Apply PCA only on the training set
        n_components = 0.9
        pca = PCA(n_components)
        X_train_PCAs = pca.fit_transform(X_train)

        # Apply the same PCA transformation to the test set
        X_test_PCAs = pca.transform(X_test)

        # Create a list of models to test
        model = LinearRegression()

        # train
        model.fit(X_train_PCAs, y_train)

        # make predictions on the PCA-transformed test set
        y_pred = model.predict(X_test_PCAs)

        # Model evaluation on test data set
        r2 = metrics.r2_score(y_test, y_pred)
        r2s.append(r2)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        maes.append(mae)
        mse = metrics.mean_squared_error(y_test, y_pred)
        mses.append(mse)
        rmse = np.sqrt(mse)
        rmses.append(rmse)
        spearman_Coef = scipy.stats.spearmanr(y_test, y_pred)
        spearman_Coefs.append(spearman_Coef[0])


    # Saving results
    res_dict = {}
    res_dict["Target_name"] = [target_name] * 10
    res_dict["Layer"] = [layer] * 10
    res_dict["Model"] = ['Linear Regression'] * 10
    res_dict["R2_score"] = r2s
    res_dict["MAE_score"] = maes
    res_dict["RMSE_score"] = rmses
    res_dict["spearman_Coef"] = spearman_Coefs

    # updating data frame with results
    res = pd.DataFrame(res_dict).reset_index(drop=True)
    results = pd.concat([results, res], axis=0, ignore_index=True)

    # Print the mean scores
    print(f"Layer: {layer}, Mean 10 splits R2: {np.mean(r2s):.2f}, Mean 10 splits Spearman Coef:{np.mean(spearman_Coefs)}")

    return results



############################# Run Predictions #############################

results_df = pd.DataFrame()


for layer in range(0, rep_layer+1):
    print(f'Running regression for {target_name} on layer {layer}')
    embeds = load_mean_embeds(embed_dir, layer)
    data = meta_data.merge(embeds, how='left', left_on='ID', right_on='ID')
    
    # Define target and features
    target = data[target_name]
    
    # define features
    features = data.iloc[:, meta_data.shape[1]:]
    res = run_regression(features, target, target_name, layer)
    results_df = pd.concat([results_df, res]).reset_index(drop=True)
     

results_df.to_csv(output_dir)

print('Process Finished')        
