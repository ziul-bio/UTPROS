#!/usr/bin/env python3 -u

# To run this script:
# python run_regression_all_layers.py -f input_file.txt

################ imports #####################
import os
import numpy as np
import pandas as pd

import scipy
import torch
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score

import argparse


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
    n_comp_list = [10, 20, 30, 50, 60, 90, 100, 200, 300, 500]
    
    # lists to store the results
    results_all = pd.DataFrame()
    
    for n_comp in n_comp_list:
        train_r2s = []
        train_maes = []
        train_mses = []
        train_rmses = []
        train_spearman_Coefs = []

        test_r2s = []
        test_maes = []
        test_mses = []
        test_rmses = []
        test_spearman_Coefs = []

        best_params = []
        
        # Define the KFold cross-validator
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
        for train_index, test_index in kf.split(features):
            
            # Split the data into training and testing sets using the indices from the KFold cross-validator
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            
            # Apply PCA only on the training set
            pca = PCA(n_comp)
            X_train_PCAs = pca.fit_transform(X_train)
            
            # Apply the same PCA transformation to the test set
            X_test_PCAs = pca.transform(X_test)
            
            # Create a list of models to test
            model = LinearRegression()
            
            # train in the whole training set
            model.fit(X_train_PCAs, y_train)
            
            
            # make predictions on the PCA-transformed test set
            y_pred = model.predict(X_test_PCAs)
            train_pred = model.predict(X_train_PCAs)


            # Model evaluation on test data set
            train_r2 = metrics.r2_score(y_test, y_pred)
            train_r2s.append(train_r2)
            train_mae = metrics.mean_absolute_error(y_test, y_pred)
            train_maes.append(train_mae)
            train_mse = metrics.mean_squared_error(y_test, y_pred)
            train_mses.append(train_mse)
            train_rmse = np.sqrt(train_mse)
            train_rmses.append(train_rmse)
            train_spearman_Coef = scipy.stats.spearmanr(y_test, y_pred)
            train_spearman_Coefs.append(train_spearman_Coef[0])


             # Model evaluation on test data set
            test_r2 = metrics.r2_score(y_test, y_pred)
            test_r2s.append(test_r2)
            test_mae = metrics.mean_absolute_error(y_test, y_pred)
            test_maes.append(test_mae)
            test_mse = metrics.mean_squared_error(y_test, y_pred)
            test_mses.append(test_mse)
            test_rmse = np.sqrt(test_mse)
            test_rmses.append(test_rmse)
            test_spearman_Coef = scipy.stats.spearmanr(y_test, y_pred)
            test_spearman_Coefs.append(test_spearman_Coef [0])

            model_name = 'LinearRegression'
     

        
        # Saving results
        res_dict = {}
        res_dict["Target_name"] = [target_name] * 10
        res_dict["Layer"] = [layer] * 10
        res_dict["N_components"] =  [n_comp] * 10
        res_dict["Model"] = model_name * 10
        
        res_dict["train_R2_score"] = train_r2s
        res_dict["train_MAE_score"] = train_maes
        res_dict["train_RMSE_score"] = train_rmses
        res_dict["train_spearman_Coef"] = train_spearman_Coefs

        res_dict["test_R2_score"] = test_r2s
        res_dict["test_MAE_score"] = test_maes
        res_dict["test_RMSE_score"] = test_rmses
        res_dict["test_spearman_Coef"] = test_spearman_Coefs

        # updating data frame with results
        results  = pd.DataFrame(res_dict).reset_index(drop=True)
        results_all = pd.concat([results_all, results], axis=0, ignore_index=True)

        # Print the mean scores
        print(f"Layer: {layer}, Num componemts: {n_comp}, Mean 10 splits train R2: {np.mean(train_r2s):.2f}, Mean 10 splits test R2: {np.mean(test_r2s):.2f}")

    return results_all


############################# Run Predictions #############################

def main():
    parser = argparse.ArgumentParser(description="Run regression for different target datasets and layers")
    parser.add_argument("-f", "--file", type=str, help="Path to the input file containing target data")

    args = parser.parse_args()

    # Check if the input file is provided
    if args.file:
        # Read the contents of the input file
        with open(args.file, "r") as file:
            lines = file.readlines()
        
            # Process each line to get the target data and other parameters
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    target_name, meta_data_file, embed_dir, output_dir = parts
                    # Load your metadata, embeddings, and other necessary data here...
                    meta_data = pd.read_csv(meta_data_file)
                    
                    # Call the run_regression function with the appropriate arguments
                    rep_layer = 48
                    results_df = pd.DataFrame()
                    for layer in range(0, rep_layer+1):
                        print(f'Running regression for {target_name} on layer {layer}')
                        embeds = load_mean_embeds(embed_dir, layer)
                        data = meta_data.merge(embeds, how='left', left_on='ID', right_on='ID')

                        # Define target and features
                        target = data[target_name]
                        # define features
                        features = data.iloc[:, meta_data.shape[1]:]

                        # run regression
                        res = run_regression(features, target, target_name, layer)
                        results_df = pd.concat([results_df, res]).reset_index(drop=True)
                
                    # Save or print the results_df DataFrame as needed...
                    results_df.to_csv(output_dir)
                    print(f'Process Finished for {target_name}')
                else:
                    print("Invalid input format. Each line should contain target_name, metadata_file, embeddings_file, and output_dir separated by tabs.")
    else:
        print("Input file not provided. Please specify an input file using the -f or --file option.")

if __name__ == "__main__":
    main()
