# Viability and cell count prediction from CellProfiler and Harmony data, individual drugs and feature importance with RF

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from toxifate import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import plotly.express as px
import pickle
import os

viability_file = "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\plate reader\\CTG-fullWellNormresults.csv"

col_list = ['Viability','Count']

path_list = [
            "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\CellPainting project data\\CellProfiler\\PerWell\\blastsAllByWellProfiler.csv",
            "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\CellPainting project data\\CellProfiler\\PerWell\\tubesMonoByWellProfiler.csv",
            "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\CellPainting project data\\CellProfiler\\PerWell\\tubesPolyByWellProfiler.csv",
            "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\CellPainting project data\\Harmony\\Nv6-PerWell\\tubesProfiles.csv",
            "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\CellPainting project data\\Harmony\\Nv6-PerWell\\blastsProfiles.csv"]

def get_output_dir(individual_drug=False, prediction_col='Viability', file_type='cp', cell_type='TubesMono'):
    if individual_drug == True:
        if file_type == "cp":
            output_directory = "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\"+prediction_col+"Pred-models\\CellProfiler\\"+cell_type+"\\"
        elif file_type == "harmony":
            output_directory = "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\"+prediction_col+"Pred-models\\Harmony\\"+cell_type+"\\"
    else:
        output_directory = "C:\\Users\\Roman\\OneDrive - National University of Ireland, Galway\\"+prediction_col+"Pred-models\\WholePanel\\"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory
    
def training_loop(X, prediction_col='Viability', drug=None, output_directory=None, csv_file_name=None):

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    predictions = []

    if drug == None:
        X = well_profiles
        model_path = output_directory +  csv_file_name + '_RF_model.sav'
        predictions_path = output_directory + csv_file_name + '_predictions.csv'
        stats_table_path = output_directory + csv_file_name + '_stats_table.csv'
        train_data_path = output_directory + csv_file_name + '_X_train.csv'
        test_data_path = output_directory + csv_file_name + '_X_test.csv'
    else :
        X = well_profiles[well_profiles['Compound'] == drug]
        model_path = output_directory + drug + '_RF_model.sav'
        predictions_path = output_directory + drug + '_predictions.csv'
        stats_table_path = output_directory + drug + '_stats_table.csv'
        train_data_path = output_directory + drug + '_X_train.csv'
        test_data_path = output_directory + drug + '_X_test.csv'

    y = X[prediction_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.to_csv(train_data_path)
    X_test.to_csv(test_data_path)

    # set predictions to be the same as observations as placeholder, then replace with predictions
    predictions_df = pd.DataFrame({'Predictions': y_test, 'Observations': y_test, 'Compound': X_test['Compound'], 'Concentration': X_test['Concentration']})
    X_train = X_train.drop(columns=['Compound','Concentration', 'Viability','Count'])
    X_test = X_test.drop(columns=['Compound','Concentration', 'Viability','Count'])

    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    stats_table = pd.DataFrame({'MSE': [mse_train, mse_test], 'R2': [r2_train, r2_test]}, index=['Train', 'Test'])
    stats_table.to_csv(stats_table_path)

    predictions.append(y_pred_test)
    print("Drug %s, MSE: %f, R2: %f    " % (drug, mse_test, r2_test), end='\r')
    importances = rf.feature_importances_
    
    # Save predictions, observations, Compound and Concentration for later plotting
    predictions_df['Predictions'] = y_pred_test
    predictions_df.to_csv(predictions_path, index=False)
    pickle.dump(rf, open(model_path, 'wb')) #save rf model


    return r2_test, mse_test, rf, importances, X_test, y_pred_test


for csv_per_well in path_list:
    csv_file_name = csv_per_well.split("\\")[-1].split(".")[0]
    print(csv_file_name)

    for prediction_col in col_list:
        print(prediction_col, end='\r\n')
        
        individual_drug = True

        viability_table_per_well = pd.read_csv(viability_file) # read viability data
        phenotypic_data = pd.read_csv(csv_per_well)           # read phenotypic data
        
        phenotypic_data = phenotypic_data.replace(to_replace='HCQ', value='HYCQ')
        phenotypic_data = phenotypic_data.drop(columns=['Unnamed: 0'], errors='ignore')

        cell_type = get_cell_type(csv_file_name)
        file_type = get_file_type(phenotypic_data)

        ################

        # aggregate viability_table_per_well by Compound and Concentration, median, discard Row, Column, Unnamed: 0,Plate,Well
        viability_table = viability_table_per_well.groupby(['Compound', 'Concentration']).median(numeric_only=True).reset_index()
        viability_table = viability_table.drop(columns=['Row', 'Column', 'Unnamed: 0'], errors='ignore')


        # aggregate table_per_well by Compound and Concentration, median, discard Row and Column
        well_profiles = phenotypic_data.reset_index()
        well_profiles.drop(columns=['Row', 'Column'], inplace=True, errors='ignore')
        well_profiles = well_profiles.join(viability_table.set_index(['Compound', 'Concentration']), on=['Compound', 'Concentration'])
        well_profiles.rename(columns={'1s luciferase (CPS)':'Viability'}, inplace=True)
        well_profiles = well_profiles.dropna(axis=1)
        well_profiles = well_profiles[well_profiles.Compound != 'DMSO']         #filter out DMSO 
        well_profiles = drop_correlated_columns(well_profiles, 0.9, numeric=True)
        well_profiles.reset_index(drop=True, inplace=True)
        well_profiles = well_profiles.reset_index(drop=True).drop(columns=['index'], errors='ignore')#.sample(frac=1)

        ################
        if individual_drug == True:
            #individual models
            output_directory = get_output_dir(individual_drug=individual_drug, prediction_col=prediction_col, file_type=file_type, cell_type=cell_type)
            importance_list = []
            R2_list = []
            MSE_list = []

            for drug in well_profiles['Compound'].unique():
                r2, mse, rf, importances, X_test, y_pred = training_loop(well_profiles, prediction_col, drug=drug, output_directory=output_directory, csv_file_name=csv_file_name)
                importance_list.append(importances)
                R2_list.append(r2)
                MSE_list.append(mse)
            importance_df = pd.DataFrame(importance_list, columns=X_test.columns, index=well_profiles['Compound'].unique())
            stats_df = pd.DataFrame(R2_list, columns=['R2'], index=well_profiles['Compound'].unique())
            stats_df['MSE'] = MSE_list
            stats_path = output_directory + 'stats_df.csv'
            import_path = output_directory + 'importance_df.csv'
            stats_df.to_csv(stats_path)
            importance_df.to_csv(import_path)

        ################
        # whole panel model
        individual_drug = False
        output_directory = get_output_dir(individual_drug=individual_drug, prediction_col=prediction_col, file_type=file_type, cell_type=cell_type)
        importance_list = []
        R2_list = []
        MSE_list = []
    
        r2, mse, rf, importances, X_test, y_pred = training_loop(well_profiles, prediction_col, drug=None, output_directory=output_directory, csv_file_name=csv_file_name)

        importance_list.append(importances)
        R2_list.append(r2)
        MSE_list.append(mse)

        importance_df = pd.DataFrame(importance_list, columns=X_test.columns, index=well_profiles['Compound'].unique())
        stats_df = pd.DataFrame(R2_list, columns=['R2'])
        stats_df['MSE'] = MSE_list

        #save stats_df and importance_df
        output_directory = get_output_dir(individual_drug=False, prediction_col=prediction_col, file_type=file_type, cell_type=cell_type)
        stats_path = output_directory + csv_file_name + '_stats_df.csv'
        import_path = output_directory + csv_file_name + '_importance_df.csv'
        stats_df.to_csv(stats_path)
        importance_df.to_csv(import_path)
