''' MLP project

main file

Paweł Józefczyk
Tomasz Gmiński

data file: 'cars.csv'
target_column: 'mpg' - miles per galon

As preliminary Project ML was used.
    - github repository: https://github.com/tgminski/Projekt_ML_final
    - files on folder: '\old_ML_procect_files\'

'''


# import libraries
import utilites_TG
import utilites_PJ

#from TG_ML_clustering_utilities import TG_data_normalize_stats, TG_clustering_ml
#from TG_ML_clustering_utilities import TG_ML_with_context

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


def main():

    pd.options.mode.chained_assignment = None  # default='warn'

    target = 'mpg'
    column_to_dummy_variable = 'origin'

#------------------------------------------------------------
# 1. Read raw data
#------------------------------------------------------------
    # 1.1 Import from network (Keaggle)

    # 1.2 Import from local file
    DATA_RAW = utilites_TG.load_data_from_csv('_src_data/cars.csv')   # pd.DataFrame
    DATA_RAW.drop(labels=['name'], axis=1, inplace=True)
    print('----------------------------------------')
    print('Data raw')
    print(DATA_RAW.head())
    print(DATA_RAW.info())


#------------------------------------------------------------
# 2. Check data 
#------------------------------------------------------------
    # 2.1 Data description and data check
        # 
        # https://github.com/tgminski/Projekt_ML_final/02_Projekt_ML_Data_preprocessing_part_1.ipynb

    

#------------------------------------------------------------
# 3. Explore data
#------------------------------------------------------------
    # 3.1 NaN checking
    data_no_nan = DATA_RAW.dropna().copy()   # formal, data not have nan
    # 3.2 Data distributions, transformations for normality
        #https://github.com/tgminski/Projekt_ML_final/02_Projekt_ML_Data_preprocessing_part_1.ipynb
    
    # 3.3 Categorical data: column 'origin'
    df_origin_dummy = pd.get_dummies(data_no_nan[column_to_dummy_variable], prefix=column_to_dummy_variable)
    data_no_nan = pd.concat([data_no_nan, df_origin_dummy], axis=1)
    data_no_nan = data_no_nan.drop(columns=[column_to_dummy_variable]).copy()  # usunięcie kolumny 'origin'
    print('----------------------------------------')
    print('Data with dummy variables')
    print(data_no_nan.info())


    # 3.4 Outlaiers selection (outlaiers in one variable context - statistic )
        # outliers from features statistics  +- 1.5*IQR
        # https://github.com/tgminski/Projekt_ML_final/02_Projekt_ML_Data_preprocessing_part_1.ipynb
        # https://github.com/tgminski/Projekt_ML_final/02_Projekt_ML_Data_preprocessing_part_2.ipynb
        
        # outliers in ML context (on part 4 and 6) 

    

#------------------------------------------------------------
# 4. ML modeling
#------------------------------------------------------------
    

    X = data_no_nan.drop([target], axis=1).copy()
    y = data_no_nan[[target]].copy()
    df_train_test = train_test_split(X, y, test_size=0.33, random_state=0)

    scaler_4_ML = StandardScaler
    model_4_ML = LinearRegression
    model_4_ML_args = {} # {'n_jobs':-1}
    

    # 4.1 Base model
    models = utilites_TG.models(df_train_test=df_train_test, data_index=data_no_nan.index)
    
    # base model - Linear regression with raw data
    models.fit_and_score_new_model(data=data_no_nan.copy(deep=True), target=target ,model_name='base_model', scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args)
    print('------------------------------------')
    print(models.results)

    #num_outliers = models.outliers_selection(data=data_no_nan, target=target)
    #print('------------------------------------')
    #print('Detected', num_outliers, 'outliers.') 


    # 4.2 base data trassformations   
    best_transformations, transformations_list = models.try_data_transformations_e(data=data_no_nan, target=target)
    print('------------------------------------')
    for key in best_transformations:
        print(key, best_transformations[key])
    print('------------------------------------')
    for temp in transformations_list:
        print(temp)
    
    # transform data and scoring model
    for i, transformation in enumerate(transformations_list):
        print('Testing transformation',i)
        data_transformed = models.data_transformation_e(data=data_no_nan, transformation=transformation)
        models.fit_and_score_new_model(data=data_transformed, target=target ,model_name='transformation_'+str(i), scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args, y_transf=transformation[target])
    print('------------------------------------')
    print(models.results)  

    best_transformation =  transformations_list[4] 
    #print(data_no_nan.info())
    data_transformed = models.data_transformation_e(data=data_no_nan, transformation=best_transformation)
    #print(data_transformed.info())

    models.fit_and_score_new_model(data=data_transformed, target=target ,model_name='transf', scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args, y_transf=transformation[target])

    # outliers selection ( 1.5*IQR on residuals)
    num_outliers = models.outliers_selection(data=data_transformed, target=target)
    print('------------------------------------')
    print('Detected', num_outliers, 'outliers.') 
    #print(models.X_train_outliers)

    models.fit_and_score_new_model(data=data_transformed, target=target ,model_name='transf_no_outliers', scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args, y_transf=transformation[target])
        
    # 4.3 Features checking
    
    df_features = models.new_features(data=data_transformed, target=target)
    print('------------------------------------')
    print(df_features.info())  
    models.fit_and_score_new_model(data=df_features, target=target ,model_name='features_all', scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args, y_transf=transformation[target])
    
    # transformations for features
    best_transformations1, transformations_list1 = models.try_data_transformations_e(data=df_features, target=target) 
    print('------------------------------------')
    print('Transformations for features')
    #for key in best_transformations1:
    #    print(key, best_transformations1[key])
    #print('------------------------------------')
    print('Transformations for features')
    #for temp in transformations_list1:
    print(transformations_list1[0])
    
    best_transformation1 =  transformations_list1[0] 
    df_features_transformed = models.data_transformation_e(data=df_features, transformation=best_transformation1)

    models.fit_and_score_new_model(data=df_features_transformed, target=target ,model_name='features_all_transf', scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args, y_transf=transformation[target])
    print('------------------------------------')
    print(models.results)  

    # 4.4 Features selection

    # Lasso
    lasso_result, lasso_best_featurtes = models.features_choose_lasso(data=df_features_transformed, target=target)
    print('------------------------------------')
    print('Lasso best_featurtes:',lasso_best_featurtes)
    print('------------------------------------')
    lasso_best_featurtes.append( target )   # uzupełnienie kolumny o target
    df_lasso_features = df_features_transformed[lasso_best_featurtes].copy()
    models.fit_and_score_new_model(data=df_lasso_features, target=target ,model_name='Lasso_features', scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args, y_transf=transformation[target])
    print(models.results) 
    

    # PCA
    df_data_pca, result_choice, best_n_components = models.features_choose_PCA(data=df_features_transformed, target=target)
    print('------------------------------------')
    print(models.results)

    # 4.6 Outlaiers selection (outlaiers in ML context)
    # 4.6 Model training
    assert 0 , 'Manual Break'
#------------------------------------------------------------
# 5. ML process optimisation
#------------------------------------------------------------

#------------------------------------------------------------
# 6. Special functions
#------------------------------------------------------------
    # 6.1 ML using in ML function context
        # 6.1.1 Data clustering (in ML context)
        # 6.1.2 Outliers selection (in ML context)
        # 6.1.3 ML modeling (in ML context)

#------------------------------------------------------------
# 7. Conclusions
#------------------------------------------------------------


    return []





if __name__ == '__main__':
    main()










