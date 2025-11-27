''' utilites file TG '''

#------------------------------------------------------------
# import libraries
#------------------------------------------------------------

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

from sklearn.metrics import root_mean_squared_error


#------------------------------------------------------------
# clases
#------------------------------------------------------------


class models():

    def __init__(self, df_train_test=None, data_index=None):
        assert type(df_train_test)==type([1]), 'data must be type list'
        assert type(df_train_test[0])==type(pd.DataFrame()), 'data[0] must be pandas DataFrame'
        assert type(df_train_test[1])==type(pd.DataFrame()), 'data[1] must be pandas DataFrame'
        assert type(df_train_test[2])==type(pd.DataFrame()), 'data[2] must be pandas DataFrame'
        assert type(df_train_test[3])==type(pd.DataFrame()), 'data[3] must be pandas DataFrame'
        assert type(data_index)==type(pd.DataFrame().index), 'data_index must be pandas DataFrame.index'
        self.X_train, self.X_test, self.y_train, self.y_test = df_train_test
        self.data_index = data_index.copy(deep=True)

        assert self.X_train.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'
        assert self.X_test.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'
        assert self.y_train.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'
        assert self.y_test.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'

        self.results = pd.DataFrame(columns=['model_name','model','score_train','score_test','rmse_train','rmse_test'])

        self.transformations = {   
            'x'  : lambda x: x ,
            'x**2' : lambda x: x**2 ,
            'x**3' : lambda x: x**3 ,
            'x**(1/2)' : lambda x: x**(1/2) ,
            'x**(1/3)' : lambda x: x**(1/3) ,
            '1/x' : lambda x: 1/x ,
            '1/x2' : lambda x: 1/(x**2) ,
            'log(x)' : lambda x: np.log(x) 
            #'exp(x)' : lambda x: np.exp(x) 
                    }
        
        self.inwerse_transformations = {   
            'x'  : lambda x: x ,
            'x**2' : lambda x: x**(1/2) ,
            'x**3' : lambda x: x**(1/3) ,
            'x**(1/2)' : lambda x: x**(2) ,
            'x**(1/3)' : lambda x: x**(3) ,
            '1/x' : lambda x: 1/x ,
            '1/x2' : lambda x: 1/(x**(1/2)) ,
            'log(x)' : lambda x: np.exp(x) 
            #'exp(x)' : lambda x: np.exp(x) 
                    }
        
        self.transformations_e = {   
            'x'  : lambda x: x ,
            'exp(x)' : lambda x: np.exp(x) ,
            'exp(exp(x))' : lambda x: np.exp(np.exp(x)) ,
            '1/exp(x)' : lambda x: 1/np.exp(x) ,
            '1/exp(exp(x))' : lambda x: 1/np.exp(np.exp(x)) 
                    }
        
        self.inwerse_transformations_e = {   
            'x'  : lambda x: x ,
            'exp(x)' : lambda x: 1/np.exp(x) ,
            'exp(exp(x))' : lambda x: 1/np.exp(np.exp(x)) ,
            '1/exp(x)' : lambda x: np.exp(x) ,
            '1/exp(exp(x))' : lambda x: np.exp(np.exp(x)) 
                    }
        
        self.features = {   
            #'x'  : lambda x,y: x ,
            'x+y' : lambda x,y: x+y ,
            'x-y' : lambda x,y: x-y ,
            'x*y' : lambda x,y: x*y 
            #'1/(x*y)' : lambda x,y: 1/(x*y)
                    }
        

    def fit_and_score_new_model(self, data=None, target=None, model_name=[], scaler=StandardScaler, model=LinearRegression, model_arg={}, y_transf='x'):
        
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'
        
        # copy by initial splitting (DataFrame indexes)
        X_train = data.loc[self.X_train.index,:].drop(target,axis=1).copy(deep=True)
        X_test = data.loc[self.X_test.index,:].drop(target,axis=1).copy(deep=True)
        y_train = data.loc[self.y_train.index,[target]].copy(deep=True)
        y_test = data.loc[self.y_test.index,[target]].copy(deep=True)

        self.fit_last_feature_names = X_train.columns.tolist()
        self.fit_last_target_names = y_train.columns.tolist()

        scaler_X = scaler()
        scaler_y = scaler()
        X_train_scaled = scaler_X.fit_transform(X_train.to_numpy())
        y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape((-1,1)))
        X_test_scaled = scaler_X.transform(X_test.to_numpy())
        y_test_scaled = scaler_y.transform(y_test.to_numpy().reshape((-1,1)))

        model = model(**model_arg)
        model.fit(X_train_scaled,y_train_scaled)
        score_base_train = model.score(X_train_scaled,y_train_scaled)
        score_base_test = model.score(X_test_scaled,y_test_scaled)
        #y_predict_train =  scaler_y.inverse_transform( model.predict(X_train_scaled).reshape((-1,1)) )#.reshape((-1,))
        #y_predict_test =  scaler_y.inverse_transform( model.predict(X_test_scaled).reshape((-1,1)) )#.reshape((-1,))
        y_predict_train_scaled =  model.predict(X_train_scaled)
        y_predict_test_scaled =  model.predict(X_test_scaled)

        self._fit_last_residuals = np.subtract(y_train_scaled, y_predict_train_scaled)

        #rmse_train = root_mean_squared_error(y_true=y_train.to_numpy().reshape((-1,1)), y_pred=y_predict_train)
        #rmse_test = root_mean_squared_error(y_true=y_test.to_numpy().reshape((-1,1)), y_pred=y_predict_test)
        rmse_train = root_mean_squared_error(y_true=y_train_scaled, y_pred=y_predict_train_scaled)
        rmse_test = root_mean_squared_error(y_true=y_test_scaled, y_pred=y_predict_test_scaled)
        
        result_index = len(self.results)
        self.results.loc[result_index,['model_name','model']] = [model_name, model]
        self.results.loc[result_index,['score_train','score_test','rmse_train','rmse_test']] = [score_base_train, score_base_test, rmse_train, rmse_test]


    def  try_data_transformations(self, data=None, target=None):
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'

        # copy by initial splitting (DataFrame indexes)
        X_train = data.loc[self.X_train.index,:].drop(target,axis=1).copy(deep=True)
        y_train = data.loc[self.y_train.index,[target]].copy(deep=True)
        
        y_train_transformed = pd.DataFrame(index=self.y_train.index)
        for trans_name, trans_lambda in self.transformations.items():
            col_name = self.y_train.columns.tolist()[0] + ' transform ' + trans_name
            y_train_transformed[col_name] = trans_lambda(y_train)
        
        best_transformations = {}
        for col_y in y_train_transformed.columns:
            list_best_results = []
            for col_X in X_train.columns:
                #print(self.X_train[col_X].dtype)
                if X_train[col_X].dtype == bool: continue
                results = {}
                X_train_temp_trans = pd.DataFrame(index=self.X_train.index)
                for trans_name, trans_lambda in self.transformations.items(): 
                    col_name = col_X + ' transform ' + trans_name
                    try:
                        old_settings = np.seterr(all='raise')
                        temp_try = trans_lambda(X_train[col_X])
                        np.seterr(**old_settings)
                        if np.isinf(temp_try.to_numpy()).sum() > 0:
                            #print('----transf contains Inf')
                            continue
                        if np.isnan(temp_try.to_numpy()).sum() > 0:
                            continue
                    except:
                        #print('----transf except')
                        continue
                    else:
                        #print('----compute')
                        
                        X_train_temp_trans[col_name] = temp_try
                        scaler_X = StandardScaler()
                        scaler_y = StandardScaler()
                        X = scaler_X.fit_transform(X_train_temp_trans[[col_name]].to_numpy().reshape((-1,1)))
                        y = scaler_y.fit_transform(y_train_transformed[[col_y]].to_numpy().reshape((-1,1)))
                        model = LinearRegression(n_jobs=-1)
                        model.fit(X,y)
                        y_pred = model.predict(X)
                        rmse = root_mean_squared_error(y_true=y, y_pred=y_pred)
                        item_name = col_name
                        results[rmse] = item_name
                best_result = sorted(results)[0]
                list_best_results.append( results[best_result] )
            best_transformations[col_y] = list_best_results
        transformations_list = []
        for key in best_transformations:
            temp_dict = {}
            key_split = key.split()
            temp_dict[key_split[0]] = key_split[-1] 
            for key1 in best_transformations[key]:
                key1_split = key1.split()
                temp_dict[key1_split[0]] = key1_split[-1] 
            transformations_list.append (temp_dict)
        
        return best_transformations, transformations_list
    
    def data_transformation(self, data=None, transformation=None):
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'
        data_transformed = pd.DataFrame(index=data.index)
        for col in data.columns:
            if col in transformation.keys():
                data_transformed[col] = self.transformations[transformation[col]](data[col])
                #print('On column:', col, ' done transformation:', transformation[col])
            else:
                data_transformed[col] = data[col]

        return data_transformed.copy(deep=True)
    
    def  try_data_transformations_e(self, data=None, target=None):
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'

        # copy by initial splitting (DataFrame indexes)
        X_train_ = data.loc[self.X_train.index,:].drop(target,axis=1).copy(deep=True)
        y_train_ = data.loc[self.y_train.index,[target]].copy(deep=True)

        scaler_X_ = StandardScaler()
        scaler_y_ = StandardScaler()
        X_train = scaler_X_.fit_transform(X_train_.to_numpy())
        y_train = scaler_y_.fit_transform(y_train_.to_numpy().reshape((-1,1)))
        
        y_train_transformed = pd.DataFrame(index=self.y_train.index)
        for trans_name, trans_lambda in self.transformations_e.items():
            col_name = y_train_.columns.tolist()[0] + ' transform ' + trans_name
            y_train_transformed[col_name] = trans_lambda(y_train)
        
        best_transformations = {}
        for col_y in y_train_transformed.columns:
            list_best_results = []
            for col_X_num, col_X in enumerate(X_train_.columns):
                #print(self.X_train[col_X].dtype)
                if X_train_[col_X].dtype == bool: continue
                results = {}
                X_train_temp_trans = pd.DataFrame(index=X_train_.index)
                for trans_name, trans_lambda in self.transformations_e.items(): 
                    #print(col_y, col_X, trans_name)
                    col_name = col_X + ' transform ' + trans_name
                    temp_try = trans_lambda(X_train[:,col_X_num])
                    X_train_temp_trans[col_name] = temp_try
                    scaler_X = StandardScaler()
                    scaler_y = StandardScaler()
                    X = scaler_X.fit_transform(X_train_temp_trans[[col_name]].to_numpy().reshape((-1,1)))
                    y = scaler_y.fit_transform(y_train_transformed[[col_y]].to_numpy().reshape((-1,1)))
                    model = LinearRegression(n_jobs=-1)
                    model.fit(X,y)
                    y_pred = model.predict(X)
                    rmse = root_mean_squared_error(y_true=y, y_pred=y_pred)
                    item_name = col_name
                    results[rmse] = item_name
                best_result = sorted(results)[0]
                list_best_results.append( results[best_result] )
            best_transformations[col_y] = list_best_results

        transformations_list = []
        for key in best_transformations:
            temp_dict = {}
            key_split = key.split()
            temp_dict[key_split[0]] = key_split[-1] 
            for key1 in best_transformations[key]:
                key1_split = key1.split()
                temp_dict[key1_split[0]] = key1_split[-1] 
            transformations_list.append (temp_dict)

        return best_transformations, transformations_list
    
    def data_transformation_e(self, data=None, transformation=None):
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.to_numpy())

        data_transformed = pd.DataFrame(index=data.index)
        for col_num, col in enumerate(data.columns):
            if col in transformation.keys():
                data_transformed.loc[:,col] = self.transformations_e[transformation[col]](data_scaled[:,col_num])
            else:
                data_transformed.loc[:,col] = data[col].copy()
        
        assert (data_transformed.index==data.index).all() ,  'Function data_transformation_e  Niezgodne indexy '

        return data_transformed.copy(deep=True)

# 4.2 Features 
    def new_features(self, data=None, target=None):
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'
        
        df_features = data.copy(deep=True)
        data_ = data.drop(target,axis=1)
        scaler = StandardScaler()
        data_np = scaler.fit_transform(data_.to_numpy())

        col_names = data_.columns.tolist()
        for col_x_num, col_x in enumerate(col_names[:-1]):
            for col_y_num, col_y in enumerate(col_names[col_x_num+1:]):
                for feature in self.features.keys():
                    col_name = col_x + '_' + col_y + '_' + feature
                    df_features[col_name] = self.features[feature]( data_np[:,col_x_num], data_np[:,col_y_num] )

        assert (df_features.index==data.index).all() ,  'Function new_features  Niezgodne indexy '
        assert df_features.index.isin(self.data_index).all(), 'Function new_features Niezgodne indexy'

        return df_features.copy(deep=True)

    # 4.3 Features selection
    def features_choose_lasso(self, data=[], target=[]):
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'

        scaler_4_ML = StandardScaler
        model_4_ML = Lasso

        result = pd.DataFrame()
        result_choice = pd.DataFrame()
        alpha_list = [ 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1  ]
        for alpha in alpha_list:
            #print('Lasso alpha=', alpha)
            model_4_ML_args = {'alpha':alpha}
            self.fit_and_score_new_model(data=data, target=target ,model_name='Lasso alpha='+str(alpha), scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args)
            model_lasso = self.results['model'].iloc[-1]
            result.index = self.fit_last_feature_names
            col_name = 'alfa ='+str(alpha)
            result[col_name] = model_lasso.coef_
            features_count = result[col_name].count()
            features_non_zero = result[col_name].eq(0).sum()
            score_train = self.results['score_train'].iloc[-1]
            result_choice.loc[alpha, ['features_count', 'features_zero', 'score_train']] = features_count , features_non_zero, score_train
            result_choice.loc[alpha, 'features_reduction'] = result_choice.loc[alpha, 'features_zero']/result_choice.loc[alpha, 'features_count']
            #result_choice.loc[alpha, 'coef2'] = result_choice.loc[alpha, 'score_train']
            result_choice.loc[alpha, 'sum reduction + score'] = result_choice.loc[alpha, 'features_reduction'] + result_choice.loc[alpha, 'score_train']

            

            #print(model_lasso.coef_)
        
        

        print(result_choice)
        best_alpha = result_choice.sort_values(by='sum reduction + score', ascending=False).index[0]
        print(best_alpha)
        col_name ='alfa ='+str(best_alpha)
        best_featurtes = result[col_name][result[col_name].ne(0)].index.to_list()

        #sns.lineplot(data=result_choice, y=['coef1', 'coef2', 'coef3'])
        result_choice['features_reduction'].plot()
        result_choice['score_train'].plot()
        result_choice['sum reduction + score'].plot()
        plt.xlabel('alfa')
        plt.legend()
        plt.show()

        print(result)
        print('features counts:')
        #result_abs = result.abs()
        #print((result_abs>0.01).sum())
        print((result!=0).sum())

        return result, best_featurtes
    
    def features_choose_PCA(self, data=None, target=None):
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'

        # copy by initial splitting (DataFrame indexes)
        X_train = data.loc[self.X_train.index,:].drop(target,axis=1).copy(deep=True)
        #X_test = data.loc[self.X_test.index,:].drop(target,axis=1).copy(deep=True)
        #y_train = data.loc[self.y_train.index,[target]].copy(deep=True)
        #y_test = data.loc[self.y_test.index,[target]].copy(deep=True)

        feature_names = X_train.columns.tolist()
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.to_numpy())
        pca_selector = PCA(n_components=len(feature_names))
        X_train_pca_transform = pca_selector.fit_transform(X_train_scaled)


        print('PCA explained_variance_ratio:', pca_selector.explained_variance_ratio_)
        print('PCA singular_values:', pca_selector.singular_values_)
        temp = pd.DataFrame(pca_selector.explained_variance_ratio_)
        temp.plot(logy=True, title='explained_variance_ratio', xlabel='features')
        plt.show()
        temp1 = pd.DataFrame(pca_selector.singular_values_)
        temp1.plot(logy=True, title='singular_values', xlabel='features')
        plt.show()

        scaler_4_ML = StandardScaler
        model_4_ML = LinearRegression
        model_4_ML_args = {}

        result_variance_ratio = pd.DataFrame()
        result_signular_values = pd.DataFrame()
        result_choice = pd.DataFrame()
        n_components_list = [ 1, 3, 5, 10, 20, 30, 40, 50, 60, 100  ]
        for n_components in n_components_list:
            pca_selector = PCA(n_components=n_components)
            pca_selector.fit(X_train_scaled)
            data_ = data.drop(target,axis=1).copy(deep=True)
            #scaler_data = StandardScaler()
            data_scaled = scaler.transform(data_.to_numpy())
            data_pca = pca_selector.transform(data_scaled)
            df_data_pca = pd.DataFrame(data_pca, index=data.index)
            df_data_pca[target] = data[target].copy()
            self.fit_and_score_new_model(data=df_data_pca, target=target ,model_name='PCA_features '+str(n_components), scaler=scaler_4_ML, model=model_4_ML, model_arg=model_4_ML_args) #, y_transf=transformation[target])
            col_name = 'n_components ='+str(n_components)
            #result_variance_ratio[col_name] = pca_selector.explained_variance_ratio_
            #result_signular_values[col_name] = pca_selector.singular_values_
            features_count = len(data_.columns)
            features_non_zero = n_components
            score_train = self.results['score_train'].iloc[-1]
            result_choice.loc[n_components, ['features_count', 'features_non_zero', 'score_train']] = features_count , features_non_zero, score_train
            result_choice.loc[n_components, 'features_reduction'] = result_choice.loc[n_components, 'features_non_zero']/result_choice.loc[n_components, 'features_count']
            #result_choice.loc[alpha, 'coef2'] = result_choice.loc[alpha, 'score_train']
            result_choice.loc[n_components, '1/features_count + score'] = 1/result_choice.loc[n_components, 'features_non_zero'] +  result_choice.loc[n_components, 'score_train']
            
        print(result_choice)
        best_n_components = result_choice.sort_values(by='1/features_count + score', ascending=False).index[0]
        best_n_components = 20
        print(best_n_components)
        #col_name ='alfa ='+str(n_components)
        #best_featurtes = result_signular_values[col_name][result[col_name].notna()].index.to_list()

        # data for best pca
        pca_selector = PCA(n_components=20)
        pca_selector.fit(X_train_scaled)
        data_ = data.drop(target,axis=1).copy(deep=True)
        #scaler_data = StandardScaler()
        data_scaled = scaler.transform(data_.to_numpy())
        data_pca = pca_selector.transform(data_scaled)
        df_data_pca = pd.DataFrame(data_pca, index=data.index)
        df_data_pca[target] = data[target].copy()

        #sns.lineplot(data=result_choice, y=['coef1', 'coef2', 'coef3'])
        #result_choice['features_count'].plot()
        result_choice['score_train'].plot()
        result_choice['1/features_count + score'].plot()
        plt.xlabel('alfa')
        plt.legend()
        plt.show()

        return df_data_pca, result_choice, best_n_components

    def outliers_selection(self, data=[], target=None):
        assert type(data)==type(pd.DataFrame()), 'data must be pandas Dataframe'
        assert data.index.isin(self.data_index).all(), 'Index of X_train must be in Data_index'

        scaler_4_ML = StandardScaler
        model_4_ML = LinearRegression
        self.fit_and_score_new_model(data=data, target=target ,model_name='Outliers selection', scaler=scaler_4_ML, model=model_4_ML) #, model_arg=model_4_ML_args)

        data_temp = pd.DataFrame(index=self.X_train.index)
        data_temp['residuals'] = self._fit_last_residuals
        Q1 = data_temp['residuals'].quantile(0.25)
        Q3 = data_temp['residuals'].quantile(0.75)
        res_low = Q1 - 1.5*(Q3-Q1)
        res_high = Q3 + 1.5*(Q3-Q1)
        out_filter = (data_temp['residuals']<res_low) | (data_temp['residuals']>res_high)

        assert data_temp.notna().all().to_numpy() , 'Function outliers_selection  nan w data_temp'
        assert len(data_temp)==len(self.X_train), 'Function outliers_selection  niezgodny rozmiar data_temp'
        assert (data_temp.index==self.X_train.index).all() ,  'Function outliers_selection  Niezgodne indexy '


        # self.X_train_outliers = self.X_train.loc[out_filter,:].copy(deep=True)
        # self.X_train = self.X_train.loc[~out_filter,:].copy(deep=True)
        # self.y_train = self.y_train.loc[~out_filter,:].copy(deep=True)
        
        # self.X_test = self.X_test.copy(deep=True)
        # self.y_train = self.y_train.copy(deep=True)

        return out_filter.sum()

#------------------------------------------------------------
# functions
#------------------------------------------------------------

def load_data_from_csv(file):
    data = pd.read_csv(file)
    return data

def save_df_data_to_csv(data, file):
    data.to_csv(file, index=False)
    return []
