import gc
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in [x for x in df.columns if 'NU_NOTA_' not in x]:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
   # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    
 
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
    
    
## --- Handy user-defined functions


def display_feat_import(X_values, y_values):
    
    """
        Display a grid containing 4 2D plots regarding feature importance
        of a dichotomous classification problem. It uses two different
        algorithms: Random Forest, and PCA.
        
        Further, it displays the cumulative explained variance given by 
        the PCA algorithm, and gives the linear combination coefficients
        from the constructed principal components.
        
        inputs: - X_values (a 2D array containing features):        pandas.DataFrame
        ------- - y_values (a 1D array containing target classes):  pandas.Series
                - ax (a matplotlib axis grid to plot figures on):   matplotlib.axes._subplots
                
        outputs: - loadings (coefficients of the linear combination
        --------            of the original variables from which the 
                            principal components are constructed):   pandas.DataFrame
    """
        
    scaler = StandardScaler()
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X_values, y_values)
            

    importances = pd.DataFrame(data={
        'Attribute': X_values.columns,
        'Importance': clf[1].feature_importances_})
        
    importances = importances.sort_values(by='Importance', ascending=False)
            
    return importances['Attribute']
