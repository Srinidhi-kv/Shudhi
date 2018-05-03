import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from scipy.stats import zscore
from datetime import datetime
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, Normalizer, Imputer

def shudhi_describe(df_train, cols= [None], empty_missing= False, plot=True, target= None):
    """This modules describes the dataset and allows you to learn its properties.
    df_train-> Dataset
    cols-> columns, in a list
            Or None -> all columns
    
    empty_missing-> True is want "" to be considered as missing
    plot-> False to turn it off
    target(str, continuous column)-> plots continuous features against this
    """
#Define Global variables and Check entries
    
    if np.array_equal(cols, [None]):
        cols=df_train.columns
        
    if not isinstance(df_train, pd.DataFrame):
        return print("Error: Invalid entry for df_train field. Enter a pandas DataFrame")
        
    if not (set(cols) <= set(df_train.columns)):
        return print("Error: Invalid entry for cols field")
    
    if not isinstance(empty_missing, bool):
        return print("Error: Invalid entry for empty_missing field")
    
    if not isinstance(plot, bool):
        return print("Error: Invalid entry for plot field")

    if not (set([target]) <= set(df_train.columns).union(set([None]))):
        return print("Error: Invalid entry for target field")

# Set Global variable cols

    all_con_cols= [] #All continuous columns of Datset
    all_cat_cols= [] #All categorical columns of Datset
    
    con_cols=[] #All continuous columns in cols given
    cat_cols=[] #All categorical columns in cols given

#Find all categorical and continuous features in the dataset    

    for col in set(df_train.columns):
        if pd.api.types.is_numeric_dtype(df_train[col])== True:
            all_con_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df_train[col])== False:
            all_cat_cols.append(col)

    if (set(cols)==set(["all"])) | (set(cols) == set(df_train.columns)):
        cols= df_train.columns #HARD START

# Check and set "cols" variable
    if np.array_equal(cols, ["con"]):
        if not all_con_cols:
            return print("No continuous colums in the dataframe")
        elif all_con_cols:
            cols= all_con_cols
            con_cols= all_con_cols
    
    elif np.array_equal(cols, ["cat"]):
        if not all_cat_cols:
            return print("No categorical colums in the dataframe")
        elif all_cat_cols:
            cols= all_cat_cols
            cat_cols= all_cat_cols

    else:
        for col in set(cols):
            if pd.api.types.is_numeric_dtype(df_train[col])== True:
                con_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df_train[col])== False:
                cat_cols.append(col)
 #------------------------------------------------------------------------------------------------------------#    
    
#Identifier    

    def shudhi_identify(df_train, cols, convert=False):

        col_identity = []

        if convert:
            print("Warning: Type conversion takes a long time, O(mn)(m features, n rows)")

        for col in df_train.columns:

            if(df_train[col].dtype.kind=="i" or df_train[col].dtype.kind=="u"): 
                col_identity.append((col,"Integer"))

            elif(df_train[col].dtype.kind=="f"):
                col_identity.append((col,"Real Value"))

            elif(df_train[col].dtype.kind=="b"):
                col_identity.append((col,"Boolean"))

            elif(df_train[col].dtype.kind=="M"):
                col_identity.append((col,"Date/Time"))

            else:
                try:
                    #Choosing fraction of sample to test on -> to improve latency
                    #Worst case- 5% rows will have "bad" data and we want to be right >95% of the time
                    #So, if length<568, consider, all samples. Else, take 10% random sample
                    if len(df_train[col].dropna())< 568:
                        fraction=1
                    elif len(df_train[col].dropna())>= 568:
                        fraction=min(1000/len(df_train[col].dropna(), 1))
                    
                    temp = pd.to_numeric((df_train[col].dropna()).sample(frac=fraction, replace=True))
                    
                    if (temp.dtype.kind=="i"):
                        col_identity.append((col,"Integer saved as string"))

                        if(convert ==True):
                            df_train[col] = pd.to_numeric(df_train[col])

                    else:
                        col_identity.append((col,"Real Value saved as string"))

                        if(convert ==True):
                            df_train[col] = pd.to_numeric(df_train[col])

                except:            
                    try:
                        if len(df_train[col].dropna())< 568:
                            fraction=1
                        elif len(df_train[col].dropna())>= 568:
                            fraction= min(1000/len(df_train[col].dropna(), 1))

                        temp = pd.to_datetime((df_train[col].dropna()).sample(frac=fraction, replace=False))
                        col_identity.append((col,"Date/Time saved as string"))

                        if(convert ==True):
                            df_train[col] = pd.to_datetime(df_train[col])

                    except:
                        col_identity.append((col,"String/Object"))

        return pd.DataFrame(col_identity, columns = ['Feature', 'Feature Type'])


#----------------------------------------------------------
    
# Stats Module
    def shudhi_stats(df_train, cols, empty_missing):
        """Counts and output number of missing, non-missing values, #Unique values and other statistics of all features
           Provides the following statistics: Count, #Missing, #Unique, #Possible Outliers, Mean, Median, Mode, Min, Max
           Defult: shudhi_stats(cols= None, empty= False)"""

        warnings.filterwarnings("ignore")
        
        df_train= df_train[cols]
    #If empty is true, consider empty space/s as nan's
        if empty_missing:
            df_train[cols]= df_train[cols].replace(r'^\s*$', np.nan, regex=True)
        
    #Call identifier function defined above    
        df_identify= shudhi_identify(df_train, cols)
        
    #Counts for all features
        df_count = pd.DataFrame(columns =['Feature', 'count'])
        df_count['Feature']= df_train.columns
        df_count['count']= len(df_train.index)

        df_missing = df_train.isnull().sum().reset_index() #Identify empty/string?
        df_missing.columns = ['Feature', '# Missing']  

        df_unique= df_train.nunique().reset_index()
        df_unique.columns = ['Feature', '# Unique']

        df_count= pd.concat([df_identify, df_count['count'], df_unique['# Unique'], df_missing['# Missing']], axis=1)

    #Do the below for continuous features only
        if con_cols:

            #df_train1=df_train.dropna(axis=1, how='all')
            
            df_mean= df_train[con_cols].mean().reset_index().round(2)
            df_mean.columns = ['Feature', 'mean']

            df_median= df_train[con_cols].median().reset_index().round(2)
            df_median.columns = ['Feature', 'median']

            df_min= df_train[con_cols].min().reset_index()
            df_min.columns = ['Feature', 'min']

            df_max= df_train[con_cols].max().reset_index()
            df_max.columns = ['Feature', 'max']

            mean = df_train[con_cols].mean()
            std = df_train[con_cols].std()

            df_outlier= pd.DataFrame(((df_train[con_cols] < (mean - 2 * std)) | (df_train[con_cols]> (mean + 2* std))).sum()).reset_index()
            #df_outlier= df_train[con_cols].dropna()[(np.abs(zscore(df_train[con_cols].dropna())) > 2).all(axis=1)].count().reset_index()
            df_outlier.columns = ['Feature', '# Outliers']

            df_stats= pd.concat([df_outlier, df_mean['mean'], df_median['median'], df_min['min'], df_max['max']], axis=1)
            df_final = df_count.merge(df_stats, on='Feature', how='left')

        elif not con_cols:
            df_final = df_count

        df_final.fillna('', inplace=True)
              
        #Check out df_train.style!! https://pandas.pydata.org/pandas-docs/stable/style.html
        return df_final

    
#------------------------------------------------------------------------------------------------------------#    

#Shudhi Plots
    def shudhi_plots(df_train, cols, target, plot):
        """This module outputs univariate distributions of features, Target vs Feature plots and (optional)Feature vs Feature plots"""       

    #        fig.subplots_adjust(hspace = .5, wspace=.001)

    #        axs = axs.ravel()
        if not plot:
            return
        
        con_cols_nz=0
        for col in con_cols:
            if len(df_train[col].dropna())>0:
                con_cols_nz+=1
              
        if con_cols:
            i=1
            con_plot = int(np.ceil(con_cols_nz/2))
            plt.subplots(1, 2, figsize=(20, 6*con_plot))
            
            print("    \033[4mUNIVATIATE PLOTS\033[0m:"+ " \033[4mContinuous Features\033[0m")
            for col in set(con_cols):
                if len(df_train[col].dropna())>0:
                    plt.subplot(con_plot, 2, i) 
                    sns.distplot(df_train[col].dropna())
                    plt.title("Distribution of Feature: \""+ str(col)+"\"", fontsize=15)
                    plt.xlabel(col, fontsize=15)
                    plt.ylabel("count", fontsize=15)            
                    i+=1
            #plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()
        
        j=0
        for col in cat_cols:
            if df_train[col].nunique()<=20:
                j+=1
        
        if cat_cols:
            i=1
            cat_plot = int(np.ceil(j/2))
            plt.subplots(1, 2, figsize=(20, 6*cat_plot))
            
            print("    \033[4mUNIVATIATE PLOTS\033[0m:"+ " \033[4mCategorical Features\033[0m")
            
            for col in set(cat_cols):
                if df_train[col].nunique() <= 20:
                    plt.subplot(cat_plot, 2, i) 
                    sns.countplot(x=df_train[col])
                    plt.title("Distribution of Feature: \""+ str(col)+"\"",  fontsize=15, color='#0F0F0F')
                    plt.xlabel(col, fontsize=15)
                    plt.ylabel("count", fontsize=15)
                    i+=1
            #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.2, top=0.15)
            plt.show()

        if target and (set([target]) <= set(con_cols)):
            i=1
            con_plot = int(np.ceil(len(con_cols)/2))
            plt.subplots(1, 2, figsize=(20, 6*con_plot))
            
            print("    \033[4mBIVARIATE PLOTS with Target\033[0m")
            for col in con_cols:
                if not col == target and len(df_train[col].dropna())>0:
                    plt.subplot(con_plot, 2, i) 
                    sns.regplot(x=df_train[col], y=df_train[target], fit_reg= False, scatter_kws={'alpha':0.6})
                    plt.title("Target vs \""+ str(col)+"\"", fontsize=15)
                    plt.xlabel(col, fontsize=15)
                    plt.ylabel(target, fontsize=15)            
                    i+=1
            plt.show()
        
        if con_cols:
            plt.figure(figsize=(2*len(con_cols),2*len(con_cols)))
            cmap = sns.diverging_palette(240, 10, n=11, as_cmap=True)
            sns.heatmap(df_train.corr(), cmap=cmap, vmax=0.5, center=0,
                        square=True, linewidths=1, cbar_kws={"shrink": 0.5}, annot=True, linecolor='grey')
            plt.title("Correlation Matrix of continuous features", fontsize=15)
            plt.legend('Correlation')
            plt.show()
        return

#    start_time = time.time()

    print("\n                                       \033[4mSUMMARY STATISTICS\033[0m")
    display(shudhi_stats(df_train, cols, empty_missing))
    print("Note: Categorical columns will not have outliers/mean/median/min/max")
#    print("--- %s seconds ---" % (time.time() - start_time))
#    start_time = time.time()
    
    print("-"*116)
    
    if plot:
        print("\n                                             \033[4mPLOTS\033[0m\n")
    
        shudhi_plots(df_train, cols, target, plot)
#    print("--- %s seconds ---" % (time.time() - start_time))
    
    return

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

def shudhi_transform(df_train, df_test= None, cols= [None], missing_strategy=None, empty_missing= False, missing_override=False
                     , scale_strategy= None, outlier_strategy= None, one_hot= False, convert= False, imbalance_strategy= False):
    """A module to transform data to make it ML library readable. Treats missing values, outliers, converts the data type and 
    scales the data  does one hot encoding.
    Default: shudhi_transform(df_train, df_test= None, cols= [None], missing_strategy=None, empty_missing= False, missing_override=False
                     , scale_strategy= None, outlier_strategy= None, one_hot= False, convert= False)"""
    
    
#Define Global variables    
    if np.array_equal(cols, [None]):
        return print("Error: No value entered for cols field")
    
    if not isinstance(df_train, pd.DataFrame):
        return print("Error: Invalid entry for df_train field. Enter a pandas DataFrame")
    
    if not df_test== None:
        if not isinstance(df_test, pd.DataFrame):
            return print("Error: Invalid entry for df_train field. Enter a pandas DataFrame")
        if not np.array_equal(df_train.columns, df_test.columns):
            return print("Error: Invalid Test Set")
    
    if not set(cols) <= set(df_train.columns):
        return print("Error: Invalid entry for cols field")

    if not set([missing_strategy]) <= set([None, 'remove', 'mean', 'median', 'mode']):
        return print("Error: Invalid entry for missing_strategy field")
    
    if not isinstance(empty_missing, bool):
        return print("Error: Invalid entry for empty_missing field")
    
    if not isinstance(missing_override, bool):
        return print("Error: Invalid entry for missing_override field")

    if not set([scale_strategy]) <= set([None, 'std', 'robust', 'min_max', 'max_abs', 'norm']):
        return print("Error: Invalid entry for scale_strategy field")
    
    if not set([outlier_strategy]) <= set([None, 'remove', 'minmax', 'mean']):
        return print("Error: Invalid entry for outlier_strategy field")
    
    if not isinstance(one_hot, bool):
        return print("Error: Invalid entry for one_hot field")
    
    if not isinstance(convert, bool):
        return print("Error: Invalid entry for convert field")
    
    if not isinstance(imbalance_strategy, bool):
        return print("Error: Invalid entry for imbalance_strategy field")

# Set Global variable cols

    all_con_cols= [] #All continuous columns of Datset
    all_cat_cols= [] #All categorical columns of Datset
    
    con_cols=[] #All continuous columns in cols given
    cat_cols=[] #All categorical columns in cols given

#Find all categorical and continuous features in the dataset    

    for col in set(df_train.columns):
        if pd.api.types.is_numeric_dtype(df_train[col])== True:
            all_con_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df_train[col])== False:
            all_cat_cols.append(col)
        
#     if np.array_equal(cols, ["all"]) | (cols == df_train.columns):
#         cols= df_train.columns #HARD START

# # Check and set "cols" variable
#     elif np.array_equal(cols, ["con"]):
#         if not all_con_cols:
#             return print("No continuous colums in the dataframe")
#         elif all_con_cols:
#             cols= all_con_cols
#             con_cols= all_con_cols
    
#     elif np.array_equal(cols, ["cat"]):
#         if not all_cat_cols:
#             return print("No categorical colums in the dataframe")
#         elif all_cat_cols:
#             cols= all_cat_cols
#             cat_cols= all_cat_cols

#     else:
    for col in set(cols):
        if pd.api.types.is_numeric_dtype(df_train[col])== True:
            con_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df_train[col])== False:
            cat_cols.append(col)

#------------------------------------------------------------------------------------------------------------   
                
#Missing Value Treatment
    
    def shudhi_missing(df_train, df_test, cols, missing_strategy, empty_missing, missing_override):
        """Function to manipulate missing values of features. Operates inplace
           Allowed imputations are  'remove', 'mean', 'median' and 'mode'
           cols-> list of column names. "all" for all columns, "con" for continuous and "cat" for categorical features
           Default: shudhi_missing(df_test= None, cols=None, missing_strategy= None, empty= False, group_by= None) """

        if missing_strategy==None:
            return df_train, df_test

#         if (group_by != None) and (group_by not in df_train.columns):
#             return print("Error: Invalid entry for group_by")

        if (missing_strategy not in ['remove', 'mode']) and (cat_cols): #If categorical columns in cols
            cols= con_cols #Consider only continuous columns
            print("Warning: Entered inconsistent column types. Only Continuous features will be imputed by mean/median")

#If empty is true, replace empty space/s as nan's- get this in the main function
        if empty_missing:
            df_train[cols]=df_train[cols].replace(r'^\s*$', np.nan, regex=True)
            if not df_test == None:
                df_test[cols]=df_test[cols].replace(r'^\s*$', np.nan, regex=True)

        if not missing_override:
            new_cols=[]
            for col in df_train[cols].columns:
                if df_train[col].isnull().mean() <= 0.1:
                    new_cols.append(col)
            cols=new_cols
            
            print("Warning: If a column has >10% missing values, it will not be acted upon unless \"override=True\" is set. The below columns are considered:")
            print(cols)

    #If all checks pass, impute using the given strategy

    #Remove rows of missing values
#         if(group_by == None):

        if missing_strategy == 'remove':
            df_train.dropna(axis=0, how='any',subset=cols, inplace=True)

            if not df_test == None:
                df_test.dropna(axis=0, how='any',subset=cols, inplace=True)
            return df_train, df_test

#Impute by mean            
        elif missing_strategy == 'mean':

            imputer= Imputer(strategy='mean')
            df_train[cols]= imputer.fit_transform(df_train[cols])

            if not df_test == None:
                df_test[cols] = imputer.transform(df_test[cols])
                
            return df_train, df_test

#Impute by median                    
        elif missing_strategy == 'median':
            imputer= Imputer(strategy='mean')
            df_train[cols]= imputer.fit_transform(df_train[cols])

            if not df_test == None:
                df_test[cols] = imputer.transform(df_test[cols])
            return df_train, df_test

#Impute by mode                    
        elif missing_strategy == 'mode':

#             if False in table:
            for col in cols:
                df_train[col]= df_train[col].fillna(df_train[col].value_counts().index[0])
                if not df_test == None:
                    df_test[col]= df_test[col].fillna(df_train[col].value_counts().index[0])
            return df_train, df_test
    
#------------------------------------------------------------------------------------------------------------ 

# OUTLIER

    def shudhi_outlier(df_train, df_test, cols, outlier_strategy):
        """Does outlier treatment. Values beyond 2 standard deviations away from mean are considered outliers.
        shudhi_outlier(df_train, df_test, cols, outlier_strategy)"""
        
        if outlier_strategy==None:
            return df_train, df_test

       #Check if all columns are numeric
        if cat_cols: #If categorical columns in cols
            cols= con_cols #Consider only continuous columns
            print("Warning: Entered inconsistent column types. Only Continuous features will be outlier treated.")

        #If all checks pass, impute using the given strategy
        if df_train[cols].isnull().values.any():
            print("Error: One or more columns have Nan's, hence not outlier treated. Please do missing value treatment and come back later.")
            return df_train, df_test
        
        if outlier_strategy=="remove":
                  
            for col in cols:

                try:
                    df_train = df_train.loc[(np.abs(zscore(df_train[col]))<=2)]
                
                except Exception as e: print("Exception thrown \n ", e)
                    
            return df_train, df_test    
        
        elif outlier_strategy=="minmax":
                   
            for col in cols:

                try:
                    min_val = df_train[col].min()
                    max_val = df_train[col].max()
                    df_train.loc[zscore(df_train[col])<-2]= max_val
                    df_train.loc[zscore(df_train[col])>2]= min_val
                                
                except Exception as e: print("Exception thrown \n ", e)
            
            return df_train, df_test        
            
        elif outlier_strategy=="mean":
            
            for col in cols:

                try:
                    outlier_mean = df_train.loc[(np.abs(zscore(df_train[col]))<=2)][col].mean()
                    df_train.loc[(np.abs(zscore(df_train[col]))>2), col] = outlier_mean
                                
                except Exception as e: print("Exception thrown \n ", e)
            
            return df_train, df_test
 
#------------------------------------------------------------------------------------------------------------ 
    
    def shudhi_scaler(df_train, df_test, cols, scale_strategy):
        """Function to normalize/scale/standardize continuous features
           Allowed imputations are  'std', 'robust', 'min_max', 'max_abs' and 'norm'
           cols-> list of columns;
           Default: shudhi_scaler(df_train, df_test, cols, scale_strategy)"""

    # StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, Normalizer

        #Run checks on the the entered arguments
        if scale_strategy== None:
            return df_train, df_test

    #Check if all columns are numeric
        if cat_cols: #If categorical columns in cols
            cols= con_cols #Consider only continuous columns
            print("Warning: Entered inconsistent column types. Only Continuous features will be scaled.")

        #If all checks pass, impute using the given strategy
        if df_train[cols].isnull().values.any():
            print("Error: One or more columns have Nan's, hence not scaled Please do missing value treatment and come back later.")
            return df_train, df_test
        
        if scale_strategy=='std':
            scaler=StandardScaler(with_mean=True, with_std= True)
            df_train[cols]= scaler.fit_transform(df_train[cols])
            if not df_test == None:
                df_test[cols]= scaler.transform(df_test[cols])
            return df_train, df_test

        if scale_strategy=='robust':
            scaler=RobustScaler()
            df_train[cols]= scaler.fit_transform(df_train[cols])
            if not df_test == None:
                df_test[cols]= scaler.transform(df_test[cols])
            return df_train, df_test

        if scale_strategy=='min_max':
            scaler=MinMaxScaler()
            df_train[cols]= scaler.fit_transform(df_train[cols])
            if not df_test == None:
                df_test[cols]= scaler.transform(df_test[cols])
            return df_train, df_test

        if scale_strategy=='max_abs':
            scaler= MaxAbsScaler()
            df_train[cols]= scaler.fit_transform(df_train[cols])
            if not df_test == None:
                df_test[cols]= scaler.transform(df_test[cols])
            return df_train, df_test

        if scale_strategy=='norm':
            scaler=Normalizer()
            df_train[cols]= scaler.fit_transform(df_train[cols])
            if not df_test == None:
                df_test[cols]= scaler.transform(df_test[cols])
            return df_train, df_test

#--------------------------------------------------------------------------------------------------------------  

    def shudhi_converter(df_train, df_test, cols, convert):
        """Converts data types. shudhi_converter(df_train, df_test, cols, convert)"""

        col_identity = []
        
        if not convert:
            return df_train, df_test
        
        if convert:
            print("Warning: Type conversion takes a long time, O(mn)(m features, n rows)")

        for col in df_train.columns:

            if(df_train[col].dtype.kind=="i" or df_train[col].dtype.kind=="u"): 
                col_identity.append((col,"Integer"))

            elif(df_train[col].dtype.kind=="f"):
                col_identity.append((col,"Real Value"))

            elif(df_train[col].dtype.kind=="b"):
                col_identity.append((col,"Boolean"))

            elif(df_train[col].dtype.kind=="M"):
                col_identity.append((col,"Date/Time"))

            else:
                try:
                    #Choosing fraction of sample to test on -> to improve latency
                    #Worst case- 5% rows will have "bad" data and we want to be right >95% of the time
                    #So, if length<568, consider, all samples. Else, take 10% random sample

                    if len(df_train[col].dropna())< 568:
                        fraction=1
                    elif len(df_train[col].dropna())>= 568:
                        fraction=min(1000/len(df_train[col].dropna(), 1))

                    temp = pd.to_numeric((df_train[col].dropna()).sample(frac=fraction, replace=True))

                    if (temp.dtype.kind=="i"):
                        col_identity.append((col,"Integer saved as string"))

                        if(convert ==True):
                            df_train[col] = pd.to_numeric(df_train[col])

                    else:
                        col_identity.append((col,"Real Value saved as string"))

                        if(convert ==True):
                            df_train[col] = pd.to_numeric(df_train[col])

                except:            
                    try:
                        if len(df_train[col].dropna())< 568:
                            fraction=1
                        elif len(df_train[col].dropna())>= 568:
                            fraction= min(1000/len(df_train[col].dropna(), 1))

                        temp = pd.to_datetime((df_train[col].dropna()).sample(frac=fraction, replace=False))
                        col_identity.append((col,"Date/Time saved as string"))

                        if(convert ==True):
                            df_train[col] = pd.to_datetime(df_train[col])

                    except:
                        col_identity.append((col,"String/Object"))
        return df_train, df_test


#--------------------------------------------------------------------------------------------------------------  

    def shudhi_onehot(df_train, df_test, cols, one_hot):
        """Performs one hot encoding on the dataframe columns. shudhi_onehot(df_train, df_test, cols)"""

        if not one_hot:
            return df_train, df_test

        if con_cols: #If categorical columns in cols
            cols= cat_cols #Consider only continuous columns
            print("Warning: Entered inconsistent column types. Only Categorical features will be one hot encoded")

        try:
            df_train = pd.get_dummies(df_train, columns=cols)         
            if(df_test!= None):            
                df_test = pd.get_dummies(df_test, columns=cols)

                for col in df_train.columns:            
                    if col not in df_test.columns:
                        df_test[col]= 0

                for col in df_test.columns:            
                    if col not in df_train.columns:
                        df_test.drop(col, inplace=True)

                return df_train, df_test        
            
            return df_train, df_test        

        except Exception as e: print("Exception thrown \n ", e)

#--------------------------------------------------------------------------------------------------------------

    df_train, df_test = shudhi_missing(df_train, df_test, cols, missing_strategy, empty_missing, missing_override)
    df_train, df_test = shudhi_outlier(df_train, df_test, cols, outlier_strategy)
    df_train, df_test = shudhi_scaler(df_train, df_test, cols, scale_strategy)
    df_train, df_test = shudhi_converter(df_train, df_test, cols, convert)
    df_train, df_test = shudhi_onehot(df_train, df_test, cols, one_hot)
    
    return df_train, df_test
