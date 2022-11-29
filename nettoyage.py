import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import urllib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype



def nettoyage(df,pct_seuil_col, pct_seuil_row,itimp,higher_percentile,lower_percentile): 
    #pct_seuil_col : seuil de remplissage pour les colonnes en %
    #pct_seuil_row : seuil de remplissage pour les lignes en %
    #itimp : Iterative Imputation for numeric values, 1 si inactif, 0 sinon
    #higher_percentile : seuil haut pour le capping
    
    
    
    #VALEURS MANQUANTES COLONNES
    print('VALEURS MANQUANTES SUR COLONNES')
      
    #On collecte le nombre de valeurs manquanets par variable
    nb_na = df.isnull().sum() 
    print ('Le dataframe comporte',df.shape[0],'lignes et',df.shape[1],'colonnes, ainsi que',nb_na.sum(),'valeurs manquantes.' )

    #On calcule le pourcentage de valeurs manquantes par variables
    index = df.index
    pct_remplissage = pd.DataFrame((100-((nb_na/len(index))*100)).astype(int), columns=['Pourcentage de remplissage (%)']) 

    #On trace les pourcentages de valeurs manquantes
    sns.set_style("whitegrid")
    pct_remplissage.sort_values('Pourcentage de remplissage (%)', ascending=False).plot.barh(x=None, y=None,xlim = [0,100],legend=False,figsize = (13,13))
    plt.title("Pourcentage de remplissage (%)",fontsize = 25)
    plt.xticks(rotation=30, horizontalalignment="center",fontsize = 15)
    plt.yticks(fontsize = 15)
    line_seuil = plt.axvline(x=pct_seuil_col, color='r', linestyle='-')
    plt.legend([line_seuil],['Seuil de remplissage'],prop={"size":15})
    plt.show()
    
    #on retire les variables avec un taux de remplissage inf au seuil
    vardrop = pct_remplissage.loc[pct_remplissage['Pourcentage de remplissage (%)'] < pct_seuil_col]
    print(len(vardrop),' variables ont un taux de remplissage sous',pct_seuil_col,'%.')
    df = df.drop(columns=[col for col in df if col in pct_remplissage.loc[pct_remplissage['Pourcentage de remplissage (%)'] < pct_seuil_col].index])
    
    
    #VALEURS MANQUANTES LIGNES
    print('----------')
    print('   ')
    print('VALEURS MANQUANTES SUR LIGNES')
    
    #On collecte le nombre de valeurs manquanets par variable
    nb_na_row = df.isnull().sum(axis=1) 
    print ('Le dataframe comporte désormais',df.shape[0],'lignes et',df.shape[1],'colonnes, ainsi que',nb_na_row.sum(),'valeurs manquantes.' )

    #On calcule le pourcentage de valeurs manquantes par lignes
    columns = df.columns
    pct_remplissage_row = pd.DataFrame((100-((nb_na_row/len(columns))*100)).astype(int), columns=['Pourcentage de remplissage (%)']) 
        
    
    #on retire les lignes avec un taux de remplissage inf au seuil
    rowdrop = pct_remplissage_row.loc[pct_remplissage_row['Pourcentage de remplissage (%)'] < pct_seuil_row]
    print(len(rowdrop),' lignes ont un taux de remplissage sous',pct_seuil_row,'%.')
    df = df.drop(pct_remplissage_row.loc[pct_remplissage_row['Pourcentage de remplissage (%)'] < pct_seuil_row].index)
    
    
    
    
    #TRAITEMENT VALEURS MANQUANTES
    str_ = list(df.select_dtypes(include=['O']).columns)
    num_ = list(df._get_numeric_data())
    
    for column in str_:
        df[column] = df[column].fillna(df[column].mode().iloc[0])
        
    if itimp == 0:
        
        for column in num_:
            med = df[column].median()
            df[column] = df[column].fillna(med)
            
            
    if itimp == 1 :
        
        #Variables _n
        df_n = df[num_].filter(regex='_n')
        for column in df_n:
            med = df_n[column].median()
            df_n[column] = df_n[column].fillna(med)
        
        df.drop(labels=df_n.columns, axis="columns", inplace=True)
        df[df_n.columns] = df_n[df_n.columns]

        
        #Variables _100g
        df_100g = df[num_].filter(regex='_100g')
        imputer = IterativeImputer(sample_posterior=True,tol = 0.1,max_iter = 10,random_state=0,n_nearest_features=None, imputation_order='ascending',missing_values = np.nan)
        imputed = imputer.fit_transform(df_100g)
        df_imputed = pd.DataFrame(imputed, columns=df_100g.columns)   
        #round(df_imputed, 2) 
        
        df.drop(labels=df_100g.columns, axis="columns", inplace=True)
        df[df_100g.columns] = df_imputed[df_100g.columns]
        
           
     
        
        
    #TRAITEMENT DES DOUBLONS
    print('----------')
    print('   ')
    print('DOUBLONS')
    
    dup = df.loc[df['code'].duplicated(keep=False),:]
    print('Le dataframe comporte',len(dup),'doublons.')
    if len(dup) != 0:
        df = df.drop_duplicates().reset_index(drop=True)

   
    
    
    #VALEURS ABERRANTES
    print('----------')
    print('   ')
    print('VALEURS ABERRANTES')
    print('   ')
    print('---Avant traitement---')
    display(df.describe())
    
    num_ = list(df._get_numeric_data())
    df_num = df[num_]
    
    #PLOT
    #pour trier par médiane
    #meds_norm = df_norm.median()
    #meds_norm.sort_values(ascending=False, inplace=True)
    #df_norm = df_norm[meds_norm.index]
    
    #sns.set_style("whitegrid")
    #plt.figure(figsize=(13,10))
    #ax = sns.boxplot(data=df_norm, orient="h", palette="Set2")
    #plt.title('Outlier - Avant traitement',fontsize = 25)
    #plt.xticks(fontsize = 15)
    #plt.yticks(fontsize = 15)
    #plt.show()



    #Traitement variables numérique
    #Valeurs nutritionnelles
    for column in df_num.filter(regex='_100g'):
        #print(column)
        if column == 'nutrition-score-fr_100g':
            df_num.loc[df_num[column] < -15] = -15
            df_num.loc[df_num[column] > 40] = 40
        else :   
            df_num.loc[df_num[column] < 0] = 0
            df_num.loc[df_num[column] > 100] = 100
            
    #Variables homogènes        
    for column in df_num.filter(regex='_n'):
        cap = df_num.filter(regex='_n').quantile([lower_percentile, higher_percentile])
        low = cap.at[cap.index[0],column]
        high = cap.at[cap.index[1],column]
        df_num.loc[df_num[column] < low] = low
        df_num.loc[df_num[column] > high] = high
        
    df.drop(labels=num_, axis="columns", inplace=True)
    df[num_] = df_num[num_]
    
    print('   ')
    print('   ')
    print('---Après traitement---')
    display(df.describe())
    
    
    #PLOT
    #pour trier par médiane
    #meds_norm = df_norm.median()
    #meds_norm.sort_values(ascending=False, inplace=True)
    #df_norm = df_norm[meds_norm.index]
    
    #sns.set_style("whitegrid")
    #plt.figure(figsize=(13,10))
    #ax = sns.boxplot(data=df_norm, orient="h", palette="Set2")
    #plt.title('Outlier - Avant traitement',fontsize = 25)
    #plt.xticks(fontsize = 15)
    #plt.yticks(fontsize = 15)
    #plt.show()   
            
    df = df.reset_index(drop=True)
    
    return df