#!/usr/bin/env python
# coding: utf-8

# # Anticipez les besoins en consommation électrique de bâtiments

# Pascaline Grondein
# 
# Début : 04/04/2022

# <i/> Vous travaillez pour la ville de Seattle. Pour atteindre son objectif de ville neutre en émissions de carbone en 2050, votre équipe s’intéresse de près aux émissions des bâtiments non destinés à l’habitation. </i>
# 
# <i/>Des relevés minutieux ont été effectués par vos agents en 2015 et en 2016. Cependant, ces relevés sont coûteux à obtenir, et à partir de ceux déjà réalisés, vous voulez tenter de prédire les émissions de CO2 et la consommation totale d’énergie de bâtiments pour lesquels elles n’ont pas encore été mesurées. Vous cherchez également à évaluer l’intérêt de l’"ENERGY STAR Score" pour la prédiction d’émissions, qui est fastidieux à calculer avec l’approche utilisée actuellement par votre équipe.</i>
# 
# https://www.energystar.gov/buildings/benchmark/analyze_benchmarking_results

# ### Table of Contents
# 
# * [I. Approche naïve avec Dummy Regressor](#chapter1)
# * [II. Regression linéaire](#chapter2)
#     * [1. Multiple](#section_2_1)
#     * [2. Regression Ridge](#section_2_2)
#     * [3. Regression Lasso](#section_2_3)
# * [III. Modèles non linéaires](#chapter3)
#     * [1. Decision Tree](#section_3_1)
#     * [2. Random Forest](#section_3_2)
# * [IV. Conclusion](#chapter4)

# In[15]:


import pandas as pd
pd.set_option('precision', 2)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import lasso_path
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn import neighbors, metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error

import time
from time import process_time

import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = "{:,.2f}".format


# Afin de déterminer si ENERGYSTARScore est une variable importante à considérer, nous allons tester les modèles avec et sans, on crée donc une version de X_train/test sans ESS.
# 
# Ensuite il est intéressant de comparer les résultats avec les targets en linéaire et log. 

# In[16]:


X_train_ESS = pd.read_csv('X_train.csv',index_col = 'OSEBuildingID')
X_train_noESS = X_train_ESS.drop(['ENERGYSTARScore'], axis = 1)

X_test_ESS = pd.read_csv('X_test.csv',index_col = 'OSEBuildingID')
X_test_noESS = X_test_ESS.drop(['ENERGYSTARScore'], axis = 1)

y_train_lin = pd.read_csv('y_train.csv',index_col = 'OSEBuildingID')
y_train_log = pd.read_csv('y_train_log.csv',index_col = 'OSEBuildingID')

y_test_lin = pd.read_csv('y_test.csv',index_col = 'OSEBuildingID')
y_test_log = pd.read_csv('y_test_log.csv',index_col = 'OSEBuildingID')


# # Fonctions

# In[17]:


def eval(y_test,y_pred,model_target,t_start, t_stop):
    #model_target : nom model + var target
    
    d = {'R²': [round(r2_score(y_test,y_pred),2)],
         'MAE': [round(metrics.mean_absolute_error(y_test, y_pred),2)],
         'MAPE': [round(mean_absolute_percentage_error(y_test, y_pred),2)],
         'RMSE': [round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)],
        'Computational time' : [round(t_stop - t_start,2)]}
    metrics_ = pd.DataFrame(data = d, index = [model_target])
    
    return metrics_


# In[18]:


def comp_pred(model,name_plot,target,y_test,y_pred,lg,ess):
    #log : log si y en log lin sinon
    #ess : ESS si ENERGYSTARScore utilsié noESS sinon
    
    Y_max = y_test[target].max()
    Y_min = y_test[target].min()
      
    globals()[name_plot] = plt.figure(figsize=(10,10))
    plots.append(name_plot)
    
    ax = sns.scatterplot(y_pred, y_test[target])
        
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('y_test',fontsize = 20)
    plt.ylabel('y_pred',fontsize = 20)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim=(Y_min, Y_max)
    plt.xlim=(Y_min, Y_max)

    X_ref = Y_ref = np.linspace(Y_min, Y_max, 100)
    plt.plot(X_ref, Y_ref, color='red', linewidth=1)
    
    plt.title('{}_{}_{}_{}'.format(model,target,lg,ess),fontsize = 20)
    plots.append('{}_{}_{}_{}'.format(model,target,lg,ess))
    #plt.savefig('{}_{}_{}_{}'.format(model,target,lg,ess))
    plt.show()


# In[19]:


def coeff_plot(coeff,model,target,var,lg,ess):
    #log : log si y en log lin sinon
    #ess : ESS si ENERGYSTARScore utilsié noESS sinon
    
    coeff.sort_values('Coefficients',key=abs,ascending=False, inplace = True)
    plt.figure(figsize=(10,10))
    sns.barplot(x=coeff['Coefficients'].head(var), y=coeff.head(var).index, data=coeff)
    
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Weights',fontsize = 20)
    plt.ylabel('',fontsize = 20)
    
    plt.title('coeff_{}_{}_{}_{}'.format(model,target,lg,ess),fontsize = 20)
    plots.append('coeff_{}_{}_{}_{}'.format(model,target,lg,ess))
    plt.show()        


# In[20]:


def alphavsMSE_lasso(model):
    
    #Moyenne mse en validation croisée pour chaque alpha
    avg_mse = np.mean(model.mse_path_,axis=1)
    
    #Graphique
    plt.figure(figsize=(8,8))
    plt.semilogx(model.alphas_,avg_mse)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Alpha',fontsize = 20)
    plt.ylabel('MSE',fontsize = 20)
    plt.title('MSE vs. Alpha')
    plt.show()
    
    #Alpha qui minimise MSE
    return model.alpha_


# In[21]:


def alphavsMSE_ridge(model_GS,target,lg,ess):
    
    results = pd.DataFrame(model_GS.cv_results_)
    
    #Graphique
    plt.figure(figsize=(8,8))
    plt.plot(results['param_alpha'],results['mean_test_score'])
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Alpha',fontsize = 20)
    plt.ylabel('R²',fontsize = 20)
    plt.title('R² vs. Alpha -{}_{}_{}'.format(target,lg,ess))
    plt.show()
    
    #Alpha qui minimise MSE
    return model_GS.best_params_


# In[22]:


def min_max(df):
    return df.style.highlight_max(subset = 'R²',color = 'lightgreen', axis = 0).highlight_min(
        subset = ['MAE','MAPE','RMSE','Computational time'],color = 'lightgreen', axis = 0)


# In[23]:


boxprops = dict(linestyle='-', linewidth=1, color='k')
medianprops = dict(linestyle='-', linewidth=1, color='k')
meanprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick')


# In[24]:


metrics_models_TotalGHGEmissions = pd.DataFrame()
metrics_models_SiteEnergyUse = pd.DataFrame()


ess_list = ['ESS','noESS']
log_list = ['lin','log']
targets = ['TotalGHGEmissions','SiteEnergyUse(kBtu)']
plots = []


# Afin de trouver quel modèle est le plus adapté à la prédiction des deux targets, nous allons en tester plusieurs :
# 
#  - En premier, un DummyRegressor, un modèle de régression très simple, qui n'aura pas de bonnes performances, mais qui nous permettra de mettre une limite basse à la comparaison globale.
#  - Ensuite, une régression linéaire : 
#      - Simple avec OLS
#      - Régularisée l2 avec Ridge
#      - Régularisée l1 avec Lasso
#  - Enfin, des modèles non linéaires, méthodes ensemblistes avec : 
#      - Decision Tree
#      - Random Forest

# # <a class="anchor" id="chapter1">I. Approche naïve avec Dummy Regressor</a>

# In[25]:


dummy_metrics_TotalGHGEmissions = pd.DataFrame()
dummy_metrics_SiteEnergyUse = pd.DataFrame()
dummy_regr = DummyRegressor(strategy="mean")


# In[26]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin
                
            t1 = time.process_time()
            
            dummy_regr.fit(X_train, y_train[target])
            y_pred = dummy_regr.predict(X_test)
            
            t2 = process_time()
            
            m = eval(y_test[target],y_pred,model_target = 'Dummy_{}_{}_{}'.format(lg,ess,target),t_start=t1,t_stop=t2)
            
            if target == 'TotalGHGEmissions' :
                dummy_metrics_TotalGHGEmissions = dummy_metrics_TotalGHGEmissions.append(m)
            else :
                dummy_metrics_SiteEnergyUse = dummy_metrics_SiteEnergyUse.append(m)      


# In[27]:


display(min_max(dummy_metrics_TotalGHGEmissions))
display(min_max(dummy_metrics_SiteEnergyUse))


# In[28]:


metrics_models_TotalGHGEmissions = metrics_models_TotalGHGEmissions.append(dummy_metrics_TotalGHGEmissions.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))
metrics_models_SiteEnergyUse = metrics_models_SiteEnergyUse.append(dummy_metrics_SiteEnergyUse.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))

display(metrics_models_TotalGHGEmissions)
display(metrics_models_SiteEnergyUse)


# Comme prédit, les performances de ce modèle simpliste sont mauvaises. On garde cependant la meilleure version pour chaque target.

# # <a class="anchor" id="chapter2">II. Régression linéaire</a>

# ## <a class="anchor" id="section_2_1">1. Multiple </a> 

# Puisqu'on utlise OLS pour la régression simple, il faut ajouter une constante aux features. 

# In[29]:


X_train_cst_ESS = sm.add_constant(X_train_ESS)
X_train_cst_noESS = sm.add_constant(X_train_noESS)
X_test_cst_ESS = sm.add_constant(X_test_ESS)
X_test_cst_noESS = sm.add_constant(X_test_noESS)


# In[30]:


lrols_metrics_TotalGHGEmissions = pd.DataFrame()
lrols_metrics_SiteEnergyUse = pd.DataFrame()


# In[31]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_cst_noESS
                X_test = X_test_cst_noESS
            else :
                X_train = X_train_cst_ESS
                X_test = X_test_cst_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin
            
            t1 = time.process_time()
            
            model = sm.OLS(y_train[target], X_train)
            res = model.fit()
            #display(res.summary())
            
            y_pred = res.predict(X_test)

            t2 = process_time()
            
            m = eval(y_test[target],y_pred,model_target = 'LROLS_{}_{}_{}'.format(lg,ess,target),t_start=t1,t_stop=t2)
            
            if target == 'TotalGHGEmissions' :
                lrols_metrics_TotalGHGEmissions = lrols_metrics_TotalGHGEmissions.append(m)
            else :
                lrols_metrics_SiteEnergyUse = lrols_metrics_SiteEnergyUse.append(m)
    
            comp_pred(model = 'lrols',name_plot = 'lrols_{}_{}_{}'.format(target,lg,ess),target=target,
                      y_test=y_test,y_pred=y_pred,lg=lg,ess=ess)
    
            coeff = pd.DataFrame(res.params, columns=['Coefficients'])
            display(coeff.sort_values('Coefficients',key=abs,ascending=False))
            coeff_plot(coeff,'lrols',target,30,lg,ess)


# In[32]:


display(min_max(lrols_metrics_TotalGHGEmissions))
display(min_max(lrols_metrics_SiteEnergyUse))


# In[33]:


metrics_models_TotalGHGEmissions = metrics_models_TotalGHGEmissions.append(lrols_metrics_TotalGHGEmissions.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))
metrics_models_SiteEnergyUse = metrics_models_SiteEnergyUse.append(lrols_metrics_SiteEnergyUse.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))

display(metrics_models_TotalGHGEmissions)
display(metrics_models_SiteEnergyUse)


# Les performances de la régression linéaire simple sont meilleurs que celles du modèle simple, comme attendu, mais il serait intéressant d'intégrer une forme de régularisation dans la régression linéaire. 
# 
# Au niveau de l'importance des variables, pour ce modèle, les types de bâtiments semblent être les principales. L'ENERGYSTARScore n'apaprait pas dans le top 30 des variables.

# ##  <a class="anchor" id="section_2_2">2. Regression ridge </a> 

# In[34]:


ridge_metrics_TotalGHGEmissions = pd.DataFrame()
ridge_metrics_SiteEnergyUse = pd.DataFrame()


# Avant d'appliquer le modèle de régression linéaire ridge, il faut trouver l'hyperparamètre de régularisation le plus adapté. On utilise donc GridSearch pour effectuer une validation croisée.

# In[35]:


n_alphas = 50
my_alphas = np.logspace(-2, 1, n_alphas)
best_alpha_ridge = pd.DataFrame()

parameters = {'alpha' : my_alphas}

ridge_GS = GridSearchCV(Ridge(fit_intercept = True),parameters,cv=5,scoring = 'r2')


# In[36]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin
                
            ridge_GS.fit(X_train, y_train[target])
            best_param = alphavsMSE_ridge(ridge_GS,target,lg,ess)
            best_param = pd.DataFrame(data = best_param, index = ['Ridge_{}_{}_{}'.format(lg,ess,target)])
            best_alpha_ridge = best_alpha_ridge.append(best_param)


# In[37]:


best_alpha_ridge


# Maintenant que nous avons nos meilleurs alphas, on peut les appliquer pour obtenir nos erreurs finales de validation.
# 

# In[38]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin
                
            t1 = time.process_time()
                
            best_alpha = best_alpha_ridge.loc[best_alpha_ridge.index == 'Ridge_{}_{}_{}'.format(lg,ess,target), 
                                              'alpha']
            
            model = Ridge(alpha=best_alpha,fit_intercept=True) 
            res = model.fit(X_train, y_train[target])        
            
            t2 = process_time()
            
            y_pred = res.predict(X_test)
            m = eval(y_test[target],y_pred,model_target = 'Ridge_{}_{}_{}'.format(lg,ess,target),t_start=t1,t_stop=t2)
            
            if target == 'TotalGHGEmissions' :
                ridge_metrics_TotalGHGEmissions = ridge_metrics_TotalGHGEmissions.append(m)
            else :
                ridge_metrics_SiteEnergyUse = ridge_metrics_SiteEnergyUse.append(m)
            
            comp_pred(model = 'ridge',name_plot = 'ridge_{}_{}_{}'.format(target,lg,ess),target=target,
            y_test=y_test,y_pred=y_pred,lg=lg,ess=ess)
    
            coeff = pd.DataFrame(res.coef_, columns=['Coefficients'],index = X_train.columns)
            display(coeff.sort_values('Coefficients',key=abs,ascending=False))
            coeff_plot(coeff,'ridge',target,30,lg,ess)


# In[39]:


display(min_max(ridge_metrics_TotalGHGEmissions))
display(min_max(ridge_metrics_SiteEnergyUse))


# In[40]:


metrics_models_TotalGHGEmissions = metrics_models_TotalGHGEmissions.append(ridge_metrics_TotalGHGEmissions.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))
metrics_models_SiteEnergyUse = metrics_models_SiteEnergyUse.append(ridge_metrics_SiteEnergyUse.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))

display(metrics_models_TotalGHGEmissions)
display(metrics_models_SiteEnergyUse)


# Les performances sont meilleures, notamment pour les émissions de carbone. 
# 
# Concernant l'importance des variables, ce modèle semble considérer la surface du bâtiment comme la variable principale, suivie des différentes catégories d'énergie utilisée.

# ### <a class="anchor" id="section_2_3">3. Regression Lasso </a> 

# In[41]:


lasso_metrics_TotalGHGEmissions = pd.DataFrame()
lasso_metrics_SiteEnergyUse = pd.DataFrame()


# Comme pour la régression Ridge, déterminons en premier lieu le meilleur terme de régularisation. 

# In[42]:


n_alphas = 50
my_alphas = np.logspace(-3, 1, n_alphas)

best_alpha_lasso = pd.DataFrame(columns = ['Version','Best Alpha'])

model = LassoCV(eps=0.001, alphas=my_alphas, fit_intercept=True, max_iter=1000, 
                tol=0.0001, cv=5, selection='cyclic',random_state=0)


# In[43]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin

            model.fit(X_train, y_train[target])
      
            best_alpha = alphavsMSE_lasso(model)
            print('Le meilleur alpha pour {} en {} et {} est'.format(target,lg,ess),best_alpha,'.')
            best_alpha_lasso = best_alpha_lasso.append({'Version' : '{}_{}_{}'.format(target,lg,ess),
                                                        'Best Alpha' : best_alpha},
                                              ignore_index=True)


# In[44]:


best_alpha_lasso


# Maintenant que nous avons nos meilleurs alphas, on peut les appliquer pour obtenir nos erreurs finales de validation.

# In[45]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin

            t1 = time.process_time()
            
            best_alpha = best_alpha_lasso.loc[best_alpha_lasso['Version'] == '{}_{}_{}'.format(target,lg,ess),
                                              'Best Alpha']
            model = LassoCV(eps=0.001, alphas=best_alpha, fit_intercept=True, max_iter=1000, 
                            tol=0.0001, cv=5, selection='cyclic',random_state=0)
            res = model.fit(X_train, y_train[target])
    
            y_pred = res.predict(X_test)
        
            t2 = process_time()
            
            m = eval(y_test[target],y_pred,model_target = 'LassoCV_{}_{}_{}'.format(target,lg,ess),t_start=t1,t_stop=t2)
            
            if target == 'TotalGHGEmissions' :
                lasso_metrics_TotalGHGEmissions = lasso_metrics_TotalGHGEmissions.append(m)
            else :
                lasso_metrics_SiteEnergyUse = lasso_metrics_SiteEnergyUse.append(m)
    
            comp_pred(model = 'lasso',name_plot = 'ridge_{}_{}_{}'.format(target,lg,ess),
                      target = target,y_test = y_test,y_pred=y_pred,lg=lg,ess=ess)
    
            coeff = pd.DataFrame(model.coef_,columns=['Coefficients'], index=X_train.columns)
            display(coeff.sort_values('Coefficients',key=abs,ascending=False))
            coeff_plot(coeff,'lasso',target,30,lg,ess)


# In[46]:


display(min_max(lasso_metrics_TotalGHGEmissions))
display(min_max(lasso_metrics_SiteEnergyUse))


# In[47]:


metrics_models_TotalGHGEmissions = metrics_models_TotalGHGEmissions.append(lasso_metrics_TotalGHGEmissions.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))
metrics_models_SiteEnergyUse = metrics_models_SiteEnergyUse.append(lasso_metrics_SiteEnergyUse.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))

display(metrics_models_TotalGHGEmissions)
display(metrics_models_SiteEnergyUse)


# Les performances sont encore une fois meilleures que pour la régresssion Ridge. cependant, on peut encore améliorer ces résultats en testant une autre forme de modèles. 
# 
# Pour l'importance des variables, pour SiteEnergyUse les types d'énergie utilisées sont au top, accompagnées d'un type de bâtiment, et de la surface du bâtiment pour l'émission de carbone.

# # <a class="anchor" id="chapter3">III. Modèles non linéaires</a> 

# ## <a class="anchor" id="section_3_1">1. Decision Tree </a> 

# In[48]:


DT_metrics_TotalGHGEmissions = pd.DataFrame()
DT_metrics_SiteEnergyUse = pd.DataFrame()


# On détermine en premier les hyper paramètres optimaux.

# In[49]:


n_var = X_train.shape[1]
best_params_DT = pd.DataFrame()

parameters = {
    'min_samples_leaf' : [5,10,50],
    'max_depth': [int(n_var/2),int(n_var/3),int(n_var/4)]
}

DT_GS = GridSearchCV(DecisionTreeRegressor(random_state=0),parameters,cv=5,scoring = 'neg_mean_squared_error')


# In[50]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin
                
            DT_GS.fit(X_train, y_train[target])
            
            gsdt_result = DT_GS.fit(X_train, y_train[target])
            params = pd.DataFrame(data = gsdt_result.best_params_, index = ['DT_{}_{}_{}'.format(lg,ess,target)])
            best_params_DT = best_params_DT.append(params)


# In[51]:


best_params_DT


# Maintenant que nous avons nos meilleurs hyperparamètres, on peut les appliquer pour obtenir nos erreurs finales de validation.

# In[52]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin
            
            t1 = time.process_time()
            
            model = DecisionTreeRegressor(max_depth=best_params_DT.at['DT_{}_{}_{}'.format(lg,ess,target),
                                                                           'max_depth'],
                                       min_samples_leaf = best_params_DT.at['DT_{}_{}_{}'.format(lg,ess,target),
                                                                         'min_samples_leaf'],)
                                      
            res = model.fit(X_train,y_train[target])
            
            y_pred = res.predict(X_test)

            t2 = process_time()
            
            m = eval(y_test[target],y_pred,model_target = 'DT_{}_{}_{}'.format(lg,ess,target),t_start=t1,t_stop=t2)

            if target == 'TotalGHGEmissions' :
                DT_metrics_TotalGHGEmissions = DT_metrics_TotalGHGEmissions.append(m)
            else :
                DT_metrics_SiteEnergyUse = DT_metrics_SiteEnergyUse.append(m)
            
    
            comp_pred('DT','DT_{}_{}_{}'.format(target,lg,ess),target,y_test,y_pred,lg,ess)
    
            coeff = pd.DataFrame(res.feature_importances_, columns=['Coefficients'],index = X_train.columns)
            display(coeff.sort_values('Coefficients',key=abs,ascending=False))
            coeff_plot(coeff,'DT',target,30,lg,ess)


# In[53]:


display(min_max(DT_metrics_TotalGHGEmissions))
display(min_max(DT_metrics_SiteEnergyUse))


# In[54]:


metrics_models_TotalGHGEmissions = metrics_models_TotalGHGEmissions.append(DT_metrics_TotalGHGEmissions.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))
metrics_models_SiteEnergyUse = metrics_models_SiteEnergyUse.append(DT_metrics_SiteEnergyUse.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))

display(metrics_models_TotalGHGEmissions)
display(metrics_models_SiteEnergyUse)


# On observe une nette amélioration des performances. 
# 
# Pour l'importance des variables, les types d'énergie utilisées sont au top, accompagnées d'un type de bâtiment.

# ## <a class="anchor" id="section_3_2">2. Random forest </a> 

# In[55]:


RF_metrics_TotalGHGEmissions = pd.DataFrame()
RF_metrics_SiteEnergyUse = pd.DataFrame()


# Regardons en premier lieu le OOB en fonction de n_estimator afin de déterminer le meilleur n_estimator.

# In[56]:


#n_estimators = (10,50,100,150,200,300,500)
min_estimators = 10
max_estimators = 200
n_estimators = [i for i in range(min_estimators,max_estimators+1,10)]

forest = RandomForestRegressor(warm_start=False, oob_score=True,max_features='sqrt')

best_n = pd.DataFrame()

for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin

          
            oob_error = pd.DataFrame(columns = ['n','oob'])

            for i in n_estimators:
                forest.set_params(n_estimators=i)
                forest.fit(X_train, y_train[target])
                d = {'n' : i,'oob' : 1 - forest.oob_score_,'index' : 'RF_{}_{}_{}'.format(target,lg,ess)}
                oob_error = oob_error.append(d, ignore_index = True)
            
            best_n = best_n.append(oob_error.max(), ignore_index = True)
                
            plt.figure(figsize=(12,10))
            ax = plt.axes()
            ax.set_ylim(0, 1)
            
            plt.plot(oob_error['n'],oob_error['oob'])
            plt.title('RF_{}_{}_{}'.format(target,lg,ess),fontsize = 20)
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.xlabel("n_estimators",fontsize = 20)
            plt.ylabel("OOB error rate",fontsize = 20)
            plt.show()
            
best_n = best_n.set_index('index')


# On observe un coude assez visible autour de 50, où les valeurs de l'OOB error rate ne semblent plus diminuer de façon flagrante. Dans Grid Search on peut tester les valeurs n_estimators entre 10 et 100. 
# 
# Déterminons maintenant les meilleurs hyperparamètres.

# In[57]:


n_var = X_train.shape[1]
best_params = pd.DataFrame()


# In[58]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin
            
            
            param_grid = {
                'n_estimators' : [10,30,50,70,100],
                'min_samples_leaf' : [5,10,50],
                'max_depth': [int(n_var/2),int(n_var/3),int(n_var/4)]
            }
            
            Random_forest_GS = GridSearchCV(estimator=RandomForestRegressor(max_features = 'sqrt',oob_score = True),
                                param_grid = param_grid,
                                cv=5,scoring='neg_mean_squared_error')
            
            
            gsrf_result = Random_forest_GS.fit(X_train, y_train[target])
            params = pd.DataFrame(data = gsrf_result.best_params_, index = ['RF_{}_{}_{}'.format(lg,ess,target)])
            best_params = best_params.append(params)


# In[59]:


best_params


# Maintenant que nous avons nos meilleurs hyperparamètres, on peut les appliquer pour obtenir nos erreurs finales de validation.

# In[60]:


for target in targets:
    for ess in ess_list:
        for lg in log_list:
            if ess == 'noESS':
                X_train = X_train_noESS
                X_test = X_test_noESS
            else :
                X_train = X_train_ESS
                X_test = X_test_ESS
                
            if lg == 'log':
                y_train = y_train_log
                y_test = y_test_log
            else : 
                y_train = y_train_lin
                y_test = y_test_lin
                
    
            t1 = time.process_time()
        
            rf = RandomForestRegressor(max_depth=best_params.at['RF_{}_{}_{}'.format(lg,ess,target),'max_depth'],
                                       min_samples_leaf = best_params.at['RF_{}_{}_{}'.format(lg,ess,target),'min_samples_leaf'],
                                       n_estimators = int(best_params.at['RF_{}_{}_{}'.format(lg,ess,target),'n_estimators'])
                                      )
            
            rf.fit(X_train,y_train[target])
            
            y_pred = rf.predict(X_test)
            
            t2 = process_time()
            
            m = eval(y_test[target],y_pred,model_target = 'RF_{}_{}_{}'.format(lg,ess,target),t_start=t1,t_stop=t2)
            
            if target == 'TotalGHGEmissions' :
                RF_metrics_TotalGHGEmissions = RF_metrics_TotalGHGEmissions.append(m)
            else :
                RF_metrics_SiteEnergyUse = RF_metrics_SiteEnergyUse.append(m)
    
            comp_pred('RF','RF_{}_{}_{}'.format(lg,ess,target),target,y_test,y_pred,lg,ess)
        
            coeff = pd.DataFrame(rf.feature_importances_,columns=['Coefficients'], index=X_train.columns)
            coeff_plot(coeff,'RF',target,30,lg,ess)


# In[61]:


display(min_max(RF_metrics_TotalGHGEmissions))
display(min_max(RF_metrics_SiteEnergyUse))


# In[62]:


metrics_models_TotalGHGEmissions = metrics_models_TotalGHGEmissions.append(RF_metrics_TotalGHGEmissions.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))
metrics_models_SiteEnergyUse = metrics_models_SiteEnergyUse.append(RF_metrics_SiteEnergyUse.nlargest(1,['R²']).nsmallest(1,['MAE','MAPE','RMSE','Computational time']))

display(metrics_models_TotalGHGEmissions)
display(metrics_models_SiteEnergyUse)


# Les performances obtenues sont nettement supérieures à celles des modèles précédents. 

# # <a class="anchor" id="chapter4">IV. Conclusion</a> 

# In[65]:


fig, ax = plt.subplots(figsize=(12,10))
sns.set(font_scale=1.4)
res = sns.heatmap(metrics_models_TotalGHGEmissions, annot = True,vmax = 2,
                  linewidths=.5,cmap="YlGnBu",square=True,fmt='.2f')
res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 16)
plt.title('Meilleurs modèles pour l\'émission de carbonne',fontsize = 20)
plt.show()


# Pour la target liée à l'émission de carbone (TotalGHGEmissions), le modèle Random Forest présente le plus haut R², ainsi que les plus bas MAE, MAPE et RMSE, quand il est combiné aux targets transformées en log et avec l'ENERGYSTARScore inclue. 

# In[64]:


fig, ax = plt.subplots(figsize=(12,10))
sns.set(font_scale=1.4)
res = sns.heatmap(metrics_models_SiteEnergyUse, annot = True,vmax = 2,
                  linewidths=.5,cmap="YlGnBu",square=True,fmt='.2f')
res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 16)
plt.title('Meilleurs modèles pour la consommation en énergie',fontsize = 20)
plt.show()


# Pour la target de consommation d'énergie (SiteEnergyUse), le modèle Random Forest présente le plus haut R², ainsi que les plus bas MAE, MAPE et RMSE, quand il est combiné aux targets transformées en log et avec l'ENERGYSTARScore inclue. 

# L'ajout de la variables ENERGYSTARScore apparaît en moyenne et en premier lieu comme une option préférable pour les différents modèles, pour les deux targets.
# Cependant, en observant les graphes d'importances des variables, on s'aperçoit qu'elle n'est que rarement classée dans les trente premières. Le modèle Ridge lui accorde une importance proche ou égale à zéro dans chaque cas, et si l'on compare les performances des modèles avec et sans la variable, on peut constater qu'elles sont très proches.
# La variables ENERGYSTARScore ne semble pas influer particulièrement sur les targets. L'ajouter ne semble pas indispensable, étant une variable compliquée à obtenir et coûteuse. 

# In[ ]:




