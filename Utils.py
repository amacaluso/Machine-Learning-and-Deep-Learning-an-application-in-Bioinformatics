# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 20:05:47 2017

@author: Antonio
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import sklearn.neural_network as nn

import math
from pylab import *
import seaborn as sns
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV 
from sklearn import svm


import sklearn as skl
from sklearn import cross_validation, linear_model


from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
import random
import time


from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV



def create_dataset( data, 
                    target_variable,
                    explanatory_variable                     
                   ):
#    
    data = data.dropna(subset = [target_variable])
    dataset = pd.concat([data[target_variable], data[ explanatory_variable ]], axis = 1)

    return dataset 




def model_estimation( data,
                      target_variable,
                      explanatory_variable,
                      test_data,
                      model = skl.linear_model.LinearRegression(), 
                        ):


    data = data.dropna(subset = [target_variable])
    
    
    fit = model.fit( data[explanatory_variable], data[target_variable] )
    predict = model.predict( test_data[explanatory_variable] )
    
    return  predict
    



def cross_validation( data, 
                      target_variable,
                      explanatory_variable, 
                      model = skl.linear_model.LinearRegression(),
                      splits = 10
                     ):
    
    
    data = data.sample( n = len(data))
    data = data.reset_index()  
    
    n = len( data )
    kf = KFold( n_splits = splits )
    Y_true = []
    Y_hat = []


    print("CROSS VALIDATION ...\n")
    
    for train_idx, test_idx in kf.split( data ):
        
        training_dataset = data.ix[ train_idx ]
        test_dataset = data.ix[ test_idx ]
#        print(len(test_dataset))
        prediction = model_estimation(training_dataset, 
                                      target_variable,
                                      explanatory_variable,
                                      test_data = test_dataset,
                                      model = model)
        Y_hat.append( prediction)
        Y_true.append(test_dataset[target_variable])
        
#    ######################################################    
#    hat = pd.concat( pd.Series(Y_hat[i]) for i in range ( len(Y_hat ) ))
#    true = pd.concat( pd.Series(Y_true[i]) for i in range ( len(Y_true ) ))
#    
#    np.corrcoef(hat, true)

    
    
    
    differenza = []
    
    
    for i in range( len(Y_hat)):
        differenza.append( Y_true[i] - Y_hat[i] )
        
    diff = []
    diff = pd.concat( differenza[i] 
                      for i in range( len( differenza)) )
    
    diff_2 = []
    diff_2 = np.power(diff, 2)
        
    SSE = sum(diff_2)
    MSE = SSE/n
    Root_MSE = sqrt(MSE)

    SE = np.sum(diff)
    
    Y = []
    Y = pd.concat( Y_true[j] for j in range( len( Y_true )))

    Dev_Y = np.var( Y )*n
    Var_Y = np.var( Y )
    RSE = SSE/Dev_Y
    RRSE = sqrt(RSE)
    
    MAE = sum( abs(diff) )/n

    differenza_media = []
    media = np.mean(Y)
    
    for i in range( len(Y_hat)):
        differenza_media.append( Y_true[i] - media)
        
    diff_media = []
    diff_media = pd.concat( differenza_media[i] 
                      for i in range( len( differenza_media)) )
    
    
    RAE = sum( abs(diff) )/sum( abs(diff_media) )

             
    return [SE, SSE, MSE, Root_MSE, RSE, RRSE, MAE, RAE, Dev_Y, Var_Y]


def plot_error (df,
                ordinata,
                ascissa,
                Y_array,
                group,
                name_file):
    types =['bs-', 'ro-','bs-',  'ro-']
    colors = ['b', 'r', 'y', 'g']
    
    for Y in Y_array :
        for i in range( len(group)):
            current_data = df.loc[( df['Y'] == Y ) & ( df['Modello'] == group[i] ),:]
            corr = np.corrcoef( current_data[ascissa], current_data[ordinata])[0,1]
            minimo = min(current_data[ordinata])
            mod_min = current_data.loc[ current_data[ordinata] == minimo,'Modello']
    
            c = colors[i]
            t = types[i]
            
            x = df.loc[( df['Y'] == Y ) & ( df['Modello'] == group[i] ), ascissa]
            y = df.loc[( df['Y'] == Y ) & ( df['Modello'] == group[i] ), ordinata]
            plt.plot(x , y, t , color = c, label = group[i])
#            plt.figure( figsize = (2,2))

        
        plt.title(Y)
        plt.legend()
#        plt.figtext(0.6, 0.3, s = [ mod_min, "{0:.4f}".format(minimo)] )
        plt.ylabel(ordinata)
        plt.xlabel(ascissa)
        plt.legend()
        path = 'results/no_PCA/'+ name_file + Y + '.png'
        savefig(path, dpi = 500)
        plt.show()
        plt.close()