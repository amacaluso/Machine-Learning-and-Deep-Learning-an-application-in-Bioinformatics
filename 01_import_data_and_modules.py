# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:08:26 2017

@author: Antonio
"""

from Utils import *
###############################################
############ Importazione dati ################
###############################################

data_X = pd.read_csv('dataset/gene_expression.csv', decimal = '.')
dati_risposta = pd.read_table('dataset/ccle_drug_response.txt', decimal = '.')

complete_dataframe = pd.merge( dati_risposta, data_X, on = 'ID')
#print('La dimensione del dataframe completo (variaibli risposta + regressori) e', complete_dataframe.shape)

X = data_X.drop('ID',1)

sns.set_style('whitegrid')

n_rows = X.shape[0]
n_col = X.shape[1]

print('La dimensione della matrice dei regressori è ', n_rows, 'x', n_col)






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
