# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 08:49:55 2017

@author: amacaluso
"""


exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descriptive.py').read(), globals())
#exec(open("05_hypothesis_test.py").read(), globals())
#exec(open("10_PCA.py").read(), globals())

#########################################################################
##################### ALL CELL LINE #####################################
#########################################################################



name_columns = ['Y', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                 'MAE', 'RAE','Deviance', 'Variance', 'alpha', 'coeff' ]

X_matrix = X.columns


Y_array = dati_risposta.columns[4:8]
# Y_array
list_data = []

for y in Y_array:
    list_data.append( create_dataset(data = complete_dataframe, 
                                     target_variable = y, 
                                     explanatory_variable = X_matrix))


tipo = list(set(dati_risposta.ix[:,2]))
tipo
###################################################################
##################### Ridge Regression ###########################
###################################################################

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV


model = RidgeCV(alphas=(0.1, 0.5, 1.0, 10, 20, 30, 50, 100, 200))#, cv = 10)
result_regression = []


for i in range( 0, len(Y_array) ):
    print(i)
#    i=0
    print( Y_array[i] )
    n = len( list_data[i] )
    data = list_data[i]
    
    X_train = data[ X_matrix ]
    y_train = data[ Y_array[i] ]

    ridge_reg = model.fit( X_train, y_train )
    coefficients = ridge_reg.coef_
    alpha = ridge_reg.alpha_
    regression = cross_validation(splits = 100, 
                                  target_variable = Y_array[i], 
                                  explanatory_variable = X_matrix,
                                  data = list_data[i],
                                  model = ridge_reg)
    
    risultati =  [Y_array[i], n] + regression + [alpha]
    risultati.append(np.ndarray.tolist(coefficients))
    result_regression.append( risultati )
        



df_regression = pd.DataFrame(result_regression)
df_regression.columns = name_columns
df_regression

writer = pd.ExcelWriter('results/ridge_regression/Regressione_ridge_no_PCA.xlsx')
df_regression.to_excel(writer)
writer.save()


###################################################################
##################### Ridge Regression Cancer type ############################
###################################################################


tipo = ['skin','aero_digestive_tract', 'breast',
        'nervous_system', 'urogenital_system',
        'digestive_system', 'blood', 'lung']
#t = 'nervous_system'
result_regression = []



name_columns = ['Y', 'Cancer type', 'Numerosità', 'SE','SSE', 'MSE', 
                'Root_MSE', 'RSE','RRSE', 'MAE', 'RAE','Deviance', 
                'Variance' ]


for t in tipo:
    
    df_regression = pd.DataFrame()#columns = name_columns)
    model = LassoCV(eps = 0.1, n_alphas = 100) #, cv = 20 )
    dati = complete_dataframe[complete_dataframe['Cancer Type']==t]

        
    
    for y in Y_array:
    
         data = create_dataset(data = dati, 
                               target_variable = y, 
                               explanatory_variable = X_matrix)
           
         n = len( data)
         print(y, t, n)
            
#         X_train = data[ X_matrix ]
#         y_train = data[ y ]
        
#            coefficients = lasso.coef_
#            alpha = lasso.alpha_
         regression = cross_validation(splits = n, 
                                       target_variable = y, 
                                       explanatory_variable = X_matrix,
                                       data = data,
                                       model = model)
                
         risultati =  [y, t, n] + regression
         result_regression.append( risultati )
    
        
    df_regression = pd.DataFrame(result_regression)
#
#writer = pd.ExcelWriter('results//Regressione_lasso ct_no_PCA.xlsx')
#df_regression.to_excel(writer)
#writer.save()


        
    df_regression.columns = name_columns
    df = df_regression
    
    
        
    
    
writer = pd.ExcelWriter('results/Regressione_lasso_ct_no_PCA.xlsx')
df_regression.to_excel(writer)
writer.save()



###################################################################
##################### Ridge Regression Cancer type ############################
###################################################################

tipo = ['skin','aero_digestive_tract', 'breast',
        'nervous_system', 'urogenital_system',
        'digestive_system', 'blood', 'lung']
#t = 'nervous_system'
result_regression = []



name_columns = ['Y', 'Cancer type', 'Numerosità', 'SE','SSE', 'MSE', 
                'Root_MSE', 'RSE','RRSE', 'MAE', 'RAE','Deviance', 
                'Variance' ]


for t in tipo:
    
    df_regression = pd.DataFrame()#columns = name_columns)
    model = RidgeCV(alphas=(0.1, 0.5, 1.0, 10, 20, 30, 50, 100, 200))#, cv = 10)

    dati = complete_dataframe[complete_dataframe['Cancer Type']==t]

        
    
    for y in Y_array:
    
         data = create_dataset(data = dati, 
                               target_variable = y, 
                               explanatory_variable = X_matrix)
           
         n = len( data)
         print(y, t, n)
            
#         X_train = data[ X_matrix ]
#         y_train = data[ y ]
        
#            coefficients = lasso.coef_
#            alpha = lasso.alpha_
         regression = cross_validation(splits = n, 
                                       target_variable = y, 
                                       explanatory_variable = X_matrix,
                                       data = data,
                                       model = model)
                
         risultati =  [y, t, n] + regression
         result_regression.append( risultati )
    
        
    df_regression = pd.DataFrame(result_regression)
    df_regression.columns = name_columns
    df = df_regression
    
    
        
    
    
writer = pd.ExcelWriter('results/Regressione_ridge_ct_no_PCA.xlsx')
df_regression.to_excel(writer)
writer.save()
