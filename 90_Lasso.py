# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:23:07 2017

@author: amacaluso
"""


exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descriptive.py').read(), globals())
#exec(open("05_hypothesis_test.py").read(), globals())
exec(open("10_PCA.py").read(), globals())

#########################################################################
##################### ALL CELL LINE #####################################
#########################################################################



name_columns = ['Y', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                 'MAE', 'RAE','Deviance', 'Variance', 'alpha', 'coeff' ]

###################################################################
##################### Ridge Regression ###########################
###################################################################

from sklearn.linear_model import LassoCV
model = LassoCV(eps = 0.1, n_alphas = 100)#, cv = 20 )
result_regression = []


for i in range( 0, len(Y_array) ):
    print(i)
#    i=0
    print( Y_array[i] )
    n = len( list_data[i] )
    data = list_data[i]
    
    X_train = data[ explanatory_variable]
    y_train = data[ Y_array[i] ]

    lasso = model.fit( X_train, y_train )
    coefficients = lasso.coef_
    alpha = lasso.alpha_
    regression = cross_validation(splits = n, 
                                  target_variable = Y_array[i], 
                                  explanatory_variable = list_data[i][X_matrix].columns,
                                  data = list_data[i],
                                  model = lasso)
    
    risultati =  [Y_array[i], n] + regression + [alpha]
    risultati.append(np.ndarray.tolist(coefficients))
    result_regression.append( risultati )
        



df_regression = pd.DataFrame(result_regression)
df_regression.columns = name_columns
df_regression




######################################################################
######################################################################

comp = [10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450]

name_columns = ['Y', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                 'MAE', 'RAE','Deviance', 'Variance', 'alpha', 'N_componenti',
                 'coeff' ]
######################### Regressione lineare ######################################


df_regression = pd.DataFrame()#columns = name_columns)
result_regression = []

for c in comp:
#    c=10
    print(c)
    componenti = pd.DataFrame(pca.components_).ix[:,range(c)]
    
    dataset = pd.concat([data_X['ID'], componenti], axis = 1)
    dataset = pd.merge(dati_risposta, dataset , on = 'ID')
    X_matrix = componenti.columns
    
    
    Y_array = dati_risposta.columns[4:8]
    # Y_array
    list_data = []
    
    for y in Y_array:
        list_data.append( create_dataset(data = dataset, 
                                         target_variable = y, 
                                         explanatory_variable = X_matrix))
    
    explanatory_variable = list_data[1][X_matrix].columns
    #    data = list_data[1] 
    
    
    
    lasso = LassoCV(eps = 0.1, n_alphas = 100, cv = 20 )

    
    
    for i in range( 0, len(Y_array) ):
        print(i)
    #    i=0
        print(Y_array[i])
        n = len( list_data[i])
        data = list_data[i]
        
        X_train = data[ explanatory_variable]
        y_train = data[ Y_array[i] ]
    
        lasso = model.fit( X_train, y_train )
        coefficients = lasso.coef_
        alpha = lasso.alpha_
        regression = cross_validation(splits = n, 
                                      target_variable = Y_array[i], 
                                      explanatory_variable = list_data[i][X_matrix].columns,
                                      data = list_data[i],
                                      model = lasso)
        
        risultati =  [Y_array[i], n] + regression + [alpha] +[c]
        risultati.append(np.ndarray.tolist(coefficients))
        result_regression.append( risultati )
        
df_regression = pd.DataFrame( result_regression )
df_regression.columns = name_columns

writer = pd.ExcelWriter('results/lasso/LASSO.xlsx')
df_regression.to_excel(writer)
writer.save()


df = df_regression

import matplotlib.pyplot as plt

xb1 = df.loc[df['Y'] == 'BMS-708163_AUC', 'N_componenti']
yb1 = df.loc[df['Y'] == 'BMS-708163_AUC', 'RRSE']
xz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'N_componenti']
yz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'RRSE']
    
xb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'N_componenti']
yb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'RRSE']
xz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'N_componenti']
yz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'RRSE']
    
    
plt.plot(xb1 , yb1, 'ro-' , label = 'BMS-708163_AUC')
plt.plot(xz1 , yz1, 'bs-', color ='b', label = 'BMS-708163_IC_50')
plt.plot(xb2 , yb2, 'ro-' , color ='y', label = 'Z-LLNle-CHO_AUC')
plt.plot(xz2 , yz2, 'bs-', color ='g', label = 'Z-LLNle-CHO_IC_50')
hlines(1, 0, max(comp), linestyles= 'dashed')
plt.title('LASSO')
plt.ylabel('RRSE')
plt.xlabel('Componenti')
plt.legend()
savefig('results/lasso/'+ 'LASSO' + '.png')
plt.close()

######################################################################
########################################################################


comp = [10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450]

import matplotlib.pyplot as plt

name_columns = ['Y', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                 'MAE', 'RAE','Deviance', 'Variance', 'n_componenti' ]

Y_array = dati_risposta.columns[4:8]

tipo = ['skin','aero_digestive_tract', 'breast',
        'nervous_system', 'urogenital_system',
        'digestive_system', 'blood', 'lung']

for t in tipo:
    n_components = []
    explained_variance = []
    
#    t = 'blood'
    
    dati = complete_dataframe[complete_dataframe['Cancer Type']==t]
    X = dati.ix[:, 8:18924]
    
    print('Analisi delle componenti principali ... ')
    for i in range(10, 500, 10):
        n_components.append(i)
        pca = PCA(n_components=i)
        pca.fit(X)
        explained_variance.append(sum(pca.explained_variance_ratio_))
    ######################### GRAFICO ######################################
    #fig, ax = plt.subplots()
    #ax.scatter(n_components, explained_variance)
    #############################################################
    
    df_regression = pd.DataFrame()#columns = name_columns)
    result_regression = []
    
    for c in comp:
        print(c)
    #==============================================================================
    # #Û    c=20
    #==============================================================================
        componenti = pd.DataFrame(pca.components_).ix[:,range(c)].reset_index(drop=True)
        dataset_ID = pd.DataFrame(dati['ID']).reset_index(drop=True)
        temp = [dataset_ID, componenti]
        dati_correnti = pd.concat(temp, axis = 1)
    
        dataset = pd.merge(dati_risposta, dati_correnti , on = 'ID')
        X_matrix = componenti.columns
        model = LassoCV(eps = 0.1, n_alphas = 100, cv = 20 )

        
    
        for y in Y_array:
    
            data = create_dataset(data = dataset, 
                                  target_variable = y, 
                                  explanatory_variable = X_matrix)
           
            n = len( data)
            print(y, t, n)
    
#            data = list_data[i]
        
            X_train = data[ X_matrix ]
            y_train = data[ y ]
        
            lasso = model.fit( X_train, y_train )
            coefficients = lasso.coef_
            alpha = lasso.alpha_
            regression = cross_validation(splits = n, 
                                          target_variable = y, 
                                          explanatory_variable = X_matrix,
                                          data = data,
                                          model = lasso)
                
            risultati =  [y, n] + regression + [c]
            result_regression.append( risultati )
    
        
    df_regression = pd.DataFrame(result_regression)
         
        
        
    df_regression.columns = name_columns
    df = df_regression
        
        
    xb1 = df.loc[df['Y'] == 'BMS-708163_AUC', 'n_componenti']
    yb1 = df.loc[df['Y'] == 'BMS-708163_AUC', 'RRSE']
    xz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'n_componenti']
    yz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'RRSE']
        
    xb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'n_componenti']
    yb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'RRSE']
    xz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'n_componenti']
    yz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'RRSE']
    
    
    plt.plot(xb1 , yb1, 'ro-' , label = 'BMS-708163_AUC')
    plt.plot(xz1 , yz1, 'bs-', color ='b', label = 'BMS-708163_IC_50')
    plt.plot(xb2 , yb2, 'ro-' , color ='y', label = 'Z-LLNle-CHO_AUC')
    plt.plot(xz2 , yz2, 'bs-', color ='g', label = 'Z-LLNle-CHO_IC_50')
    hlines(1, 0, max(comp), linestyles= 'dashed')# [‘solid’ | | ‘dashdot’ | ‘dotted’], optional)
    plt.title(t)
    plt.ylabel('RRSE')
    plt.xlabel('Componenti')
    plt.legend()
    savefig('results/ridge_regression/Cancer_type/'+ t + '.png')
    plt.close()

   

