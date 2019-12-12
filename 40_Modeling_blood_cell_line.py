# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 11:27:55 2017

@author: Antonio
"""

exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
# exec(open('03_Descriptive.py').read(), globals())
# exec(open("05_hypothesis_test.py").read(), globals())
exec(open("10_PCA.py").read(), globals())


dataset["combine_BMS"] = dataset[ "BMS-708163_IC_50" ] * dataset[ "BMS-708163_AUC"] 
dataset["combine_Z-LLNl"] = dataset[ "Z-LLNle-CHO_IC_50" ] * dataset[ "Z-LLNle-CHO_AUC"] 

Y_array = pd.Series( Y_array )
target_variables = list(Y_array)

target_variables.append("combine_BMS")  
target_variables.append("combine_Z-LLNl")


df_regression_blood = pd.DataFrame()

name_columns = ['Y', 'Cancer type', 'Numerosità', 'SE',
                'SSE', 'Deviance', 'MSE', 'Variance', 'Root_MSE', 'RSE',
                'RRSE', 'RAE', 'MAE' ]

type_ct = "blood"     

for y in target_variables:
    
#    y = target_variables[0]
    print( y , "\n\n")
    train_data = dataset[ dataset['Cancer Type'] == type_ct]
    test_data = dataset[ dataset['Cancer Type'] != type_ct]
        
    train_dataset = create_dataset(data = train_data,
                                   target_variable = y,
                                   explanatory_variable = X_matrix)
    
    
    test_dataset = create_dataset(data = test_data,
                                  target_variable = y,
                                  explanatory_variable = X_matrix)
        
    prediction = model_estimation(data = train_dataset, 
                                  target_variable = y,
                                  explanatory_variable = X_matrix,
                                  test_data = test_dataset)

    
##############################################################################
##############################################################################
##############################################################################

    Y_hat = prediction
    Y_true = test_dataset[y]
    n = len( test_dataset)
#    stats.stats.pearsonr( Y_hat, Y_true)   
#    plt.plot( Y_hat, Y_true,'ro')

    differenza = Y_hat - Y_true
    differenza_2 = np.power(differenza, 2)
            
    SSE = sum(differenza_2)
    MSE = SSE/n
    Root_MSE = sqrt(MSE)
    SE = np.sum(differenza)
        
    Var_Y = np.var( Y_true )
    Dev_Y = np.var( Y_true )*n
        
    if (Dev_Y == 0):
        RSE = None
        RRSE = None
    else:
        RSE = SSE/Dev_Y
        RRSE = sqrt(RSE)
            
    MAE = sum( abs(differenza) )/n


    differenza_media = []
    media = np.mean(Y_true)
    
    differenza_media =  Y_true - media
       
    if (sum( abs(differenza_media)) != 0):
        RAE = sum( abs(differenza) )/sum( abs(differenza_media) )
    else:
        RAE = None
        
 

                 
    risultati =  pd.DataFrame([ y, type_ct, n, SE, SSE, Dev_Y, MSE, Var_Y, 
                               Root_MSE, RSE, RRSE, RAE, MAE ]).transpose()
    
        
    df_regression_blood = df_regression_blood.append(risultati )

df_regression_blood.columns = name_columns
    
writer = pd.ExcelWriter('results/train_blood/Regressione_lineare_blood.xlsx')
df_regression_blood.to_excel(writer)
writer.save()


#np.save("results/train_blood/Risultati_regressione_blood.npy", df_regression_blood)
#df_regression_blood.to_csv("results/CSV/train_blood/Risultati_reg_blood.csv", sep=';')


###################################################################
############### Support Verctor Machine ###########################
###################################################################


df_svm_blood = pd.DataFrame()



for y in target_variables:
    
#    y = target_variables[0]
    print( y , "\n\n")
    train_data = dataset[ dataset['Cancer Type'] == type_ct]
    test_data = dataset[ dataset['Cancer Type'] != type_ct]
        
    train_dataset = create_dataset(data = train_data,
                                   target_variable = y,
                                   explanatory_variable = X_matrix)
    
    
    test_dataset = create_dataset(data = test_data,
                                  target_variable = y,
                                  explanatory_variable = X_matrix)
    
    
    parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
                  'C':[1,3,5,7,9,11,13,15,17,19], 
                  'gamma': [0.01,0.03,0.04,0.1,0.2,0.4,0.6]}
    svr = svm.SVR()
    grid = GridSearchCV(svr, parameters, n_jobs = 2)
    X_train = train_dataset[ explanatory_variable]
    y_train = train_dataset[ y ]
    
    
    print( "Scelta dei parametri \n")
    
    start_time = time.time()

    
    SVM = grid.fit( X_train, y_train )
    print("--- %s seconds ---" % (time.time() - start_time),"\n\n")

    print( grid.best_params_ ,"\n\n")
    print("Stima del modello \n")
    
    
        
    prediction = model_estimation(data = train_dataset, 
                                  target_variable = y,
                                  explanatory_variable = X_matrix,
                                  test_data = test_dataset,
                                  model = SVM)
    
##############################################################################
##############################################################################
##############################################################################

    Y_hat = prediction
    Y_true = test_dataset[y]
    n = len( test_dataset)
#    stats.stats.pearsonr( Y_hat, Y_true)   
#    plt.plot( Y_hat, Y_true,'ro')

    differenza = Y_hat - Y_true
    differenza_2 = np.power(differenza, 2)
            
    SSE = sum(differenza_2)
    MSE = SSE/n
    Root_MSE = sqrt(MSE)
    SE = np.sum(differenza)
        
    Var_Y = np.var( Y_true )
    Dev_Y = np.var( Y_true )*n
        
    if (Dev_Y == 0):
        RSE = None
        RRSE = None
    else:
        RSE = SSE/Dev_Y
        RRSE = sqrt(RSE)
            
    MAE = sum( abs(differenza) )/n
    
    differenza_media = []
    media = np.mean(Y_true)
    
    differenza_media =  Y_true - media
       
    if (sum( abs(differenza_media)) != 0):
        RAE = sum( abs(differenza) )/sum( abs(differenza_media) )
    else:
        RAE = None
        
              
    risultati =  pd.DataFrame([ y, type_ct, n, SE, SSE, Dev_Y, MSE, Var_Y, 
                               Root_MSE, RSE, RRSE, RAE, MAE ]).transpose()

    df_svm_blood = df_svm_blood.append(risultati )



df_svm_blood.columns = name_columns
df_svm_blood

writer = pd.ExcelWriter('results/train_blood/SVM_blood.xlsx')
df_svm_blood.to_excel(writer)
writer.save()

    
#np.save("results/train_blood/Risultati_svm_blood.npy", df_svm_blood)
#df_svm_blood.to_csv("results/CSV/train_blood/Risultati_svm_blood.csv", 
                    # sep=';')



###################################################################
######################## Neural Network ###########################
###################################################################

from sklearn.grid_search import GridSearchCV
# from sklearn import ae, mlp
import sklearn.neural_network as nn

df_nn_blood = pd.DataFrame()



for y in target_variables:
    
#    y = target_variables[0]
    print( y , "\n\n")
    train_data = dataset[ dataset['Cancer Type'] == type_ct]
    test_data = dataset[ dataset['Cancer Type'] != type_ct]
        
    train_dataset = create_dataset(data = train_data,
                                   target_variable = y,
                                   explanatory_variable = X_matrix)
    
    
    test_dataset = create_dataset(data = test_data,
                                  target_variable = y,
                                  explanatory_variable = X_matrix)
    
    
    parameters = {'learning_rate': ['constant', 'adaptive'],
                  'hidden_layer_sizes': [[64, 32, 16, 8, 4, 2],
                                         [48, 36, 24, 12, 4], 
                                         [24, 12, 6, 3, 1],
                                         [10, 5, 3],
                                         [8, 4, 2],
                                         [6, 3],
                                         [4, 2],
                                         [6],
                                         [4],
                                         [2]],
                  'activation' : ['identity',
                                  'logistic'],
                  'max_iter': [6000] }
    
    nn_reg = nn.MLPRegressor()
    grid = GridSearchCV(nn_reg, param_grid = parameters, n_jobs = 2)
        
    X_train = train_dataset[ explanatory_variable]
    y_train = train_dataset[ y ]
    
    print( "Scelta dei parametri Reti Neurali MLP \n")
    
    start_time = time.time()
    
    NeurNet = grid.fit( X_train, y_train)
    
    print("--- %s seconds ---" % (time.time() - start_time),"\n\n")

    print( NeurNet.best_params_ ,"\n\n")
    print("Stima del modello \n")
        
    
        
    prediction = model_estimation(data = train_dataset, 
                                  target_variable = y,
                                  explanatory_variable = X_matrix,
                                  test_data = test_dataset,
                                  model = NeurNet)
    
##############################################################################
##############################################################################
##############################################################################

    Y_hat = prediction
    Y_true = test_dataset[y]
    n = len( test_dataset)
    differenza = Y_hat - Y_true
    differenza_2 = np.power(differenza, 2)
            
    SSE = sum(differenza_2)
    MSE = SSE/n
    Root_MSE = sqrt(MSE)
    SE = np.sum(differenza)
        
    Var_Y = np.var( Y_true )
    Dev_Y = np.var( Y_true )*n
        
    if (Dev_Y == 0):
        RSE = None
        RRSE = None
    else:
        RSE = SSE/Dev_Y
        RRSE = sqrt(RSE)
            
    MAE = sum( abs(differenza) )/n
    
    differenza_media = []
    media = np.mean(Y_true)
    
    differenza_media =  Y_true - media
       
    if (sum( abs(differenza_media)) != 0):
        RAE = sum( abs(differenza) )/sum( abs(differenza_media) )
    else:
        RAE = None
        
              
    risultati =  pd.DataFrame([ y, type_ct, n, SE, SSE, Dev_Y, MSE, Var_Y, 
                               Root_MSE, RSE, RRSE, RAE, MAE ]).transpose()

        
    df_nn_blood = df_nn_blood.append(risultati )

       

df_nn_blood.columns = name_columns


writer = pd.ExcelWriter('results/train_blood/MLP_blood.xlsx')
df_nn_blood.to_excel(writer)
writer.save()

    
#np.save("results/train_blood/Risultati_nn_blood.npy", df_nn_blood)
#df_nn_blood.to_csv("results/CSV/train_blood/Risultati_nn_blood.csv", sep=';')


