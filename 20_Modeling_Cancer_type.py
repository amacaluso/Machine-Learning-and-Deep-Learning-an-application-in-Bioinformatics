# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:38:44 2017

@author: Antonio
"""


exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descriptive.py').read(), globals())
#exec(open("05_hypothesis_test.py").read(), globals())
exec(open("10_PCA.py").read(), globals())
#exec(open("15_Modeling_all_cell_line.py").read(), globals())



dataset["combined_BMS"] = dataset[ "BMS-708163_IC_50" ] * dataset[ "BMS-708163_AUC"] 
dataset["combined_Z-LLNl"] = dataset[ "Z-LLNle-CHO_IC_50" ] * dataset[ "Z-LLNle-CHO_AUC"] 

Y_array = pd.Series( Y_array )
target_variables = list(Y_array)

target_variables.append("combined_BMS")  
target_variables.append("combined_Z-LLNl")


###################################################################
##################### Linear Regression ###########################
###################################################################


name_columns = ['Y', 'Cancer type', 'Numerosità', 'SE',
                'SSE', 'Deviance','MSE', 'Variance', 'Root_MSE', 'RRSE',
                'RSE', 'RAE', 'MAE']

df_regression_ct = pd.DataFrame( )
        
        
for y in target_variables:
    
    print( y , "\n\n")
    
    for t in tipo:
        
        print( t )

        train_data = dataset[ dataset['Cancer Type'] != t]
        test_data = dataset[ dataset['Cancer Type'] == t]
        
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
            
        risultati =  pd.DataFrame([ y, t, n, SE, SSE, Dev_Y, MSE, Var_Y, 
                                   Root_MSE, RSE, RRSE, RAE, MAE ]).transpose()
            
        df_regression_ct = df_regression_ct.append( risultati )

df_regression_ct.columns = name_columns  
    
writer = ExcelWriter('results/Cancer_type/Regressione_lineare_ct.xlsx')
df_regression_ct.to_excel(writer)
writer.save()


#np.save("results/Cancer_type/Risultati_regressione.npy", df_regression_ct)
#df_regression_ct.to_csv("results/CSV/Cancer_type/Risultati_reg.csv", sep=';')




###################################################################
############### Support Verctor Machine ###########################
###################################################################


df_SVM_ct = pd.DataFrame( columns = name_columns )


for y in target_variables:
    print( y , "\n\n")
    
    for t in tipo:
        print( t )
        train_data = dataset[ dataset['Cancer Type'] != t]
        test_data = dataset[ dataset['Cancer Type'] == t]
        train_dataset = create_dataset(data = train_data,
                                       target_variable = y,
                                       explanatory_variable = X_matrix)
    
    
        test_dataset = create_dataset(data = test_data,
                                      target_variable = y,
                                      explanatory_variable = X_matrix)
        
        parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
                      'C':[1,3,5,7,9,11,13,15,17,19], 
                      'gamma': [0.01,0.03,0.04,0.1,0.2,0.4,0.6]
                       }
      
        
        svr = svm.SVR()
        grid = GridSearchCV(svr, parameters, n_jobs = 4)
        X_train = train_dataset[ X_matrix]
        y_train = train_dataset[ y ]
        
        print( "Scelta dei parametri \n")
        
        start_time = time.time()
        SVM = grid.fit( X_train, y_train )
        end_time = time.time()
        time_spent = end_time-start_time
 #       print("--- %s seconds ---" % time_spent),"\n\n")
    
        print( SVM.best_params_ ,"\n\n")
        print("Stima del modello \n")
            
        
        prediction = model_estimation(data = train_dataset, 
                                      target_variable = y,
                                      explanatory_variable = X_matrix,
                                      test_data = test_dataset,
                                      model = SVM)
        
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

        media = np.mean(Y_true)
    
        differenza_media =  Y_true - media
        
        if (sum( abs(differenza_media)) != 0):
            RAE = sum( abs(differenza) )/sum( abs(differenza_media) )
        else:
            RAE = None
                     
        risultati =  pd.DataFrame([ y, t, n, SE, SSE, Dev_Y, MSE, Var_Y, 
                                   Root_MSE, RSE, RRSE, RAE, MAE ]).transpose()
        
        
        df_SVM_ct = df_SVM_ct.append (risultati )
         
np.save("results/Cancer_type/Risultati_SVM_ct.py", df_SVM_ct)

writer = ExcelWriter('results/Cancer_type/SVM_ct.xlsx')
df_SVM_ct.to_excel(writer)
writer.save()


#df_SVM_ct.to_csv("results/CSV/Cancer_type/Risultati_SVM_ct.csv", sep=';')
#dati_prova = pd.concat([y_train, X_train], axis = 1)
#np.save( "results/dataset.npy", dati_prova )
#dati_prova.to_csv("results/CSV/dataset.csv", sep=';')

###################################################################
######################## Nerual Network ###########################
###################################################################


df_nn_ct = pd.DataFrame() #columns = name_columns )


for y in target_variables:
    print( y , "\n\n")
    
    for t in tipo:
        print( t )
        train_data = dataset[ dataset['Cancer Type'] != t]
        test_data = dataset[ dataset['Cancer Type'] == t]

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
                                             [4, 2],
                                             [2]],
                      'activation' : ['logistic'],
                      'max_iter': [6000] }
        
        nn_reg = nn.MLPRegressor()
        grid = GridSearchCV(nn_reg, param_grid = parameters, n_jobs = 3)
            
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
                     
        media = np.mean(Y_true)
    
        differenza_media =  Y_true - media
        
        if (sum( abs(differenza_media)) != 0):
            RAE = sum( abs(differenza) )/sum( abs(differenza_media) )
        else:
            RAE = None
                     
        risultati =  pd.DataFrame([ y, t, n, SE, SSE, Dev_Y, MSE, Var_Y, 
                                   Root_MSE, RSE, RRSE, RAE, MAE ]).transpose()
            
        df_nn_ct = df_nn_ct.append(risultati )

df_nn_ct.columns = name_columns  


writer = ExcelWriter('results/Cancer_type/MLP_ct.xlsx')
df_nn_ct.to_excel(writer)
writer.save()

np.save("results/Cancer_type/Risultati_nn_ct.py", df_nn_ct)
# df_nn_ct.to_csv("results/CSV/Cancer_type/Risultati_nn_ct.csv", sep=';')
#dati_prova = pd.concat([y_train, X_train], axis = 1)


