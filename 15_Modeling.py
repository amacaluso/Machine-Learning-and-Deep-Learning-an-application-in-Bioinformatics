# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 19:03:24 2017

@author: Antonio
"""

exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
# exec(open('03_Descrittive.py').read(), globals())
# exec(open("05_Test_d_ipotesi.py").read(), globals())
exec(open("10_PCA.py").read(), globals())


name_columns = ['Y', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                 'MAE', 'RAE','Deviance', 'Variance', 'Modello' ]

###################################################################
##################### Linear Regression ###########################
###################################################################


model = skl.linear_model.LinearRegression()
result_regression = []


for i in range( 0, len(Y_array) ):
    print(i)
#    i=0
    print(Y_array[i])
    n = len( list_data[i])

    regression = cross_validation(splits = n, 
                                  target_variable = Y_array[i], 
                                  explanatory_variable = list_data[i][X_matrix].columns,
                                  data = list_data[i] )
    
    risultati =  [Y_array[i], n] + regression + ['reg_lin']
    result_regression.append( risultati )
        



df_regression = pd.DataFrame(result_regression)
df_regression.columns = name_columns
df_regression
#
np.save("results/all_cell_line/all_cell_CL_regression.npy", df_regression)
# pd.DataFrame(df_regression).to_excel("results/CSV/Risultati_regression_all_CL.csv")
# np.load("results/all_cell_CL_regression.npy")

## from pandas import ExcelWriter

writer = ExcelWriter('results/Regressione_lineare_ALL.xlsx')
df_regression.to_excel(writer)
writer.save()

   


###################################################################
############### Support Verctor Machine ###########################
###################################################################

result_svm_list = []

for i in range( 0, len(Y_array) ) :
    print( i )
    print( Y_array[i] )
    data = list_data[i]
    parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
                  'C':[1,3,5,7,9,11,13,15,17,19], 
                      'gamma': [0.01,0.03,0.04,0.1,0.2,0.4,0.6]}
    svr = svm.SVR()
    grid = GridSearchCV(svr, parameters, n_jobs = 2)
    X_train = data[ explanatory_variable]
    y_train = data[ Y_array[i] ]
    
    
    print( "Scelta dei parametri \n")
    
    start_time = time.time()

    
    SVM = grid.fit( X_train, y_train )
    print("--- %s seconds ---" % (time.time() - start_time),"\n\n")

    print( grid.best_params_ ,"\n\n")
    print("Stima del modello \n")
    n = len( list_data[i])

    result_svm = cross_validation(splits = 20, 
                                  target_variable = Y_array[i],
                                  explanatory_variable = list_data[i][X_matrix].columns,
                                  data = list_data[i],
                                  model = SVM)
    
    risultati =  [Y_array[i], n] + result_svm +['SVM']
    result_svm_list.append( risultati )
    

df_svm = pd.DataFrame(result_svm_list)
df_svm.columns = name_columns
df_svm
#
np.save("results/all_cell_line/all_cell_CL_svm.npy", df_svm)
writer = ExcelWriter('results/SVM_ALL.xlsx')
df_svm.to_excel(writer)
writer.save()



###################################################################
############### Neural Network MLP ################################
###################################################################

from sklearn.grid_search import GridSearchCV
# from sklearn import ae, mlp
import sklearn.neural_network as nn
#from sklearn.neural_network import Layer

result_mlp_list = []


for i in range( 0, len(Y_array) ) :
 #   i = 1
    print( i )
    print( Y_array[i] )
    data = list_data[i]
    
    parameters = {'learning_rate': ['constant', 'adaptive'],
                  'hidden_layer_sizes': [[64, 32, 16, 8, 4, 2],
                                         [48, 36, 24, 12, 4], 
                                         [24, 12, 6, 3, 1],
                                         [10, 5, 3],
                                         [4, 2],
                                         [2]],
                  'activation' : [#'identity',
                                  'logistic'],
                  'max_iter': [60000] }
    
    nn_reg = nn.MLPRegressor()
    grid = GridSearchCV(nn_reg, param_grid = parameters, n_jobs = 3)
        
    X_train = data[ explanatory_variable]
    y_train = data[ Y_array[i] ]
    
    print( "Scelta dei parametri Reti Neurali MLP \n")
    
    start_time = time.time()
    
    NeurNet = grid.fit( X_train, y_train)
    
    print("--- %s seconds ---" % (time.time() - start_time),"\n\n")

    print( NeurNet.best_params_ ,"\n\n")
    print("Stima del modello \n")
    
    n = len( list_data[i])
    
    result_nn = cross_validation( splits = 20, 
                                  target_variable = Y_array[i],
                                  explanatory_variable = list_data[i][X_matrix].columns,
                                  data = list_data[i],
                                  model = NeurNet)
    
    risultati =  [Y_array[i], n] + result_nn + ['MLP']
    result_mlp_list.append( risultati )
    
#    result_svm_list.append( result_svm )
    

df_nn = pd.DataFrame(result_mlp_list)
df_nn.columns = name_columns
df_nn

np.save("results/all_cell_line/all_cell_line_nn.npy", df_nn)

writer = ExcelWriter('results/MLP_ALL.xlsx')
df_nn.to_excel(writer)
writer.save()

#df = pd.concat([df_regression, df_svm, df_nn])
#writer = ExcelWriter('results/models_ALL.xlsx')
#df.to_excel(writer)
#writer.save()


