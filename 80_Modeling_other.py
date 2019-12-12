# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 23:11:29 2017

@author: Antonio
"""


exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descriptive.py').read(), globals())
#exec(open("05_hypothesis_test.py").read(), globals())

#######################################################################
#######################################################################

Y_array = dati_risposta.columns[4:8]
X_names = X.columns
tipo = list(set(dati_risposta.ix[:,2]))
data = complete_dataframe

cor_df = pd.DataFrame( )

# cor_df.index = X_names
list_cor = []
for y in Y_array:
#    y = Y_array[0]
    current_data = data.dropna(subset = [y])
    correlation_vec = []
    for x in X_names:
        correlation_vec.append( np.corrcoef(current_data[y], 
                                            current_data[x])[1,0])
    list_cor.append( correlation_vec )

cor_df = pd.DataFrame( list_cor )
cor_df.columns = X_names
cor_df.index = Y_array
cor_df = cor_df.transpose()
cor_df = abs(cor_df)
#######################################################################
#######################################################################

name_columns = ['Y', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                'MAE', 'RAE','Deviance', 'Variance', 'n_var','Modello' ]

result_regression = []

ridge = RidgeCV(alphas=(0.1, 0.5, 1.0, 10, 20, 30, 50, 100, 200))#, cv = 10)
result_ridge = []

lasso = LassoCV(eps = 0.1, n_alphas = 100)#, cv = 20 )
result_lasso = []

########################################################################
########################################################################
##################### Regressione Lineare ##############################
########################################################################

var_set = [10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450]

    
for y in Y_array:

    for v in  var_set:
        
        print( y, v, '\n\n\n')
        
        variables = cor_df.nlargest( n = v, columns = y ).index 
        dataset = create_dataset ( data, y, variables) 
        n = len(dataset)
        
################## REGRESSIONE LINEARE CLASSICA #################
        print('Regressione Lineare \n')

        regression = cross_validation(splits = n, 
                                      target_variable = y, 
                                      explanatory_variable = variables,
                                      data = dataset )
        
        risultati_reg =  [y, n] + regression + [v] + ['reg_lin']
        result_regression.append( risultati_reg )

#################################################################    

######################### RIDGE REGRESSION ######################
        print('Regressione Ridge \n')
         
        regression = cross_validation(splits = n, 
                                      target_variable = y, 
                                      explanatory_variable = variables,
                                      data = dataset,
                                      model = ridge)
        
        risultati_ridge =  [y, n] + regression + [v] + ['ridge']
        result_ridge.append( risultati_ridge )

#################################################################

######################### RIDGE REGRESSION ######################
        print('Regressione Lasso \n')

        regression = cross_validation(splits = n, 
                                      target_variable = y, 
                                      explanatory_variable = variables,
                                      data = dataset,
                                      model = lasso)
        
        risultati_lasso =  [y, n] + regression + [v] + ['lasso']
        result_lasso.append( risultati_lasso )

#################################################################
    


df_regression = pd.DataFrame(result_regression)
df_lasso = pd.DataFrame(result_lasso)
df_ridge = pd.DataFrame(result_ridge)

all_regression = df_regression.append( df_lasso).append(df_ridge)

all_regression.columns = name_columns
df_regression
    
# np.corrcoef( df_regression['n_var'], df_regression['RSE'])

writer = pd.ExcelWriter('results/no_PCA/Regressione_lineare_ALL.xlsx')
df_regression.to_excel(writer)
writer.save()


########################################################################
########################################################################
########################################################################



########################################################################
########################################################################
##################### MLP E SVM ########################################
########################################################################

var_set = [ 2,  10, 20, 50, 70, 100, 300]
Y_array = ['BMS-708163_IC_50', 'Z-LLNle-CHO_IC_50']

result_mlp_list = []
result_svm_list = []
    
for y in Y_array:

    for v in  var_set:
        
        print( y, v, '\n\n\n')
        
        variables = cor_df.nlargest( n = v, columns = y ).index 
        dataset = create_dataset ( data, y, variables) 
        n = len(dataset)
        
################################ SVM ############################
        print('Support Vector Machine \n')
        
        ## ****************** Scelta parametri *****************##
        
        parameters = {'kernel':('poly', 'rbf'),
                      'C':[1,3,9,13,17], 
                      'gamma': [0.01,0.04,0.1,0.4,0.6]}


        svr = svm.SVR()
        grid = GridSearchCV(svr, parameters, n_jobs = 3)
        X_train = dataset[ variables ]
        y_train = dataset[ y ]
            
        print( "Scelta dei parametri")
        start_time = time.time()
        SVM = grid.fit( X_train, y_train )
        print("--- %s seconds ---" % (time.time() - start_time))
        print( SVM.best_params_ )
 
            ## ************************************************** ##
        
        result_svm = cross_validation(splits = min(n,5), 
                                      target_variable = y,
                                      explanatory_variable = variables,
                                      data = dataset,
                                      model = SVM)
            
        risultati =  [y, n] + result_svm + [v] + ['SVM']
        result_svm_list.append( risultati )
 
##################################################################    

################################### MLP ##########################
        print('Multilayer Perceptron\n')
        
        ## ****************** Scelta parametri *****************##
         
        parameters = {'learning_rate': ['constant', 'adaptive'],
                      'hidden_layer_sizes': [#[64, 32, 16, 8, 4, 2],
                                             [48, 36, 24, 12, 4], 
                                             [24, 12, 6, 3, 1],
                                             [10, 5, 3],
                                             [4, 2],
                                             [2]],
                      'activation' : [#'identity',
                                      'logistic'],
                      'max_iter': [6000] }



        
        nn_reg = nn.MLPRegressor()
        grid = GridSearchCV(nn_reg, param_grid = parameters, n_jobs = 3)
            
        X_train = dataset[ variables]
        y_train = dataset[ y ]
        
        print( "Scelta dei parametri Reti Neurali MLP")
        start_time = time.time()
        NeurNet = grid.fit( X_train, y_train)
        print("--- %s seconds ---" % (time.time() - start_time))
        print( NeurNet.best_params_)

        ## ************************************************** ##
        result_nn = cross_validation( splits = min(n,5), 
                                      target_variable = y,
                                      explanatory_variable = variables,
                                      data = dataset,
                                      model = NeurNet )
        
        risultati =  [ y, n] + result_nn + [v] + ['MLP']
        result_mlp_list.append( risultati )


#################################################################

df_mlp = pd.DataFrame(result_mlp_list)
df_svm = pd.DataFrame(result_svm_list)
df_ML = df_mlp.append( df_svm )

df_ML.columns = name_columns

    

writer = pd.ExcelWriter('results/no_PCA/MLP_ALL.xlsx')
df_ML.to_excel(writer)
writer.save()


    
plot_error(df = all_regression, 
           ordinata='RSE', 
           ascissa = 'n_var',
           Y_array = list(set(all_regression[ 'Y' ] )),
           group = list(set(all_regression[ 'Modello' ] )),
           name_file = 'REG_')
#        
#        plt.plot(xz1 , yz1, 'bs-', color ='b', label = 'BMS-708163_IC_50')
#        plt.plot(xb2 , yb2, 'ro-' , color ='y', label = 'Z-LLNle-CHO_AUC')
#        plt.plot(xz2 , yz2, 'bs-', color ='g', label = 'Z-LLNle-CHO_IC_50')
#        hlines(1, 0, max(comp), linestyles= 'dashed')# [‘solid’ | | ‘dashdot’ | ‘dotted’], optional)
#        plt.title('MLP')
#        plt.ylabel('RRSE')
#        plt.xlabel('Componenti')
#        plt.legend()
#        savefig('results/Immagini/'+ 'MLP_all' + '.png')
        

    
#xz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'n_componenti']
#yz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'RRSE']
#    
#xb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'n_componenti']
#yb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'RRSE']
#xz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'n_componenti']
#yz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'RRSE']
    
    

















