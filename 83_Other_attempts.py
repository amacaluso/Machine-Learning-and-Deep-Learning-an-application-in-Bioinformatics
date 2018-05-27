# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 23:11:29 2017

@author: Antonio
"""


exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descrittive.py').read(), globals())
#exec(open("05_Test_d_ipotesi.py").read(), globals())

#######################################################################
#######################################################################

Y_array = dati_risposta.columns[4:8]
X_names = X.columns
tipo = list(set(dati_risposta.ix[:,2]))
data = complete_dataframe

training_type_ct = "blood"

train_data = data[ data['Cancer Type'] == training_type_ct]


cor_df = pd.DataFrame( )

# cor_df.index = X_names
list_cor = []



for y in Y_array:
#    y = Y_array[0]
    current_data = train_data.dropna(subset = [y])
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

result_regression = []

ridge = RidgeCV(alphas=(0.1, 0.5, 1.0, 10, 20, 30, 50, 100, 200))#, cv = 10)
result_ridge = []

lasso = LassoCV(eps = 0.1, n_alphas = 100)#, cv = 20 )
result_lasso = []

########################################################################
########################################################################
##################### Regressione Lineare ##############################
########################################################################
tipo = ['bone', 'kidney', 'pancreas', 'breast',
        'lung', 'soft_tissue', 'thyroid', 'digestive_system',
        'aero_digestive_tract', 'nervous_system', 'skin',
        'urogenital_system']

var_set = [10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450]
df_result_regressions = pd.DataFrame()
    
for y in Y_array:

    for v in  var_set:
        
        print( y, v, '\n\n\n')
        

        variables = cor_df.nlargest( n = v, columns = y ).index 
        train_dataset = create_dataset ( train_data, y, variables) 
#        n = len(dataset)
        
################## REGRESSIONE LINEARE CLASSICA #################
        print('Regressione Lineare \n')
        
        for t in tipo:
            test_data = complete_dataframe
            test_data = test_data[ test_data['Cancer Type'] == t]
            test_dataset = create_dataset ( test_data, y, variables )
            
            prediction = model_estimation(data = train_dataset, 
                                          target_variable = y,
                                          explanatory_variable = variables,
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
                
                      
            risultati_reg_lin =  pd.DataFrame([ y, t, n, SE, 
                                               SSE, Dev_Y, MSE, 
                                               Var_Y, Root_MSE, 
                                               RSE, RRSE, RAE, MAE ] + 
                                               [v,'Reg_lin']).transpose()
                
                
            df_result_regressions = df_result_regressions.append(risultati_reg_lin )
                
    
    #################################################################    
    
    ######################### RIDGE REGRESSION ######################
            print('Regressione Ridge \n')
             
            prediction = model_estimation(data = train_dataset, 
                                          target_variable = y,
                                          explanatory_variable = variables,
                                          test_data = test_dataset,
                                          model = ridge)
    
    
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
                
                      
            risultati_ridge =  pd.DataFrame([ y, t, n, SE, SSE, 
                                             Dev_Y, MSE, Var_Y, 
                                             Root_MSE, RSE, RRSE, 
                                             RAE, MAE ] + [v,'Ridge']).transpose()
                
                
            df_result_regressions = df_result_regressions.append( risultati_ridge )
            
    #################################################################
    
    ######################### LASSO REGRESSION ######################
            print('Regressione Lasso \n')
    
            prediction = model_estimation(data = train_dataset, 
                                          target_variable = y,
                                          explanatory_variable = variables,
                                          test_data = test_dataset,
                                          model = lasso)
    
    
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
                
                      
            risultati_lasso =  pd.DataFrame([ y, t, n, SE, SSE, 
                                             Dev_Y, MSE, Var_Y, 
                                             Root_MSE, RSE, RRSE, 
                                             RAE, MAE ] + [v,'Lasso']).transpose()
                
                
            df_result_regressions = df_result_regressions.append( risultati_lasso )

#################################################################
name_columns = ['Y', 'Cancer type', 'Numerosità', 'SE','SSE', 
                'Deviance', 'MSE', 'Variance', 'Root_MSE', 'RSE',
                'RRSE', 'RAE', 'MAE', 'num_var', 'Model' ]

df_result_regressions.columns = name_columns
df = df_result_regressions.copy()

    
# np.corrcoef( df_regression['n_var'], df_regression['RSE'])

writer = pd.ExcelWriter('results/train_blood/test_set_diviso_per_ct/Regressions_no_PCA.xlsx')
df_result_regressions.to_excel(writer)
writer.save()


########################################################################
########################################################################
########################################################################



########################################################################
########################################################################
##################### MLP E SVM ########################################
########################################################################

var_set = [ 2,  10, 20, 50, 70, 100, 300]
# Y_array = ['BMS-708163_IC_50', 'Z-LLNle-CHO_IC_50']

df_svm = pd.DataFrame()
    
for y in Y_array:

    for v in  var_set:
        
        print( y, v, '\n\n\n')
        

        variables = cor_df.nlargest( n = v, columns = y ).index 
        train_dataset = create_dataset ( train_data, y, variables) 
#        n = len(train_dataset)
        
        
################################ SVM ############################
        print('Support Vector Machine \n')
        
        ## ****************** Scelta parametri *****************##
        
        parameters = {'kernel':('poly', 'rbf'),
                      'C':[1,3,9,13,17], 
                      'gamma': [0.01,0.04,0.1,0.4,0.6]}


        svr = svm.SVR()
        grid = GridSearchCV(svr, parameters, n_jobs = 3)
        X_train = train_dataset[ variables ]
        y_train = train_dataset[ y ]
            
        print( "Scelta dei parametri")
        start_time = time.time()
        SVM = grid.fit( X_train, y_train )
        print("--- %s seconds ---" % (time.time() - start_time))
        print( SVM.best_params_ )
 
            ## ************************************************** ##
        
        for t in tipo: 
            
            print('SVM \n', t, '\n')

            test_data = complete_dataframe
            test_data = test_data[ test_data['Cancer Type'] == t]
            test_dataset = create_dataset ( test_data, y, variables )
    
            prediction = model_estimation(data = train_dataset, 
                                          target_variable = y,
                                          explanatory_variable = variables,
                                          test_data = test_dataset,
                                          model = SVM)
    
    
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
                
                      
            risultati_SVM =  pd.DataFrame([ y, t, n, SE, SSE, 
                                             Dev_Y, MSE, Var_Y, 
                                             Root_MSE, RSE, RRSE, 
                                             RAE, MAE ] + [v,'SVM']).transpose()
                
                
            df_svm = df_svm.append( risultati_SVM )

       


#################################################################
#
#df_mlp = pd.DataFrame(result_mlp_list)
#df_svm = pd.DataFrame(result_svm_list)
#df_ML = df_mlp.append( df_svm )

df_svm.columns = name_columns

    

writer = pd.ExcelWriter('results/train_blood/test_set_diviso_per_ct/SVM_no_PCA.xlsx')
df_svm.to_excel(writer)
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
    
    

















