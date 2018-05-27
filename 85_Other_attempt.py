# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 22:05:12 2017

@author: Antonio
"""



exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descrittive.py').read(), globals())
#exec(open("05_Test_d_ipotesi.py").read(), globals())

name_columns = ['Y', 'Cancer type', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                'MAE', 'RAE','Deviance', 'Variance', 'n_var','Modello' ]

result_regression = []

ridge = RidgeCV(alphas=(0.1, 0.5, 1.0, 10, 20, 30, 50, 100, 200))#, cv = 10)
result_ridge = []

lasso = LassoCV(eps = 0.1, n_alphas = 100)#, cv = 20 )
result_lasso = []





#######################################################################
#######################################################################

Y_array = dati_risposta.columns[4:8]
X_names = X.columns
tipo = ['skin','aero_digestive_tract', 'breast',
        'nervous_system', 'urogenital_system',
        'digestive_system', 'blood', 'lung']
data = complete_dataframe



Y_t_array = np.array([(y, t) for y in Y_array for t in tipo])
X = data.ix[:, 8:18924]
cor_df = pd.DataFrame( )

# cor_df.index = X_names
list_cor = []

for y in Y_array:
    
    for t in tipo:
        
        print(y, t)
    #    t = tipo[1]
        current_data = data[ complete_dataframe[ 'Cancer Type' ] == t ]
        current_data = current_data.dropna(subset = [y])
        correlation_vec = []
        for x in X_names:
            correlation_vec.append( np.corrcoef(current_data[y], 
                                                current_data[x])[1,0])
        list_cor.append( correlation_vec )

cor_df = pd.DataFrame( list_cor )
cor_df.columns = X_names
cor_df.index = Y_t_array
cor_df = cor_df.transpose()
cor_df = abs(cor_df)

#######################################################################
#######################################################################



var_set = [10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450]

     
for y in Y_array:
    
    for t in tipo:
        
        for v in  var_set:
            
            print( y, t, v, '\n\n\n')
            col = ( y , t )
            variables = cor_df[col].nlargest( n = v ).index
            current_data = data[ data ['Cancer Type'] == t ] 
            dataset = create_dataset ( current_data, y, variables) 
            n = len(dataset)
            
    ################## REGRESSIONE LINEARE CLASSICA #################
            print('Regressione Lineare \n')
    
            regression = cross_validation(splits = n, 
                                          target_variable = y, 
                                          explanatory_variable = variables,
                                          data = dataset )
            
            risultati_reg =  [y, t, n] + regression + [v] + ['reg_lin']
            result_regression.append( risultati_reg )
    
    #################################################################    
    
    ######################### RIDGE REGRESSION ######################
            print('Regressione Ridge \n')
             
            regression = cross_validation(splits = n, 
                                          target_variable = y, 
                                          explanatory_variable = variables,
                                          data = dataset,
                                          model = ridge)
            
            risultati_ridge =  [y, t, n] + regression + [v] + ['ridge']
            result_ridge.append( risultati_ridge )
    
    #################################################################
    
    ######################### RIDGE REGRESSION ######################
            print('Regressione Lasso \n')
    
            regression = cross_validation(splits = n, 
                                          target_variable = y, 
                                          explanatory_variable = variables,
                                          data = dataset,
                                          model = lasso)
            
            risultati_lasso =  [y, t, n] + regression + [v] + ['lasso']
            result_lasso.append( risultati_lasso )
    
    #################################################################
        
    
    
    df_regression = pd.DataFrame(result_regression)
    df_lasso = pd.DataFrame(result_lasso)
    df_ridge = pd.DataFrame(result_ridge)
    
    all_regression = df_regression.append( df_lasso).append(df_ridge)
    
    all_regression.columns = name_columns
        
    # np.corrcoef( df_regression['n_var'], df_regression['RSE'])
    
    writer = pd.ExcelWriter('results/Cancer_type/no_PCA/Regressione_lineare_ALL.xlsx')
    all_regression.to_excel(writer)
    writer.save()
    
    
    ########################################################################
    ########################################################################
    ########################################################################













