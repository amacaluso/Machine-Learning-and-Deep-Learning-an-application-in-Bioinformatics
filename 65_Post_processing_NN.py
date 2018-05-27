# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 22:44:43 2017

@author: Antonio
"""


exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descrittive.py').read(), globals())
#exec(open("05_Test_d_ipotesi.py").read(), globals())


from sklearn.grid_search import GridSearchCV
# from sklearn import ae, mlp
import sklearn.neural_network as nn

n_components = []
explained_variance = []


print('Analisi delle componenti principali ... ')

for i in range(10, 500, 10):
    n_components.append(i)
    pca = PCA(n_components=i)
    pca.fit(X)
    explained_variance.append(sum(pca.explained_variance_ratio_))
    
name_columns = ['Y', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                 'MAE', 'RAE','Deviance', 'Variance', 'n_componenti' ]

#vlines(10,0,1,color='k',linestyles='solid')
comp = [10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450]

######################### MLP  ######################################

result_nn = []
result_mlp_list = []

for c in comp:
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
    
    
    
#    model = skl.linear_model.LinearRegression()
    
    
    for i in range( 0, len(Y_array) ):
        print(i)
    #    i=0
        print(Y_array[i])
        n = len( list_data[i])
        
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
        
        result_nn = cross_validation( splits = n, 
                                      target_variable = Y_array[i],
                                      explanatory_variable = list_data[i][X_matrix].columns,
                                      data = list_data[i],
                                      model = NeurNet)
        
        risultati =  [Y_array[i], n] + result_nn + [c]
        result_mlp_list.append( risultati )
    
#    result_svm_list.append( result_svm )
    

df_nn = pd.DataFrame(result_mlp_list)
df_nn.columns = name_columns
df_nn


df = df_nn



import matplotlib.pyplot as plt

xb1 = df.loc[df['Y'] == 'BMS-708163_AUC', 'n_componenti']
yb1 = df.loc[df['Y'] == 'BMS-708163_AUC', 'RSE']
xz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'n_componenti']
yz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'RSE']
    
xb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'n_componenti']
yb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'RSE']
xz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'n_componenti']
yz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'RSE']
    
    
plt.plot(xb1 , yb1, 'ro-' , label = 'BMS-708163_AUC')
plt.plot(xz1 , yz1, 'bs-', color ='b', label = 'BMS-708163_IC_50')
plt.plot(xb2 , yb2, 'ro-' , color ='y', label = 'Z-LLNle-CHO_AUC')
plt.plot(xz2 , yz2, 'bs-', color ='g', label = 'Z-LLNle-CHO_IC_50')
hlines(1, 0, max(comp), linestyles= 'dashed')# [‘solid’ | | ‘dashdot’ | ‘dotted’], optional)
plt.title('Reti Neurali MLP')
plt.ylabel('RSE')
plt.xlabel('Componenti')
plt.legend()
savefig('Presentazione/'+ 'MLP_all' + '.png')
plt.close()

############################################################################
############################################################################


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
    
    df_nn = pd.DataFrame()#columns = name_columns)
    result_nn_list = []
    
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
        
        
    
        for y in Y_array:
    
            data = create_dataset(data = dataset, 
                                  target_variable = y, 
                                  explanatory_variable = X_matrix)
           
            n = len( data)
            print(y, t, n)
            
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
                
            X_train = data[ X_matrix]
            y_train = data[ y ]
            
            print( "Scelta dei parametri Reti Neurali MLP \n")
            
            start_time = time.time()
            
            NeurNet = grid.fit( X_train, y_train)
            
            print("--- %s seconds ---" % (time.time() - start_time),"\n\n")
        
            print( NeurNet.best_params_ ,"\n\n")
            print("Stima del modello \n")
            
            
            
        
            result_nn = cross_validation(splits = n, 
                                          target_variable = y,
                                          explanatory_variable = X_matrix,
                                          data = data,
                                          model = NeurNet)
            
            risultati =  [y, n] + result_nn +[c]
            result_nn_list.append( risultati )
 
            
            
        
    df_nn = pd.DataFrame(result_nn_list)
         
        
        
    df_nn.columns = name_columns
    df = df_nn
        
        
    xb1 = df.loc[df['Y'] == 'BMS-708163_AUC', 'n_componenti']
    yb1 = df.loc[df['Y'] == 'BMS-708163_AUC', 'RRSE']
    xz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'n_componenti']
    yz1 = df.loc[df['Y'] == 'Z-LLNle-CHO_AUC', 'RRSE']
        
    xb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'n_componenti']
    yb2 = df.loc[df['Y'] == 'BMS-708163_IC_50', 'RRSE']
    xz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'n_componenti']
    yz2 = df.loc[df['Y'] == 'Z-LLNle-CHO_IC_50', 'RRSE']
    
    import matplotlib.pyplot as plt

    
    plt.plot(xb1 , yb1, 'ro-' , label = 'BMS-708163_AUC')
    plt.plot(xz1 , yz1, 'bs-', color ='b', label = 'BMS-708163_IC_50')
    plt.plot(xb2 , yb2, 'ro-' , color ='y', label = 'Z-LLNle-CHO_AUC')
    plt.plot(xz2 , yz2, 'bs-', color ='g', label = 'Z-LLNle-CHO_IC_50')
    hlines(1, 0, max(comp), linestyles= 'dashed')# [‘solid’ | | ‘dashdot’ | ‘dotted’], optional)
    plt.title(t + ' MLP')
    plt.ylabel('RRSE')
    plt.xlabel('Componenti')
    plt.legend()
    savefig('results/Immagini/MLP/'+ t + '.png')
    plt.close()


