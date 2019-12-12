# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:54:36 2017

@author: Antonio
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:06:56 2017

@author: Antonio
"""


exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descriptive.py').read(), globals())
#exec(open("05_hypothesis_test.py").read(), globals())

comp = [10, 20, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450]
import matplotlib.pyplot as plt

name_columns = ['Y', 'Numerosità', 'SE','SSE', 'MSE', 'Root_MSE', 'RSE','RRSE',
                 'MAE', 'RAE','Deviance', 'Variance', 'n_componenti' ]

Y_array = dati_risposta.columns[4:8]

tipo = ['skin','aero_digestive_tract', 'breast',
        'nervous_system', 'urogenital_system',
        'digestive_system', 'blood', 'lung']


######################################################################
######################################################################

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
    df1 = pd.DataFrame()
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
        
        
    
        for y in Y_array:
    
            data = create_dataset(data = dataset, 
                                  target_variable = y, 
                                  explanatory_variable = X_matrix)
           
            n = len( data)
            print(y, t, n)
    
            
            if n> 5:   
                regression = cross_validation(splits = int(n/3), 
                                              target_variable = y, 
                                              explanatory_variable = X_matrix,
                                              data = data )
                
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
    
    import matplotlib.pyplot as plt

    
    plt.plot(xb1 , yb1, 'ro-' , label = 'BMS-708163_AUC')
    plt.plot(xz1 , yz1, 'bs-', color ='b', label = 'BMS-708163_IC_50')
    plt.plot(xb2 , yb2, 'ro-' , color ='y', label = 'Z-LLNle-CHO_AUC')
    plt.plot(xz2 , yz2, 'bs-', color ='g', label = 'Z-LLNle-CHO_IC_50')
    hlines(1, 0, max(comp), linestyles= 'dashed')# [‘solid’ | | ‘dashdot’ | ‘dotted’], optional)
    plt.title(t)
    plt.ylabel('RRSE')
    plt.xlabel('Componenti')
    plt.legend()
    savefig('results/Immagini/regressione_lineare/'+ t + '.png')
    plt.close()

 
    
    
    
    
######################################################################
######################################################################

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
    
    df_svm = pd.DataFrame()#columns = name_columns)
    df1 = pd.DataFrame()
    result_svm_list = []
    
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
            
            
            parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
                          'C':[1,3,5,7,9,11,13,15,17,19], 
                          'gamma': [0.01,0.03,0.04,0.1,0.2,0.4,0.6]}
            svr = svm.SVR()
            grid = GridSearchCV(svr, parameters, n_jobs = 3)
            X_train = data[ X_matrix]
            y_train = data[ y ]
            
            
            print( "Scelta dei parametri \n")
            
            start_time = time.time()
        
            
            SVM = grid.fit( X_train, y_train )
            print("--- %s seconds ---" % (time.time() - start_time),"\n\n")
        
            print( grid.best_params_ ,"\n\n")
            print("Stima del modello \n")
            n = len( data )
        
            result_svm = cross_validation(splits = min(n,10), 
                                          target_variable = y,
                                          explanatory_variable = X_matrix,
                                          data = data,
                                          model = SVM)
            
            risultati =  [y, n] + result_svm +[c]
            result_svm_list.append( risultati )
 
            
            
        
    df_svm = pd.DataFrame(result_svm_list)
         
        
        
    df_svm.columns = name_columns
    df = df_svm
        
        
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
    plt.title(t + ' SVM')
    plt.ylabel('RRSE')
    plt.xlabel('Componenti')
    plt.legend()
    savefig('results/Immagini/SVM/'+ t + '.png')
    plt.close()
