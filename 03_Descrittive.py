# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:20:17 2017

@author: Antonio
"""

exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())



###############################################
############ Descrittive delle colonne ########
###############################################


###############################################
################## MEDIE ######################
###############################################

means = pd.DataFrame( X.apply( np.mean ) )
summary_means = means.describe(percentiles = [0,1])

means = X.apply( np.mean )
sns.kdeplot(np.array( means ), bw=0.5)
means


###############################################
############## Deviazioni standard ############
###############################################

std = pd.DataFrame(X.apply( np.std ))
summary_std = std.describe(percentiles = [0,1])

std = X.apply( np.std )
sns.kdeplot(np.array( std ), bw=0.5)

###############################################
######### Coefficienti di variazione ##########
###############################################

CV = pd.DataFrame(X.apply( stats.variation ))
summary_cv = CV.describe(percentiles = [0,1])

CV = X.apply( stats.variation )
sns.kdeplot(np.array( CV ), bw=0.5)

###############################################


###############################################
################ Analisi missing ##############
###############################################

tipo = list(set(dati_risposta.ix[:,2]))
tipo

lista_null = []

for i in tipo:   
    dati_correnti = dati_risposta[dati_risposta['Cancer Type'] == i]
    current_n_rows = dati_correnti.shape[0]
    lista_null.append([i,
                       current_n_rows,
                       dati_correnti.ix[: , 4].isnull().sum(),
                       dati_correnti.ix[: , 5].isnull().sum(),
                       dati_correnti.ix[: , 6].isnull().sum(),
                       dati_correnti.ix[: , 7].isnull().sum()])

dataframe_null_type = pd.DataFrame.from_records(lista_null)
dataframe_null_type.columns = ['Cancer_type', 'total_n_row',
                               'null_BMS_IC_50','null_BMS_AUC',
                               'null_Z_IC_50', 'null_Z_AUC']
                            
sum_missing = dataframe_null_type.apply( np.sum )
    
 
MIN = X.min().min()
MAX = X.max().max()
print(MIN, MAX)

sns.countplot(x="Cancer Type", data=dati_risposta, palette="Greens_d");
plt.subplots_adjust(bottom=0.25)
plt.xticks(rotation=60)
plt.title('\n Distribuzione variabile Cancer Type')
savefig("Presentazione/frequenze.png") #, transparent=True)
plt.show()

import matplotlib.pyplot as plt

dati_risposta.ix[:, [2, 4]].boxplot( by='Cancer Type')
plt.subplots_adjust(bottom=0.15)
plt.xticks(rotation=60)
savefig("Presentazione/boxplot1.png", dpi = 900) #, transparent=True) #), dpi = 500)
plt.show()


dati_risposta.ix[:, [2, 5]].boxplot( by='Cancer Type')
plt.subplots_adjust(bottom=0.25)
plt.xticks(rotation=60)
savefig("Presentazione/boxplot2.png") #, transparent=True) #), dpi = 500)

plt.show()

dati_risposta.ix[:, [2, 6]].boxplot( by='Cancer Type')
plt.subplots_adjust(bottom=0.25)
plt.xticks(rotation=60)
savefig("Presentazione/boxplot3.png") #, transparent=True) #), dpi = 500)

plt.show()

dati_risposta.ix[:, [2, 7]].boxplot( by='Cancer Type')
plt.subplots_adjust(bottom=0.25)
plt.xticks(rotation=60)
savefig("Presentazione/boxplot4.png") #, transparent=True) #), dpi = 500)
plt.show()