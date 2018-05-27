# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:41:51 2017

@author: Antonio
"""
# import heapq
###############################################
########## Analisi componenti principali ######
###############################################

exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
#exec(open('03_Descrittive.py').read(), globals())
#exec(open("05_Test_d_ipotesi.py").read(), globals())


n_components = []
explained_variance = []


print('Analisi delle componenti principali ... ')

for i in range(10, 500, 10):
    n_components.append(i)
    pca = PCA(n_components=i, svd_solver='auto') #, 
    pca.fit(X)
#    pca.fit(pd.DataFrame.transpose(X))
    explained_variance.append(sum(pca.explained_variance_ratio_))
#plt.plot(n_components, explained_variance, 'ro)

len(pca.explained_variance_)
#vlines(10,0,1,color='k',linestyles='solid')
prova = pca.components_.shape

comp = [20, 50, 70, 100, 150, 200, 300]

fig, ax = plt.subplots()
ax.scatter(n_components, explained_variance, color = 'r')

for i, txt in enumerate(comp):
    indice = int((comp[i]/10)-1)
    h = round(explained_variance[indice],2)
    print(h)
    temp = [h, comp[i]]
    ax.annotate(h, (comp[i], explained_variance[indice]),
                horizontalalignment='top', verticalalignment='top')

fig.savefig('Presentazione/PCA.png')#, transparent=True)
# cov_matrix = np.cov(X, rowvar = 0)
# corr_matrix = np.corrcoef(X, rowvar = 0)



#pd.DataFrame(cov_matrix).head(10)
#pd.DataFrame(corr_matrix).head(10)
#eigenval, eigenvec = scipy.linalg.eig(corr_matrix)


######################### ALL CELL-LINE ######################################

N_componenti = 40
componenti = pd.DataFrame(pca.components_).ix[:,range(N_componenti)]

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

tipo = list(set(dati_risposta.ix[:,2]))
tipo