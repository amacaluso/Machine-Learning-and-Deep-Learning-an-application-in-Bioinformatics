
exec(open("Utils.py").read(), globals())
exec(open("01_Importazione_dati_e_moduli.py").read(), globals())
exec(open('03_Descriptive.py').read(), globals())


##############################################
######### Test ipotesi v. risposta ###########
##############################################

# dati_risposta = pd.read_table('ccle_drug_response.txt', decimal = '.')

BMS_IC_50 = dati_risposta.ix[:, [2,4]].dropna()
# n1 = len(BMS_IC_50)

BMS_AUC = dati_risposta.ix[:, [2,5]].dropna()
# n2 = len(BMS_AUC)


Z_IC_50 = dati_risposta.ix[:, [2,6]].dropna()
# n3 = len(Z_IC_50)

Z_AUC = dati_risposta.ix[:, [2,7]].dropna()
# n4 = len(Z_AUC)

lista_dataset = [BMS_IC_50, BMS_AUC, Z_IC_50, Z_AUC]
# lista_num = [n1, n2, n3, n4]


tipo = list(set(dati_risposta.ix[:,2]))

#
lista_group = []
for data in lista_dataset: 
    
    lista  = []

    for i in tipo:
        #print(i)
        lista_corrente = []
        
        for j in data.index:
            #print(j)
            if data.ix[j,0] == i:
                lista_corrente.append(data.ix[j,1])
    #            
        lista.append(lista_corrente)
    
    lista_group.append(lista)

dataframe_null_type.to_csv("results/missing.csv", sep=';') 
    


for lista in lista_group: # Correggere
    f_val, p_val = stats.f_oneway(lista[0], lista[1], lista[2], lista[3], 
                                  lista[4], lista[5], lista[6], lista[7],
                                  lista[8], lista[9], lista[10], lista[11],
                                  lista[12])  
    print( f_val, p_val )
    
    
for lista in lista_group:  # Correggere
    f_val, p_val = stats.mstats.kruskalwallis(lista[0], lista[1], lista[2],
                                              lista[3], lista[4], lista[5], 
                                              lista[6], lista[7], lista[8],
                                              lista[9], lista[10], lista[11],
                                              lista[12])  
    print( f_val, p_val )



##############################################################################
########################## CONFRONTI MULTIPLI ################################
##############################################################################


farmaci = [ "BMS_IC_50", "BMS_AUC", "Z_IC_50", "Z_AUC" ]

alpha_reale = 0.05/13

for k in range(0, len(lista_group)):
    
    for i in range(0, len(lista_group[k])):
        
        for j in range(i+1, len(lista_group[k])):
            t_val, p_val = scipy.stats.ttest_ind(
                    lista_group[k][i], lista_group[k][j])
            
            if(p_val < alpha_reale):
                print("Variabile risposta: ", farmaci[k]," --> ", tipo[i],
                  "vs", tipo[j], ":  p-value = ", p_val )
            

