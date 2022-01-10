# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:28:40 2021

@author: huguet
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster


from sklearn import metrics
from scipy.spatial.distance import cdist 


##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features

# Note 1 : 
# dans les jeux de données considérés : 2 features (dimension 2 seulement)
# t =np.array([[1,2], [3,4], [5,6], [7,8]]) 
#
# Note 2 : 
# le jeu de données contient aussi un numéro de cluster pour chaque point
# --> IGNORER CETTE INFORMATION ....
#    2d-4c-no9.arff   xclara.arff

#Jeu de donnes en format arff
#path = './artificial/'
#databrut = arff.loadarff(open(path+"banana.arff", 'r'))
#datanp = np.array([[x[0],x[1]] for x in databrut[0]])

#Jeu de données en format txt
path = './new-data/'
filename = "d32.txt"
databrut = pd.read_csv(path+filename, sep=" ", encoding = "ISO-8859-1", skipinitialspace=True)
datanp = databrut.to_numpy()


########################################################################
# Preprocessing: standardization of data
########################################################################

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(datanp)

data_scaled = scaler.transform(datanp)

import scipy.cluster.hierarchy as shc

print("-------------------------------------------")
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()

# Evaluation method
silhouette = [] 
DB= [] 
mapping1 = {} 
mapping2 = {} 
K = range(2,10) 

  
for k in K: 
    tps1 = time.time()
    model_scaled = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='complete')
    yhat = model_scaled.fit(data_scaled)
    tps2 = time.time() 
    silhouette.append(metrics.silhouette_score(data_scaled, yhat.labels_, metric ='euclidean'))
    DB.append(metrics.davies_bouldin_score(data_scaled, model_scaled.labels_))
    
plt.plot(K, silhouette, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('score') 
plt.title('The Elbow Method using silhouette_score') 
plt.show() 

plt.plot(K, DB, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('score') 
plt.title('The Elbow Method using davies_boulding') 
plt.show() 

print("temps d'execution agglomeratif' : ", tps2-tps1, " secondes")
