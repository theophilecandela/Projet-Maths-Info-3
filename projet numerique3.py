import numpy as np
import math
import matplotlib.pyplot as plt
import random as r

#Discrétisation
A=0
B=500
N=101 #Nombre de points de discrétisation
Delta = (B-A)/(N-1)
discretization_indexes = np.arange(N)
discretization = discretization_indexes*Delta
#Paramètres du modèle

mu=-5
a = 50
sigma2 = 12

#Données

observation_indexes = [0,20,40,60,80,100]
depth = np.array([0,-4,-12.8,-1,-6.5,0])

#Indices des composantes correspondant aux observations et aux componsantes non observées
unknown_indexes=list(set(discretization_indexes)-set(observation_indexes))

##1 Covariance
def cov(h, a = 50, sigma_carre = 12 ):
    return (sigma_carre * np.exp(np.abs(h) / a))
    
##2 Matrice de Distance
Distance = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        Distance[i][j] = np.abs( Delta * (i -j))
        
##3  Matrice de Covariance de Z
Covariance = cov(Distance)

##4 Matrice extraites:
n = len(observation_indexes)
p = len(unknown_indexes)

cov_obs = np.zeros((n,n))
for i, k in enumerate(observation_indexes): #C_Z
    for j, l in enumerate(observation_indexes):
        cov_obs[i][j] = Covariance[k][l]
        
cov_YZ = np.zeros((p,n))
for i,k in enumerate(unknown_indexes):
    for j,l in enumerate(observation_indexes):
        cov_YZ[i][j] = Covariance[k][l]

# cov_ZY = np.zeros((n,p))
# for i,k in enumerate(observation_indexes):
#     for j,l in enumerate(unknown_indexes):
#         cov_ZY[i][j] = Covariance[k][l]
        
        
cov_unknown = np.zeros((p,p))   #= C_Y
for i, k in enumerate(unknown_indexes):
    for j, l in enumerate(unknown_indexes):
        cov_unknown[i][j] = Covariance[k][l]
        
###5 Espérance conditionnelle
obs= np.array([depth])
E_YZ = mu * np.ones((p, 1)) + np.dot(np.dot(cov_YZ, np.linalg.inv(cov_obs)), (obs.T - mu * np.ones((n, 1))))
# plt.plot(unknown_indexes, E_YZ.T[0], label = 'Espérance conditionelle de la profondeur')
# plt.scatter(observation_indexes, depth, color = 'red', label = "Points d'observation")
# plt.legend()
#plt.show()

###6 Variance conditionnelle
#La variance conditionnelle est donnée par var_YZ = CS_Y = C_Y − C_{Y,Z}C_{Z}^{-1}C_{Z,Y}
var_YZ = cov_unknown - np.dot(cov_YZ, np.dot(np.linalg.inv(cov_obs), cov_YZ.T))
var_YZ = -var_YZ

# plt.plot(unknown_indexes, [var_YZ[i][i] for i in range(p)], label = 'Espérance conditionelle de la profondeur')
# plt.show()

###7 Simulation:
def loi_normale():
    u = 0
    v = 0
    while u == 0 or v == 0:
        u = r.random()
        v = r.random()
    return np.sqrt(-2*np.log(u)) * np.cos(2*np.pi * v)


def Cholesky(A):
    #A symetrique definie positive
    L = np.zeros(A.shape)
    L[0][0] = np.sqrt(A[0][0])
    for i in range(1, len(L)):
        j = 0
        s = 0
        while j < i:
            sum = 0
            for k in range(j):
                sum += L[i][k] * L[j][k]   
            L[i][j] = (A[j][i] - sum) / L[j][j]
            s += (L[i][j]) ** 2
            j += 1
        L[i][i] = np.sqrt(A[i][i] - s)
    return L
    
L = Cholesky(var_YZ)

def simulation():
    Gaussien = np.array([[loi_normale() for i in range(p)]])
    Gaussien = Gaussien.T
    
    Y = E_YZ + np.dot(L, Gaussien)
    return Y

Y = simulation()
# plt.plot(unknown_indexes, E_YZ.T[0], label = 'Espérance conditionelle de la profondeur')
# plt.plot(unknown_indexes, Y.T[0], label = 'Simulation conditionnelle')
# plt.scatter(observation_indexes, depth, color = 'red', label = "Points d'observation")
# plt.legend()
# plt.show()

###8 Longueur de cable
def longueur(Z):
    l = 0
    for i in range(1, len(Z)):
        l += np.sqrt(Delta ** 2 + (Z[i] - Z[i-1]) **2 )
    return l
    
#9
def liste_profondeur(Y, prof, i_obs):
    #prend un vecteur imulé en parametre, et depth et observations_indexes
    #renvoie la liste de toutes les profondeurs, dont celles des sites d'observation
    y = list(Y.T[0])
    z = list(prof)[::-1]
    for i in i_obs:
        a = z.pop()
        y = y[:i] + [a] + y[i::]
    return y

def moyenne_longueur(nb = 100):
    s = 0
    lo = []
    for i in range(nb):
        Y = simulation()
        Z = liste_profondeur(Y, depth, observation_indexes)
        l = longueur(Z)
        lo.append(l)
        s += l
    #     plt.plot(discretization, Z)
    # plt.plot(discretization, liste_profondeur(E_YZ, depth, observation_indexes), color = 'red')
    # plt.plot(discretization, np.array(liste_profondeur(np.array([[np.sqrt(var_YZ[i][i]) for i in range(p)]]).reshape((p,1)), [0 for i in depth], observation_indexes)) + np.array(liste_profondeur(E_YZ, depth, observation_indexes)), color = 'red')
    # plt.plot(discretization, - np.array(liste_profondeur(np.array([[var_YZ[i][i] for i in range(p)]]).reshape((p,1)), [0 for i in depth], observation_indexes)) + np.array(liste_profondeur(E_YZ, depth, observation_indexes)), color = 'red')
    # plt.show()
    #     
    s = s / nb
    return s, lo
    
    
longueur_esperance = longueur(liste_profondeur(E_YZ, depth, observation_indexes))
        

##10
s, l = moyenne_longueur(100)
Mn = []
for i in range(1, len(l) + 1):
    Mn.append(sum(l[:i])/(i))
    
# plt.plot([i for i in range(1, len(Mn)+1)], Mn)
# plt.show()

n_bins = 20

# plt.hist(l, bins=n_bins)
# plt.show()

##
# s, l = moyenne_longueur(10000)
# len([x for x in l if x>525])
d = sorted(l)
conserver = int(0.95 * len(l))
ecarts = [(i, abs(longueur - s)) for i, longueur in enumerate(d)]
ecarts = sorted(ecarts, key = lambda x: x[1])

