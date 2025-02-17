# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2025

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------

class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        good = 0
        for desc, label in zip(desc_set, label_set):
            prediction = self.predict(desc)
            if prediction == label:
                good += 1

        return good / len(label_set) 
    
#-----------------------------------------------------------------------------------

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)
        self.k = k

        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        
        diff = self.desc_set - np.array(x)
        distance = np.linalg.norm(diff, axis=1) # tableau des dist entre x et les pts
        distance = np.column_stack((distance, self.label_set))
        distance = distance[distance[:, 0].argsort()] #tri par rapport aux distances à x

        labels = distance[:self.k, 1] # plus que les labels
        mostLab = labels.sum()
        return mostLab    
    
        # raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """       
        if (self.score(x)<0):
            return -1
        else :
            return 1

        # raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set
        # raise NotImplementedError("Please Implement this method")
        
#-----------------------------------------------------------------------------------

class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)
        random_vector = np.random.randn(input_dimension)
        norm = np.linalg.norm(random_vector)
        self.w = random_vector / norm
        
                
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage")
        # raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
        # raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x)<0:
            return -1
        else:
            return 1
        # raise NotImplementedError("Please Implement this method")

#-----------------------------------------------------------------------------------

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self,input_dimension)
        self.epsilon = learning_rate
        self.init = init
        # initialisation de w
        if (self.init):
            self.w = np.zeros(input_dimension)
        else:
            self.w = np.random.uniform(-0.001, 0.001, input_dimension)
            
        self.allw =[self.w.copy()] # stockage des premiers poids
    
    def get_allw(self):
        return self.allw
            
            
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        X_set = desc_set.copy()
        Y_set = label_set.copy()
        desc_lab = np.column_stack((X_set, Y_set))
        np.random.shuffle(desc_lab)
        
        for i in range(desc_lab.shape[0]): #parcours de tous les exemples du set
            # calcul du score de x par le perceptron
            predict = self.predict(desc_lab[i, 0:2])
            
            if (predict != desc_lab[i, 2]):
                neww = self.w + self.epsilon * desc_lab[i, 2] * desc_lab[i, 0:1] # w=w+e*yi*xi
                self.w = neww
                self.allw.append(neww.copy())
            
            
        # raise NotImplementedError("Please Implement this method")
    
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale /
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        diffNorms = []
        self.nbIter = 0 #nb itérations
        # nb_no_chgt = 0
        wAfter = self.w
        while (self.nbIter < nb_max): #arret si trop d'itérations inchangées (convergence) ou trop d'itérations
            self.nbIter+=1
            wPrevious = wAfter
            self.train_step(desc_set, label_set)
            wAfter = self.w
            #détection de changements ou non
            if (np.linalg.norm(np.abs(wAfter - wPrevious)) < seuil):
                break
            # else:
            #     nb_no_chgt = 0
            
            diffNorms.append(np.abs(wAfter - wPrevious))
        return diffNorms
        
        # raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # on calcule le produit scalaire entre x et w (= de quel coté de la droite x se situe)
        return np.dot(x, self.w) #si erreur faire x.T !!!
        
        # raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if (self.score(x)<0):
            return -1
        else:
            return 1
        # raise NotImplementedError("Please Implement this method")