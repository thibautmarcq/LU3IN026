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
import copy 

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
        # pred =np.array([self.predict(d) for d in desc_set])
        # bon = np.sum(pred == label_set)
        # total = len(label_set)
        # return (bon /total)*100
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()
    
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
        Classifier.__init__(self, input_dimension)
        self.epsilon = learning_rate
        self.init = init
        # initialisation de w
        if self.init:
            self.w = np.zeros(input_dimension)
        else:
            self.w = np.random.uniform(-0.001, 0.001, input_dimension)
        
        # Initialisation de allw
        self.allw = [self.w.copy()]
            
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        indices = np.arange(desc_set.shape[0])
        np.random.shuffle(indices)
        
        for i in indices: # parcours de tous les exemples du set
            # calcul du score de x par le perceptron
            predict = self.predict(desc_set[i])
            
            if predict != label_set[i]:
                self.w = self.w + self.epsilon * label_set[i] * desc_set[i] # w = w + epsilon * yi * xi
                # Mise à jour de allw après chaque mise à jour des poids
                self.allw.append(self.w.copy())
    
    def train(self, desc_set, label_set, nb_max=1000, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """

        liste_difference = []
    
        for i in range(nb_max):
            ancien = self.w.copy()  
            self.train_step(desc_set, label_set)  
            diff_norm = np.linalg.norm(self.w-ancien)  
    
            liste_difference.append(diff_norm)  

            if diff_norm < seuil:
                # print("seuil atteint après", i, "iteration")
                break  

            # seuil = 80% dans ce cas on calcule la norme
            # self.accuracy(data) < 80% on continue sinon break
        return liste_difference
    
    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # on calcule le produit scalaire entre x et w (= de quel côté de la droite x se situe)
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prédiction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        else:
            return 1
    
    def get_allw(self):
        """ Retourne la liste de tous les poids w utilisés pendant l'entraînement
        """
        return self.allw
    
#-----------------------------------------------------------------------------------
    
class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
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
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
                    
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        indices = np.arange(desc_set.shape[0])
        np.random.shuffle(indices)
        
        for i in indices: # parcours de tous les exemples du set
            # calcul du score de x par le perceptron
            score = self.score(desc_set[i])
            
            if score * label_set[i] < 1:
                self.w = self.w + self.epsilon * (label_set[i] - score) * desc_set[i] # w = w + epsilon * (yi - f(xi)) * xi
                # Mise à jour de allw après chaque mise à jour des poids
                self.allw.append(self.w.copy())
                
#-----------------------------------------------------------------------------------
             
class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """
        self.cl_bin = cl_bin
        self.classifiers = []
        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """    
        self.classes = np.unique(label_set)
        self.classifiers= [copy.deepcopy(self.cl_bin) for _ in self.classes]
            
        for i, cls in enumerate(self.classes):
            binary_labels = np.where(label_set == cls, 1, -1)
            self.classifiers[i].train(desc_set, binary_labels)
        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return [clf.score(x) for clf in self.classifiers]
        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        scores = self.score(x)
        return self.classes[np.argmax(scores)]
        # raise NotImplementedError("Vous devez implémenter cette fonction !")
        
#-----------------------------------------------------------------------------------
