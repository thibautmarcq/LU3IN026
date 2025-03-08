# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 

def crossval(X, Y, n_iterations, iteration):
    start_idx = int(iteration * (X.shape[0] // n_iterations))
    end_idx = int((iteration + 1) * (X.shape[0] // n_iterations))

    Xtest = X[start_idx:end_idx]
    Ytest = Y[start_idx:end_idx]
    
    Xapp = np.concatenate((X[:start_idx],X[end_idx:]))
    Yapp = np.concatenate((Y[:start_idx],Y[end_idx:]))

    return Xapp, Yapp, Xtest, Ytest

def crossval_strat(X, Y, n_iterations, iteration):
    classes = np.unique(Y) 
    indices_par_classe = {c: np.where(Y==c)[0] for c in classes}

    
    test_indices= []
    train_indices =[]

    for c in classes:
        indices = indices_par_classe[c]  # Indices de la classe actuelle
        taille_test = len(indices) // n_iterations 
        
        
        start =iteration*taille_test
        end=(iteration+1)*taille_test
        test_indices.extend(indices[start:end]) 
        train_indices.extend(np.concatenate((indices[:start], indices[end:]))) 

    
    Xtest,Ytest=X[test_indices],Y[test_indices]
    Xapp,Yapp=X[train_indices],Y[train_indices]

    return Xapp,Yapp,Xtest,Ytest


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return (np.mean(L), np.std(L))


def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    perf = []
    for i in range(nb_iter):
        classif = copy.deepcopy(C)

        Xapp,Yapp,Xtest,Ytest = crossval_strat(DS[0], DS[1], nb_iter, i)
        classif.train(Xapp,Yapp)
        accuracy = classif.accuracy(Xtest,Ytest)
        perf.append(accuracy)

        print(f"Itération {i} : taille base apprentissage= {len(Xapp)} taille base de test= {len(Xtest)} taux de bonne classif: {accuracy:0.4f}")
    
    taux_moyen, taux_ecart = analyse_perfs(perf)
    return (perf, taux_moyen, taux_ecart)