# AI-Journey / Parcours en Intelligence Artificielle

Key/ Cle: 

- 🛑 : Not completed / Non complété
- ⚠️ : To be revised / À réviser
- ✅ : Completed / Complété
 

## Theory / Théorie :

### Matric Calculus / Calcul Matriciel : 
- Changement de base / Change of basis ✅
- Diagonalisation / Diagonalization ✅ 
- Singular Value Decomposition / Décomposition en valeurs singulières ✅  
- Kronecker Product and matrix derivatives / Produit de Kronecker et dérivées matricielles ✅
- [Lagrange Duality / Dualite Lgrangian](https://www.youtube.com/watch?v=thuYiebq1cE&t=1793s) ✅


### Probability and Satistics / Probabilité et Statistiques : 
- Lois discretes / Discrete distributions ✅
- Lois continues (Normale, Student, Khi-deux) / Continuous distributions (Normal, Student's t, Chi-square) ✅
- Lois multivariables / Multivariable distributions ✅



### Machine Learning :
* #### [CS229 Stanford course videos / Vidéos de cours de Stanford](https://www.youtube.com/watch?v=het9HFqo1TQ&t=2143s) (Lectures completed 13/20) 🛑
    * **Lecture 1 : Introduction and Examples** ✅
    * **Lecture 2 : Linear Regression and Gradient Descent** ✅
        - Cost Function (Least average squares) / Fonction de Coût (Méthode des moindres carrés)
        - Normal Equations / Équations Normales
    * **Lecture 3 : Locally Weighted & Logistic Regression / Régression Locale Pondérée & Logistique** ✅
        - Weight function / Fonction de Poids
        - Sigmoid function and binary classification / Fonction Sigmoïde et Classification Binaire
        - Probabilistic Interpretation / Interprétation probabiliste
        - Likelihood of parameters / Vraisemblance des Paramètres
    * **Lecture 4 : Perceptron & Generalized Linear Model** ✅
        - Perceptron
        - exponential family / Famille Exponentielle
        - Generalised Linear Models (GLMs) 
        - Softmax Regression and Multi-class Classification 
    * **Lecture 5 : Gaussian Discriminant Analysis & Naive Bayes** ✅
        - Generative Learning Algorithms / Algorithmes d'Apprentissage Génératif
        - Gaussian Dicriminant Analysis / Analyse Discriminante Gaussienne
        - Multivariate Gaussian Distribution / Distribution Gaussienne Multivariée
        - Singular Value Decomposition (SVD) / Décomposition en Valeurs Singulières (SVD)
        - Introduction to Naive Bayes / Introduction au Naïf Bayes
    * **Lecture 6 : Support vector Machines** ✅
        - Naives Bayes & Spam classifier : Multivariate and Multinomial models / Naïf Bayes et Trieur de Spam : Modèles Multivariés et Multinomiaux
        - Laplace smoothing 
        - PROs and CONs of Generative & Discriminative Learning Algorithms / Avantages et Inconvénients des Algorithmes d'Apprentissage Génératif et Discriminant
        - Introduction to Support Vector Machines (SVMs) / Introduction aux Machines à Vecteurs de Support (SVMs)
        - Functional and Geometric margin / Marge Fonctionnelle et Géométrique
    * **Lecture 7 : Kernels** ✅
        - Feature & Dimension augmentation / Augmentation de Caractéristiques et de Dimensions
        - Kernel trick and computation reduction / Noyaux et Optimisation
        - Lagrange Duality / Dualité de Lagrange
    * **Lecture 8 :  Data Splits, Models & Cross-Validation** ✅
        - Bias & Variance trade-off / Compromis Biais et Variance
        - Regularisation / Régularisation
        - Train/Devloppment/Test Split / Séparation Entraînement/Développement/Test
        - K-fold Cross Validation / Validation Croisée en K-Volets
    * **Lecture 9 : Approx/Estimation Error & ERM / Erreur d'Approximation/Estimation & Minimisation du Risque Empirique** ⚠️
        - Formal definition of Bias and Variance / Définition formelle du Biais et de la Variance
        - Regularisation and Variance / Régularisation et Variance
        - Consistency and Efficiency of a model / Cohérence et Efficacité d'un modèle
        - Approximation/Empirical/Estimation/Irreductible/Generalisation Error / Erreur d'Approximation/Empirique/Estimation/Irréductible/Généralisation
        - Empirical Risk Minimisation / Minimisation du Risque Empirique
        - Uniform Convergence (Hoeffding's Inequality) / Convergence Uniforme (Inégalité de Hoeffding)
        - VC Dimension / Dimension VC
    * **Lecture 10 : Decision Trees and Ensemble Methods / Arbres de Décision et Méthodes d'Ensemble** ✅
        - Miss-classification and Cross-entropy loss / Erreur de Classification et Perte d'Entropie Croisée
        - Information gain / Gain d'Information
        - Regression trees / Arbres de Régression
        - Pros (Fast, simple, Low Bias) and Cons (Bad additivity, High Variance) / Avantages (Rapide, Simple, Faible Biais) et Inconvénients (Mauvaise Additivité, Haute Variance)
        - Regularisation techniques for Decision trees / Techniques de Régularisation pour les Arbres de Décision
        - Runtime complexity / Complexité 
        - Ensembling Techniques(Bagging/Bootstrap sampling, Boosting, Random Forests, Stacking) / Techniques d'Ensemble (Bagging/Échantillonnage Bootstrap, Boosting, Forêts Aléatoires, Stacking)
    * **Lecture 11 : Introduction to Neural Networks / Introduction aux Réseaux de Neurones** ✅
        - Intro : Logistic Regression to Neural Networks / Introduction : De la Régression Logistique aux Réseaux Neuronaux
        - Activation functions / Fonctions d'Activation
        - Architecture and parameters / Architecture et Paramètres
        - Loss and cost function / Fonction de Perte et Coût
        - Forward and Backward propogation equations / Équations de la Propagation Avant et Arrière 
    * **Lecture 12 : Backprop & Improving Neural Networks Rétropropagation & Amélioration des Réseaux Neurones** ✅
        - Concrete example of back propogation / Exemple Concret de la Rétropropagation
        - Improving Neural networks / Améliorer les Réseaux Neurones :
            * advantages and disadvantages of different (ReLU, sigmoid, Tanh) / Avantages et Inconvénients des différentes Fonctions d'Activation (ReLU, Sigmoïde, Tanh)
            * Vanishing and Exploding  gradients / Gradients Disparus et Explosifs
            * Symmetry Problem / Problème de Symétrie
            * Initialization Schemes / methodes d'Initialisation 
                1. Xavier Initialization 
                2. He Initialization
                3. np.random.rand(shape)*np.sqrt(1/n[L-1])
            * Normalization techniques / Techniques de Normalisation
            * Optimization / Optimisation
                1. Mini-Bathch Gradient Descent
                2. Momentum Algorithm    
    * **Lecture 13 : Debugging ML Models and Error Analysis / Débogage des Modèles ML et Analyse des Erreurs** ✅
        - Diagnostics for debugging learning algorithms / Diagnostics pour Déboguer les Algorithmes d'Apprentissage  
            * Convergence of the optimization algoritm / Convergence de l'Algorithme d'Optimisation
            * Rightness of the objective function / Validité de la Fonction Objectif
        - Error analysis and ablative analysis / Analyse des Erreurs et Analyse Ablative
        - Premature statistical optimization  
  





     

* #### [stanford course notes / Notes de cours de Stanford](https://cs229.stanford.edu/syllabus-autumn2018.html) 🛑

## Practical aspect and exercises / Aspect pratique et exercices :

### CS299-Stanford-Problem-Sets 🛑: 

* #### [PS0-2018](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS0-2018) [ [question / sujet](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS0-2018/ps0.pdf) ; [solution](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS0-2018/ps0_Solution.pdf) ]: ✅
    - Linear Algebra & Matrix Calculus

* #### [PS1-2018](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS1-2018) [ [question/sujet](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS1-2018/ps1.pdf) ; [solution](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS1-2018/ps01_Solution.pdf) ]: ✅
    - Linear Classifiers (logistic regression and GDA)
    - Incomplete, Positive-Only Labels
    - Poisson Regression
    - Convexity of Generalized Linear Models
    - Locally Weighted Linear Regression
    
* #### [PS2-2018](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS2): 🛑
* #### [PS3-2018](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS3): 🛑
* #### [PS4-2018](./Problem-Sets%20Maths%20and%20Code/CS229-Machine-Learning-Stanford/PS4): 🛑


### Coursera Courses / Cours Coursera :
* ####  Machine Learning Specialization / Spécialisation en Apprentissage Automatique ✅ [View Certificate](./Certificates/Coursera%20E4LSNKK33ML8.pdf) : 
    * **[Supervised Machine Learning: Regression and Classification (Beginners Level) / Apprentissage automatique supervisé : Régression et Classification (Niveau débutant)](https://www.coursera.org/specializations/machine-learning-introduction) : 03-05-2024 - 05-07-2024** ✅  
        - Self implementation of cost & gradient descent functions for Linear regrssion & logistic classification / Implémentation des fonctions de coût et de descente
            de gradient pour la régression linéaire et la classification logistique 
        - Feature Selection / Sélection de caractéristiques
        - Feature Scaling and Normalisation / Mise à l'échelle et Normalisation
        - Regularisation / Régularisation

    * **[Supervised Machine Learning: Advanced Learning Algorithms (Intermediate Level) / Apprentissage Supervisé : Algorithmes d'Apprentissage Avancés (Niveau Intermédiaire)](https://www.coursera.org/learn/advanced-learning-algorithms) : 06-07-2024 08-08-2024**✅   
        - Introduction to Neural Networks / Introduction aux Réseaux de Neurones
        - Tensor Flow
        - Softmax & Multiclass Classification / Softmax et Classification Multiclasse
        - Activation Functions (linear, sigmoid, relu, softmax, softmax) / Fonctions d'Activation (linéaire, sigmoïde, relu, softmax)
        - NN Layers (Dense and Convolutional) / Couches RN (Dense et Convolutionnelle) 
        - lossfunction (MeanErrorSquared, BinaryCrossEntropy,SparseCategoricalCrossEntropy, from_logits) / Fonctions de Perte (Erreur Quadratique Moyenne, Entropie
            Croisée Binaire, Entropie Croisée Catégorique Éparse, from_logits)
        - Implementation of Neural Networks with TF / Implémentation de Réseaux de Neurones avec TF
        - Self implementation of Neural Networks with python (without using pre-built libraries) / Implémentation autonome de réseaux de neurones avec Python (sans
            l'utilisation des bibliothèques pré-construites)
        - Optimizers (Adam) / Optimiseurs (Adam)
        - Forward and Backward Propogation / Propagation Avant et Arrière
        - Decision trees and Tree ensembles / Arbres de Décision et Ensembles d'Arbres
        - Entropy and Information Gain / Entropie et Gain d'Information
        - F1-Score
    
    * **[Unsupervised Learning, Recommenders, Reinforcement Learning (Intermediate Level) / Apprentissage Non Supervisé, Systèmes de Recommandation, Apprentissage par Renforcement (Niveau Intermédiaire)](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning) : 09-08-2024 08-09-2024**✅
        - Introduction to unsupervised learning / Introduction à l'Apprentissage Non Supervisé
        - Clustering & K-means algorithm 
        - Anomaly detection and Gaussian distribution / Détection d'Anomalies et Distribution Gaussienne
        - Recommendation systems / Systèmes de Recommandation
        - Collaborative and Content based filtering / Filtrage Collaboratif et Basé sur le Contenu
        - Reninforcement Learning & Reward function / Apprentissage par renforcement et fonction de récompense

* ####  IBM AI Engineering Professional Certificate / Certificat Professionnel en Ingénierie de l'IA d'IBM : 
    * **[Machine Learning with Python](https://www.coursera.org/learn/machine-learning-with-python) : 22-08-2024 20-09-2024**✅
        - Basic ML Algorithms (Linear & logistic regression, K-Nearest Neighbors, Regression Trees)


* ### Other Resources  
    * #### [ScikitLearn Tutorial (FreeCodeCamp)](https://www.youtube.com/watch?v=0B5eIE_1vpU&t=884s) ✅
        - Preprocessing / Prétraitement 
        - ML Pipelines / Pipelines ML
        - Cross Validation / Validation croisée

    
