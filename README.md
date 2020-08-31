# BDA Movie Recommender System

Summary
This bachelor project is in the domain of Machine Learning and Recommender Systems. Generally, user-user and item-item neighborhoods are computed in the feature space (using e.g. ratings data). The user/items that are neighbors of each other are then used in the computation of recommendations. Pearson Correlation or Cosine Similarity is used in the feature space to compute the distances. Afterwards, the Top-N (usu. 30-50) nearest/closest neighbors are identified. The idea of this project is to compute the user and item distances in the Principal Component Analysis (PCA) space. Using the distances in the PCA space users/items neighbors are then identified in the PCA space using e.g. Mahalanobis Distance.