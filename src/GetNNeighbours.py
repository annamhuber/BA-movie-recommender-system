from src.DataPreprocessing import DataPreprocessing
from src.Similarities import Similarities
from src.PrincipalComponentAnalysis import PrincipalComponentAnalysis
import numpy as np
import tensorflow as tf
import pandas as pd
import MySQLdb
from sqlalchemy import create_engine
import time

class GetNNeighbours(object):
    def __init__(self):
        self.df_ratings = pd.read_csv('/data/ml-25m/ratings.csv', sep=",",usecols=[0, 1, 2], memory_map=True,low_memory=True, nrows=1000000)
        self.datapreprocessing = DataPreprocessing()
        self.similarities = Similarities()
        self.principalcomponantanalysis = PrincipalComponentAnalysis()

    def _find_n_neighbours(self, df: pd.DataFrame, n) -> pd.DataFrame:
        df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                          .iloc[:n].index,
                                          index=['top{}'.format(i) for i in range(1, n + 1)]), axis=1)
        return df


    def get_n_neighbours(self, n: int):
        #data = self.df_ratings.sample(n=1000000,random_state=1)
        data = self.df_ratings
        # user= self.datapreprocessing.create_user_rating_vec(data)
        # similar_users = self.similarities.get_cosine_similarity(user)
        # nearest_neighbours_user = self._find_n_neighbours(similar_users, n)
        #
        # items = self.datapreprocessing.create_item_rating_vec(data)
        # similar_items = self.similarities.get_cosine_similarity(items)
        # nearest_neighbours_items = self._find_n_neighbours(similar_items, n)

        items = self.datapreprocessing.create_item_rating_vec(data)
        items_pca, covarianz_matrix = self.principalcomponantanalysis.tf_pca(items)
        similar_items_pca = self.similarities.get_mahalanobis_distance(items_pca, covarianz_matrix)
        nearest_neighbours_items_pca = self._find_n_neighbours(similar_items_pca,n)

        # user = self.datapreprocessing.create_item_rating_vec(data)
        # user_pca, covarianz_matrix = self.principalcomponantanalysis.tf_pca(user)
        # similar_users_pca = self.similarities.get_mahalanobis_distance(user_pca, covarianz_matrix)
        # nearest_neighbours_users_pca = self._find_n_neighbours(similar_users_pca, n)

        return nearest_neighbours_items_pca #, nearest_neighbours_user, nearest_neighbours_items,nearest_neighbours_users_pca

    def df_to_sql(self,df: pd.DataFrame):
        engine = create_engine('mysql://root:banana@localhost/NACHBARN')  # enter your password and database names here
        df.to_sql('itemnachbarn_PCA_1M', con=engine, index=True, index_label='movieId', if_exists='replace')
        # df.to_sql('itemnachbarn_PCA_2M', con=engine, index=True, index_label='movieId', if_exists='replace')
        # df.to_sql('itemnachbarn_PCA_5M', con=engine, index=True, index_label='movieId', if_exists='replace')
        # df.to_sql('usernachbarn_PCA_1M', con=engine, index=True, index_label='userId', if_exists='replace')
        # df.to_sql('usernachbarn_PCA_2M', con=engine, index=True, index_label='userId', if_exists='replace')
        # df.to_sql('usernachbarn_PCA_5M', con=engine, index=True, index_label='userId', if_exists='replace')
        # df.to_sql('itemnachbarn_FS_1M', con=engine, index=True, index_label='movieId', if_exists='replace')
        # df.to_sql('itemnachbarn_FS_2M', con=engine, index=True, index_label='movieId', if_exists='replace')
        # df.to_sql('itemnachbarn_FS_5M', con=engine, index=True, index_label='movieId', if_exists='replace')
        # df.to_sql('itemnachbarn_FS_7M', con=engine, index=True, index_label='movieId', if_exists='replace')
        # df.to_sql('usernachbarn_FS_1M', con=engine, index=True, index_label='userId', if_exists='replace')
        # df.to_sql('usernachbarn_FS_2M', con=engine, index=True, index_label='userId', if_exists='replace')
        # df.to_sql('usernachbarn_FS_5M', con=engine, index=True, index_label='userId', if_exists='replace')
        # df.to_sql('usernachbarn_FS_7M', con=engine, index=True, index_label='userId', if_exists='replace')




if __name__ == '__main__':
    start_time = time.time()
    Result = GetNNeighbours()
    nachbaren = Result.get_n_neighbours(100)
    nachbaren = nachbaren + 1
    nachbaren.index = nachbaren.index +1
    print(nachbaren.head(20))
    Result.df_to_sql(nachbaren)
    print('{0:.2f}'.format(time.time() - start_time) + ' seconds')
