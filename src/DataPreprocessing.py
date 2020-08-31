import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial import distance


class DataPreprocessing(object):

    def create_user_rating_vec(self,df:pd.DataFrame) -> pd.DataFrame:
        print(df.head())
        print(df.dtypes)
        df.movieId = df.movieId.astype('int32')
        df.userId = df.userId.astype('int32')
        df.rating = df.rating.astype('float32')

        # limit ratings to user ratings that have rated more that 20 movies
        #df = df.groupby('userId').filter(lambda x: len(x) >= 20)
        final = pd.pivot_table(df,values='rating',index='userId',columns='movieId')

        # Replacing NaN by zero
        final = final.fillna(0)

        return final

    def create_item_rating_vec(self,df:pd.DataFrame) -> pd.DataFrame:
        print(df.head())
        print(df.dtypes)
        df.movieId = df.movieId.astype('int32')
        df.userId = df.userId.astype('int32')
        df.rating = df.rating.astype('float32')

        # limit ratings to user ratings that have rated more that 20 movies
        #df = df.groupby('movieId').filter(lambda x: len(x) >= 20)

        ratings = pd.pivot_table(df,values='rating',index='movieId',columns='userId')

        #Replacing NaN by zero
        ratings = ratings.fillna(0)

        return ratings
