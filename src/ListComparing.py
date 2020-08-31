import pandas as pd
import numpy as np
import MySQLdb
from sqlalchemy import create_engine, types

class ListComparing(object):

    def read_from_sql(self)-> pd.DataFrame:
        engine = create_engine('mysql://root:banana@localhost/NACHBARN')
        #FS = pd.read_sql_table('usernachbarn_FS_5M', engine, index_col='userId')
        #FS = pd.read_sql_table('usernachbarn_FS_2M', engine, index_col='userId')
        #FS = pd.read_sql_table('usernachbarn_FS_1M', engine, index_col='userId')
        #FS = pd.read_sql_table('itemnachbarn_FS_5M', engine, index_col='movieId')
        #FS = pd.read_sql_table('itemnachbarn_FS_2M', engine, index_col='movieId')
        FS = pd.read_sql_table('itemnachbarn_FS_1M', engine, index_col='movieId')

        print(FS.head())
        print(FS.shape)
        return FS


    def read_from_sql_pca(self)-> pd.DataFrame:
        engine = create_engine('mysql://root:banana@localhost/NACHBARN')
        #user_FS = pd.read_sql_table('usernachbarn_FS_5M', engine, index_col='userId')
        #PCA = pd.read_sql_table('usernachbarn_PCA_5M', engine, index_col='userId')
        #PCA = pd.read_sql_table('usernachbarn_PCA_2M', engine, index_col='userId')
        #PCA = pd.read_sql_table('usernachbarn_pca_1M', engine, index_col='userId')
        #PCA = pd.read_sql_table('itemnachbarn_PCA_5M', engine, index_col='movieId')
        #PCA = pd.read_sql_table('itemnachbarn_PCA_2M', engine, index_col='movieId')
        PCA = pd.read_sql_table('itemnachbarn_pca_1M', engine, index_col='movieId')
        print(PCA.head())
        print(PCA.shape)

        return PCA

    def compute_jaccard_similarity_score(self,x, y):
        """
        Jaccard Similarity J (A,B) = | Intersection (A,B) | /
                                        | Union (A,B) |
        """
        intersection_cardinality = len(set(x).intersection(set(y)))
        union_cardinality = len(set(x).union(set(y)))
        return intersection_cardinality / float(union_cardinality)

    def tanimoto_coefficient(self, p_vec, q_vec):
        tan_vec=[]
        for i in range (len(p_vec)-1):
            if p_vec[i] == q_vec[i]:
                tan_vec.append(1)
            else:
                tan_vec.append(0)

        return (sum(tan_vec)/len(tan_vec))


    def get_jaccard_similarity(self, df:pd.DataFrame, dff:pd.DataFrame):
        jaccard = []
        df = df.reset_index(drop=True)
        dff = dff.reset_index(drop=True)
        for index in range(0, df.shape[0]):
            row = df.iloc[index]
            vector = []
            for i in range(len(row) - 1):
                compare_item_1 = row[i]
                vector.append(compare_item_1)
            print('Row {} '.format(index))
            for index1 in range(0, dff.shape[0]):
                row1 = dff.iloc[index1]
                vector2 = []
                for j in range(len(row1) - 1):
                    compare_item_2 = row1[j]
                    vector2.append(compare_item_2)

                print('Row {}'.format(index1))
                coeff = self.compute_jaccard_similarity_score(vector, vector2)
                jaccard.append(coeff)
                break
        return jaccard



    def get_tanimoto(self, df:pd.DataFrame, dff:pd.DataFrame):
        tanimoto = []
        df = df.reset_index(drop=True)
        dff = dff.reset_index(drop=True)
        for index in range (0,df.shape[0]):
            row = df.iloc[index]
            vector = []
            for i in range (len(row)-1):
                compare_item_1 = row[i]
                vector.append(compare_item_1)
            print('Row {} '.format(index))
            for index1 in range(0,dff.shape[0]):
                row1 = dff.iloc[index1]
                vector2 = []
                for j in range (len(row1)-1):
                    compare_item_2 = row1[j]
                    vector2.append(compare_item_2)

                print('Row {}'.format(index1))
                coeff = self.tanimoto_coefficient(np.array(vector), np.array(vector2))
                tanimoto.append(coeff)
                break
        return tanimoto


if __name__ == '__main__':
    LC=ListComparing()
    user_FS=LC.read_from_sql()
    user_PCA=LC.read_from_sql_pca()
    tanimoto = LC.get_tanimoto(user_PCA, user_FS)
    tanimoto_df = pd.DataFrame(tanimoto)
    tanimoto_df.index = tanimoto_df.index +1
    tanimoto_df.index.name = 'userId'
    tanimoto_df.columns = ['tanimoto_coefficient']
    jaccard = LC.get_jaccard_similarity(user_PCA, user_FS)
    jaccard_df = pd.DataFrame(np.asarray(jaccard).transpose())
    jaccard_df.index = jaccard_df.index + 1
    jaccard_df.index.name = 'userId'
    jaccard_df.columns = ['jaccard_coefficient']
    engine = create_engine('mysql://root:banana@localhost/NACHBARN')  # enter your password and database names here
    coeff = pd.merge(tanimoto_df, jaccard_df, left_index=True, right_index=True)
    coeff.to_sql('Compared_User_1M', con=engine, index=True, index_label='movieId', if_exists='replace')
    print(coeff.head(10).to_latex())



