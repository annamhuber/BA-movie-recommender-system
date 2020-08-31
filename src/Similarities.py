import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial import distance
from scipy import sparse

class Similarities(object):
    def get_cosine_similarity(self,df):
        # user similarity on replacing NAN by user avg
        print("Computing cosine similarity")
        df_sparse= sparse.csr_matrix(df,dtype=np.int32)
        b = cosine_similarity(df_sparse)
        np.fill_diagonal(b, 0)
        cosine_smlrty = pd.DataFrame(b, index=df.index)
        cosine_smlrty.columns = df.index
        print(cosine_smlrty.head())
        return cosine_smlrty


    def get_mahalanobis_distance(self,df,Cov):
        print("Starting Mahalonobisdistanz")
        vec = df.values
        i=0
        mahalanobis_dist = []
        Cov=Cov.numpy()
        for i in range(len(vec)-1):
            print("Calculating Row .{}".format(i))
            j=0
            row_vec = []
            for j in range(len(vec)-1):
                dist= distance.mahalanobis(vec[i],vec[j],Cov)
                dist= 1-dist

                row_vec.append(dist)
                j+=1

            mahalanobis_dist.append(row_vec)
            i+=1
        mahalanobis_dist=np.asarray(mahalanobis_dist, dtype=np.float32)
        np.fill_diagonal(mahalanobis_dist, 0)
        maha_similarity = pd.DataFrame(mahalanobis_dist) # , index=df.index)
        print(maha_similarity.head())
        return maha_similarity



