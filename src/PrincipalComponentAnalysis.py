import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



class PrincipalComponentAnalysis(object):

    def normalize(self,data):
        # creates a copy of data
        X = tf.identity(data)
        # calculates the mean
        X -= tf.reduce_mean(data, axis=0)
        return X


    def explained_variance(self, matrix,cov):
        diag = np.diagonal(matrix)
        summe = sum(diag)
        diag_cov = np.diagonal(cov.numpy())
        print(diag_cov)
        i=0
        explained_variance= []
        for i in range (len(diag_cov)-1):
            varianz= diag_cov[i]/summe
            explained_variance.append(varianz)
            i= i+1
        return explained_variance

     # Note that the diagonal sum is still 3.448, which says that all 3 components account for all the multivariate variability.
    # The 1st principal component accounts for or "explains" 1.651 / 3.448 = 47.9 % of the overall variability;
    # the 2nd one explains 1.220 / 3.448 = 35.4 % of it;
    # the 3rd one explains .577 / 3.448 = 16.7 % of it.



    def tf_pca(self,df):
        df=df.head(501)
        X = tf.constant(df.values)
        X_normalized = self.normalize(X)
        Covarianz_Matrix = tfp.stats.covariance(X_normalized)
        print(sum(np.diagonal(Covarianz_Matrix.numpy())))
        e, v = tf.linalg.eigh(tf.tensordot(tf.transpose(Covarianz_Matrix), Covarianz_Matrix, axes=1))


        # sort eigenvectors by eigenvalues (tf sorts eigenvalues in non-decreasing order)
        e = e.numpy()
        v = v.numpy()
        idx = e.argsort()[::-1]
        e = e[idx]
        v = v[:, idx]
        print("Eigen Vectors: \n{} \nEigen Values: \n{}".format(v, e))
        expl_varianz = self.explained_variance(v,Covarianz_Matrix)
        print("Writing explained varianz to csv")
        pd.DataFrame(expl_varianz).to_csv('/data/explained_varianz.csv')
        print("Done")
        X_new = tf.transpose(tf.tensordot(v, tf.transpose(X), axes=1)).numpy()  #v does not need to be transposed, as TF represents each eigenvector in rows instead of columns
        df_new = pd.DataFrame(X_new,index=df.index)
        df_new.columns = df.columns
        return df_new,Covarianz_Matrix


if __name__ == '__main__':
    a = PrincipalComponentAnalysis()
    a.tf_pca()