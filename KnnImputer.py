#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class KNNImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, col_ix, job_type, n_neighbors=5, n_jobs=None):
        '''
        Args:
            col_ix (int): index of response column
            job_type (str): "regression" or "classification"
            n_neighbors (int): number of n neighbors
            n_jobs (int): amount of cores to use, use -1 for all cores
        '''
        self.col_ix = col_ix
        self.model = None
        if job_type=='regression':
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors,
                                             n_jobs=n_jobs)
        elif job_type=='classification':
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                              n_jobs=n_jobs)
        else:
            raise ValueError('Unvalid job_type. Only "regression" or "classification" is allowed.')
        
    def fit(self, X, y=None):
        '''
        Args:
            X (numpy.ndarray): Predictors to train the imputer. Only samples that
                corespond with val_to predict are used for training.
        '''
        X_temp = np.delete(np.copy(X), self.col_ix, 1)
        y = X[:,self.col_ix]
        index_train = np.invert(np.isnan(y))
        self.model.fit(X_temp[index_train,:], y[index_train])
        return self
    
    def transform(self, X):
        '''
        Args:
            X (numpy.ndarray): Original train data or new samples to predict values.
                If response contains values other than val_to_predict they are left
                unchanged, only val_to_predict are imputed.
                
        Returns:
            numpy.ndarray of imputed values
        '''
        X_temp = np.delete(np.copy(X), self.col_ix, 1)
        y = X[:,self.col_ix]
        index_fit = np.isnan(y)
        imputations = self.model.predict(X_temp[index_fit, :])
        np.place(y, index_fit, imputations)
        return y.reshape(-1,1)

