# KnnImputer
Regression or Classification imputer that uses K-nearest neighbors algorithm to imput missing or false data.
Algorithm is compatible with sklearn Pipeline and uses same comands as any sklearn code.

### Initial Parameters
- col_ix: index of response column
- val_to_predict: discrete value you want to predict e.g. np.nan
- job_type: is this regression or classification task
- n_neighbors: how manny neighbors to use when fitting the model
- n_jobs: how manny cpu cores to use


### Using the algortitm for predictions
After setting intial parameters, use **fit** function to train the Knn model on the data. After use **transform**
function to imput the data points using trained model. If you want to fit and transform the same data you can
use **fit_transform** function.

