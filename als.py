import pandas as pd
import numpy as np
from scipy.linalg import pinv
from scipy.linalg import norm
from sklearn.utils.extmath import fast_dot
from sklearn.linear_model import LinearRegression


from toolbox.itertools_recipes import shuffle_and_deal
from toolbox.ml.linear_model.nnls import NNLS

def als_crossval(x_orig, n_latent, samples_regressor=None,
                 features_regressor=None, n_folds=10, n_iter=100, pos=False):
    # Todo: make this more general for other similar methods. Pass in created model object to fit

    n, m = x_orig.shape
    if isinstance(x_orig, pd.DataFrame):
        x = x_orig.values
    elif isinstance(x_orig, np.ndarray):
        x = x_orig.copy()
    else:
        raise ValueError("X must be a Pandas DataFrame or numpy array")
    zeroed = np.nan_to_num(x.astype(float))
    #zeroed = x.astype(float).fillna(0)

    nonzero_indices_per_row = [zeroed[s].nonzero()[0] for s in range(n)]
    #nonzero_indices_per_row = [zeroed.loc[s].nonzero()[0] for s in zeroed.index]
    nonzero_indices = []
    for i, nzipr in enumerate(nonzero_indices_per_row):
        for nz in nzipr:
            nonzero_indices.append((i, nz))
    n_nz = len(nonzero_indices)
    #fold_size = n_nz / n_folds
    folds = shuffle_and_deal(nonzero_indices, n_folds)
    losses = []
    for fold in folds:
        fold_len = len(fold)
        fold_copy = zeroed.copy()
        zipped_held_out_nzi = list(zip(*fold))
        try:
            fold_copy[zipped_held_out_nzi] = 0
            #fold_copy.iloc[zipped_held_out_nzi] = 0
        except Exception as e:
            n, m = fold_copy.shape
            rows, cols = zipped_held_out_nzi
            for arr, l in zip([rows, cols], [n, m]):
                print((arr.max() >= l or arr.min() < -l))
            print([row for row in rows if row >= n])
            print([col for col in cols if col >= m])
            #print(zipped_held_out_nzi)
            raise e
        als = ALSFactorizer(samples_regressor=samples_regressor,
                            features_regressor=features_regressor,
                            k=n_latent, max_iter=n_iter,
                            zero_is_nan=True, pos=pos)
        als.fit(zeroed)
        reconstructed = als.reconstruction_
        mask = np.ones(fold_copy.shape).astype(bool)
        mask[zipped_held_out_nzi] = False
        #mask = pd.DataFrame(True, index=fold_copy.index, columns=fold_copy.columns)
        #mask.iloc[zipped_held_out_nzi] = False
        #print(x)
        #print(reconstructed)
        #print(x - reconstructed)
        sse = (np.ma.array(x - reconstructed, mask=mask) ** 2).sum()
        mse = sse / float(fold_len)
        #print(sse, fold_len)
        loss = mse
        losses.append(loss)
    mean_loss = np.mean(losses)
    return mean_loss


class ALSFactorizer(object):

    def __init__(self, k, samples_regressor=None, features_regressor=None,
                 max_iter=100, tol=10E-5, zero_is_nan=False, pos=False):
        if samples_regressor is None or features_regressor is None:
            if pos:
                self.samples_regressor = NNLS()
                self.features_regressor = NNLS()
            else:
                self.samples_regressor = LinearRegression()
                self.features_regressor = LinearRegression()
        else:
            self.samples_regressor = samples_regressor
            self.features_regressor = features_regressor
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.components_ = None
        self.bases_ = None
        self.reconstruction_err_ = None
        self.reconstruction_ = None
        self.hi_ = None
        self.zero_is_nan = zero_is_nan

    def fit(self, X_orig, verbose=False, reconstruct_only_missing=False):
        """

        :param X_orig: Pandas DataFrame or numpy 2d array
        :param verbose:
        :return:
        """
        if isinstance(X_orig, pd.DataFrame):
            X = X_orig.values
        elif isinstance(X_orig, np.ndarray):
            X = X_orig.copy()
        else:
            raise ValueError("X must be a Pandas DataFrame or numpy array")

        if self.zero_is_nan:
            X[X == 0] = np.nan
        # Todo: warm starts
        # Todo: back up if RMSE gets worse
        n_samples, n_features = X.shape
        h = np.random.rand(self.k, n_features)
        h /= 100
        #h[:, 0] = np.mean(X, axis=0)
        w = np.random.rand(n_samples, self.k)
        w /= 100

        rmse_dif = 1.0
        rmse = 100000
        iter_idx = 0

        nonnan = np.logical_not(np.isnan(X))

        while iter_idx < self.max_iter and abs(rmse_dif) > self.tol:
            #print("h:", h)
            for i in range(n_samples):
                xi = X[i]
                y = xi[nonnan[i]]
                x = h[:, nonnan[i]]
                self.samples_regressor.fit(x.T, y)
                # need to assign to only some, when there are zeros.
                coeffs = self.samples_regressor.coef_
                w[i] = coeffs
            #print('w:', w)
            for j in range(n_features):
                xj = X[:, j]
                y = xj[nonnan[:, j]]
                x = w[nonnan[:, j]]
                self.features_regressor.fit(x, y)
                coeffs = self.features_regressor.coef_
                h[:, j] = coeffs
            #print(h)
            prediction = fast_dot(w, h)
            new_rmse = norm(X[nonnan] - prediction[nonnan])
            rmse_dif = rmse - new_rmse
            #if rmse_dif > 0:
            rmse = new_rmse
            if verbose:
                print("RMSE:{0}\tdif:{1}".format(new_rmse, rmse_dif))
            iter_idx += 1
            #print(rmse_dif)
            if rmse_dif < 0:
                break
        self.components_ = h.T
        self.hi_ = pinv(h)
        self.bases_ = w
        self.reconstruction_err_ = norm(X[nonnan] - prediction[nonnan])
        if reconstruct_only_missing:
            prediction[nonnan] = X[nonnan]
        if isinstance(X_orig, pd.DataFrame):
            self.reconstruction_ = pd.DataFrame(prediction, index=X_orig.index, columns=X_orig.columns)
        elif isinstance(X_orig, np.ndarray):
            self.reconstruction_ = prediction
        return self

    def recommend(self, u):
        u_prime = self.reconstruct_new(u)
        new_vals = u_prime[u.isnull()]
        return new_vals.sort_values(ascending=False)

    def reconstruct_new(self, u):
        new_sample = np.ma.array(u, mask=np.isnan(u.values))
        wu = np.ma.dot(new_sample, self.hi_)
        u_prime = np.ma.dot(wu, self.components_.T)
        return pd.Series(u_prime, index=u.index)