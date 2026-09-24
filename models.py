"""Model zoo (with small tuning grids) for the benchmark."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def gpr(n_feat=3, nu=2.5, restarts=8, seed=0):
    k = C(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=nu) \
        + WhiteKernel(1e-2, (1e-6, 1e0))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=restarts, random_state=seed)

def tuned(est, grid, cv=5):
    return GridSearchCV(est, grid, cv=KFold(cv, shuffle=True, random_state=1),
                        scoring="neg_mean_squared_error", n_jobs=1)

def zoo(n_feat=3, seed=0):
    sc = StandardScaler
    ttr = lambda m: TransformedTargetRegressor(m, transformer=StandardScaler())
    return {
        "Linear": ttr(make_pipeline(sc(), LinearRegression())),
        "Poly3-Ridge": ttr(make_pipeline(sc(), PolynomialFeatures(3), StandardScaler(), RidgeCV(alphas=np.logspace(-4, 2, 20)))),
        "KNN": ttr(tuned(make_pipeline(sc(), KNeighborsRegressor(weights="distance")), {"kneighborsregressor__n_neighbors": [2, 3, 5, 8]})),
        "SVR": ttr(tuned(make_pipeline(sc(), SVR()), {"svr__C": [1, 10, 100, 1000], "svr__gamma": [0.05, 0.2, 0.5, 1], "svr__epsilon": [0.01, 0.05]})),
        "RandomForest": RandomForestRegressor(500, min_samples_leaf=1, random_state=seed, n_jobs=1),
        "ExtraTrees": ExtraTreesRegressor(500, random_state=seed, n_jobs=1),
        "GradBoost": tuned(GradientBoostingRegressor(random_state=seed), {"max_depth": [2, 3], "n_estimators": [200, 500], "learning_rate": [0.03, 0.1]}),
        "XGBoost": tuned(XGBRegressor(random_state=seed, n_jobs=1, verbosity=0), {"max_depth": [2, 3, 4], "n_estimators": [300, 600], "learning_rate": [0.03, 0.1]}),
        "LightGBM": tuned(LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, min_child_samples=3), {"num_leaves": [4, 8], "n_estimators": [300, 600], "learning_rate": [0.03, 0.1]}),
        "CatBoost": tuned(CatBoostRegressor(random_seed=seed, verbose=0, thread_count=1), {"depth": [3, 5], "iterations": [500], "learning_rate": [0.03, 0.1]}),
        "MLP": ttr(tuned(make_pipeline(sc(), MLPRegressor(max_iter=4000, random_state=seed)), {"mlpregressor__hidden_layer_sizes": [(32, 32), (64, 64)], "mlpregressor__alpha": [1e-3, 1e-1], "mlpregressor__activation": ["tanh"]})),
        "GPR-Matern52-ARD": make_pipeline(sc(), gpr(n_feat, 2.5, seed=seed)),
        "GPR-RBF-ARD": make_pipeline(sc(), GaussianProcessRegressor(
            C(1.0, (1e-2, 1e2)) * RBF(np.ones(n_feat), (1e-2, 1e2)) + WhiteKernel(1e-2, (1e-6, 1)), normalize_y=True, n_restarts_optimizer=8, random_state=seed)),
    }


from sklearn.base import BaseEstimator, RegressorMixin, clone

class TrendGP(BaseEstimator, RegressorMixin):
    """Linear (or poly) trend + Gaussian-process residual. Returns predictive std like a GP."""
    def __init__(self, n_feat=2, deg=1, restarts=8):
        self.n_feat, self.deg, self.restarts = n_feat, deg, restarts
    def fit(self, X, y):
        self.sc_ = StandardScaler().fit(X)
        Z = self.sc_.transform(X)
        self.trend_ = make_pipeline(PolynomialFeatures(self.deg), RidgeCV(alphas=np.logspace(-4, 2, 20))).fit(Z, y)
        self.gp_ = gpr(self.n_feat, 2.5, self.restarts).fit(Z, y - self.trend_.predict(Z))
        return self
    def predict(self, X, return_std=False):
        Z = self.sc_.transform(X)
        mu, sd = self.gp_.predict(Z, return_std=True)
        mu = self.trend_.predict(Z) + mu
        return (mu, sd) if return_std else mu
