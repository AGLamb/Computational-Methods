from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import acf, graphics, pacf
from scipy.stats import ttest_1samp
import statsmodels.api as sm
import pandas as pd
import numpy as np
import random


def main():
    """Loading the data"""
    df_X, df_Y, df_True, df_False = Process_data()

    """Estimating a simple regression"""
    Model, y_res = Regress_OLS(df_Y, df_X)
    AR_error = Regress_AR(y_res, 1)
    t = AR_error.params[1]

    t_star = Monte_Carlo(df_X, 99999)

    p_value = min(np.where(t < t_star, True, False).mean(), np.where(t_star <= t, True, False).mean())
    print(p_value)

    return


def Monte_Carlo(df, B):
    N = df.shape[0]
    t_star = list()
    for i in range(B):

        """Simulating N samples, choosing N as the X matrix dimension"""
        y_star = Simulate(N)

        """Regressing the simulated sample on the original independent variables"""
        Model_star, y_star_res = Regress_OLS(y_star, df)

        """Estimating an AR(1) model with the residuals"""
        AR_star_Model = Regress_AR(y_star_res, 1)
        t_star.append(AR_star_Model.params[1])
    return np.array(t_star)


def Regress_OLS(Dependent, Independent):
    Model = sm.OLS(Dependent, Independent)
    Results = Model.fit()
    return Results, Results.resid


def Regress_AR(Variable, Lags):
    Model = AutoReg(Variable, Lags)
    Results = Model.fit()
    return Results


def Simulate(number):
    y_star = list()
    for i in range(number):
        observation = random.normalvariate(0, 1)
        y_star.append(observation)
    Simulated_vector = np.array(y_star)
    return Simulated_vector


def Process_data():
    return pd.read_csv('Regressors.txt', header=None), pd.read_csv('Observables.txt', header=None),\
           pd.read_csv('True_null.txt', header=None), pd.read_csv('False_null.txt', header=None)


main()
