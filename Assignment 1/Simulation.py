from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import acf, graphics, pacf
from scipy.stats import ttest_1samp
import statsmodels.api as sm
import pandas as pd
import numpy as np
import random


B_iterations = 9999
Alpha = 0.1


def main():
    """Loading the data"""
    df_X, df_Y, df_True, df_False = Process_data()

    """Estimating a simple regression"""
    Model, y_res = Regress_OLS(df_Y, df_X)
    db_test = durbin_watson(y_res)

    """Estimating the AR(1) Model with the errors of the given matrix"""
    AR_errors = Regress_AR(y_res, 1)
    t = AR_errors.params[1]

    """Running the Monte Carlo Simulation to get t* and DB* statistics """
    # Low_limit_statistic, High_limit_statistic = Alpha * (B_iterations + 1), (1 - Alpha) * (B_iterations + 1)
    t_star, db_star = Monte_Carlo(df_X, B_iterations)
    # db_star.sort()
    # print(db_star[int(Low_limit_statistic)], db_star[int(High_limit_statistic)])

    """Calculating the Monte Carlo p-values"""
    p_value_t = min(np.where(t < t_star, True, False).mean(), np.where(t_star <= t, True, False).mean())
    p_value_db = min(np.where(db_test < db_star, True, False).mean(), np.where(db_star <= db_test, True, False).mean())
    print(p_value_t, p_value_db)
    
    """Calculating the Critical values c1 and c2 for db"""
    c1, c2 = critical_values(db_star, Alpha)
    print(c1, c2)
    # plotter()

    return


"""Pending the creation of a function to plot the results from above"""


def plotter():
    return


def Monte_Carlo(df, B):
    N = df.shape[0]
    t_star = list()
    db_star = list()
    for i in range(B):
        """Simulating N samples, choosing N as the X matrix dimension"""
        y_star = Simulate(N)

        """Regressing the simulated sample on the original independent variables"""
        Model_star, y_star_res = Regress_OLS(y_star, df)

        """Estimating an AR(1) model with the residuals"""
        AR_star_Model = Regress_AR(y_star_res, 1)

        """Calculating the t- and DB- statistics"""
        t_star.append(AR_star_Model.params[1])
        db_star.append(durbin_watson(y_star_res))

    return np.array(t_star), np.array(db_star)


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
    return pd.read_csv('Regressors.txt', header=None), pd.read_csv('Observables.txt', header=None), \
           pd.read_csv('True_null.txt', header=None), pd.read_csv('False_null.txt', header=None)


def critical_values(db_star, alpha):
    c2_th_percentile = 100 * (1 - alpha) 
    c2 = np.percentile(db_star, int(c2_th_percentile))

    c1_th_percentile = 100 * (alpha) 
    c1 = np.percentile(db_star, int(c1_th_percentile))
    return c1, c2

main()
