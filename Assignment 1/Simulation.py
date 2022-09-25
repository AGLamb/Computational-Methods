from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import acf, graphics, pacf
from scipy.stats import ttest_1samp
import statsmodels.api as sm
import pandas as pd
import numpy as np
import random

B_iterations = 99999
Alpha = 0.1


def main():
    """Loading the data"""
    df_X, df_Y, df_True, df_False = Process_data()

    """Estimating a simple regression"""
    Model, y_res = Regress_OLS(df_Y, df_X)
    db_test = durbin_watson(y_res)

    """Estimating the AR(1) Model with the errors of the given matrix"""
    AR_errors = Regress_AR(y_res, 1)
    rho = AR_errors.params[1]

    """Running the Monte Carlo Simulation to get t* and DB* statistics """
    rho_star, db_star = Monte_Carlo(df_X, B_iterations)

    """Calculating the Monte Carlo p-values"""
    p_value_db = min(np.where(db_test < db_star, True, False).mean(), np.where(db_star <= db_test, True, False).mean())

    """Calculating the Critical values c1 and c2 for db"""
    c1, c2 = critical_values(db_star, Alpha)

    """Printing and plotting the results"""
    print(f'The Rejection Region is: (0, {c1:.2f}) U ({c2:.2f}, 4)')
    print(f'Monte Carlo p-value = {p_value_db:.2f}')
    # plotter()
    
    """Calculating the rejection rate of the True null on the Null"""
    rejection_rate_true = rejection_rate_db_test(df_True, df_X, c1, c2)
    print(f'The rejection rate is: {rejection_rate_true:.2f}')

    """Question 5"""
    rejection_rate_false = rejection_rate_db_test(df_False, df_X, c1, c2)
    print(f'The rejection rate is: {rejection_rate_false:.2f}')
    return


"""Pending the creation of a function to plot the results from above"""


def plotter():
    return


def Monte_Carlo(df, B):
    N = df.shape[0]
    rho_star = list()
    db_star = list()
    for i in range(B):
        """Simulating N samples, choosing N as the X matrix dimension"""
        y_star = Simulate(N)

        """Regressing the simulated sample on the original independent variables"""
        Model_star, y_star_res = Regress_OLS(y_star, df)

        """Estimating an AR(1) model with the residuals"""
        AR_star_Model = Regress_AR(y_star_res, 1)

        """Calculating the t- and DB- statistics"""
        rho_star.append(AR_star_Model.params[1])
        db_star.append(durbin_watson(y_star_res))

    return np.array(rho_star), np.array(db_star)


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

    c1_th_percentile = 100 * alpha
    c1 = np.percentile(db_star, int(c1_th_percentile))
    return c1, c2

    
def rejection_rate_db_test(df_Y, df_X, c1, c2):
    rejected = 0
    accepted = 0
    for column in df_Y.columns:

        Model, y_res = Regress_OLS(df_Y[column], df_X)
        db_test = durbin_watson(y_res)

        if db_test < c1 or db_test > c2:
            rejected += 1
        else:
            accepted += 1
    return 100 * (rejected / (rejected + accepted))


main()
