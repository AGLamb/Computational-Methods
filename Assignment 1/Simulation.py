from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import acf, graphics, pacf
from scipy.stats import ttest_1samp
import statsmodels.api as sm
import pandas as pd
import numpy as np
import random

B_iterations = 999
Alpha = 0.1


def main():
    """Loading the data"""
    df_X, df_Y, df_True, df_False = Process_data()

    """Estimating a simple regression"""
    Model, y_res = Regress_OLS(df_Y, df_X)
    db_test = durbin_watson(y_res)

    """Running the Monte Carlo Simulation to get t* and DB* statistics """
    """rho_star,"""
    db_star = Monte_Carlo(df_X, B_iterations)

    """Calculating the critical values and the MC p-value"""
    c1, c2 = critical_values(db_star, Alpha)
    p_value_db = MC_Pvalue(db_star, db_test, c1, c2)

    """Printing and plotting the results"""
    print(f'Question 2 \nThe Rejection Region is: (0, {c1:.2f}) U ({c2:.2f}, 4)')
    print(f'Monte Carlo p-value = {p_value_db:.2f}')

    """Calculating the rejection rate of the True null on the Null"""
    rejection_rate_true = rejection_rate_db_test(df_True, df_X, c1, c2)
    print(f'Question 3 \nThe rejection rate is: {rejection_rate_true:.2f}')

    """Question 5"""
    rejection_rate_false = rejection_rate_db_test(df_False, df_X, c1, c2)
    print(f'Question 5 \nThe rejection rate is: {rejection_rate_false:.2f}')

    # plotter()
    return


"""Pending the creation of a function to plot the results from above"""


def MC_Pvalue(db_star, db_test, c1, c2):
    rejected = 0
    for i in range(len(db_star)):
        if db_star[i] > c2 or db_star[i] < c1:
            rejected += 1

    if db_test > c2 or db_test < c1:
        rejected += 1

    return rejected / (len(db_star) + 1)


def plotter():
    return


def Monte_Carlo(df, B):
    N = df.shape[0]
    # rho_star = list()
    db_star = list()
    for i in range(B):
        """Simulating N samples, choosing N as the X matrix dimension"""
        y_star = Simulate(N)

        """Regressing the simulated sample on the original independent variables"""
        # Model_star, y_star_res = Regress_OLS(y_star, df)

        """Estimating an AR(1) model with the residuals"""
        # AR_star_Model = Regress_AR(y_star_res, 1)

        """Calculating the t- and DB- statistics"""
        # rho_star.append(AR_star_Model.params[1])
        db_star.append(durbin_watson(y_star))

    return np.array(db_star)  # , np.array(rho_star)


def Regress_OLS(Dependent, Independent):
    Model = sm.OLS(Dependent, Independent)
    Results = Model.fit()
    return Results, Results.resid


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
    c2_th_statistic = 100 * (1 - (alpha / 2))
    c2 = np.percentile(db_star, int(c2_th_statistic))

    c1_th_statistic = 100 * (alpha / 2)
    c1 = np.percentile(db_star, int(c1_th_statistic))
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
