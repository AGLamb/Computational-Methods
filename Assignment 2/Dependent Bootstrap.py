from statsmodels.tsa.api import AutoReg, adfuller
from scipy.stats import norm, t
import statsmodels.api as sm
import random as rnd
import pandas as pd
import numpy as np
import random


B = 99
alpha = 0.05


def main():
    df_X, df_Y = Process_data()
    df_X, df_Y = df_X.to_numpy(), df_Y.to_numpy()

    naive_rate = naiveTtest(df_X)
    naive_pvalue = 1 - naive_rate
    print(f'Naive Test\nTest rejects H0 is approximately: {naive_rate * 100:.2f}%')
    print(f'The p-value is approximately: {naive_pvalue:.2f}')

    NP_rejection = bootstrap(df_X, "np")
    NP_pvalue = 1 - NP_rejection
    print(f'Non Paremetric Bootstrap\nThe rejection rate is on average: {np.average(NP_rejection) * 100:.2f}%')
    print(f'The p-value is on average: {np.average(NP_pvalue):.2f}')

    Wild_rejection = bootstrap(df_X, "wild")
    Wild_pvalue = 1 - Wild_rejection
    print(f'Wild Bootstrap\nThe rejection rate is on average: {np.average(Wild_rejection) * 100:.2f}%')
    print(f'The p-value is on average: {np.average(Wild_pvalue):.2f}')

    Pair_rejection = bootstrap(df_X, "wild")
    Pair_pvalue = 1 - Pair_rejection
    print(f'Pairs Bootstrap\nThe rejection rate is on average: {np.average(Pair_rejection) * 100:.2f}%')
    print(f'The p-value is on average: {np.average(Pair_pvalue):.2f}')
    return


def Process_data():
    return pd.read_csv('Timeseries_het.txt', header=None), pd.read_csv('Timeseries_dep.txt', header=None)


def naiveTtest(df_X):
    rejected = 0
    n = df_Y.shape[1]

    for i in range(n):
        Tn = test_stat(df_Y[:, i], df_X)
        if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_Y) - 2)):
            rejected += 1

    result = (rejected / n)
    return result


def rejection_rate(t_star, df_y):
    rejected = 0
    for i in range(len(t_star)):
        if abs(t_star[i]) >= abs(t.ppf(alpha / 2, len(df_y) - 2)):
            rejected += 1
    return rejected / (len(t_star) + 1)


def bootstrap(df_y, bootstrap_type):
    n = 200  # df_y.shape[1]
    p_vector = list()
    for i in range(n):

        Tn_column = list()
        Tn = adfuller(df_y[:, i])
        Tn_column.append(Tn)

        for j in range(B):
            y_star, Tn = Simulate_type(df_y[:, i], bootstrap_type)
            Tn_column.append(Tn)

        Tn_column = np.array(Tn_column).transpose()
        p_value = rejection_rate(Tn_column, df_y[:, i])
        p_vector.append(p_value)

    p_vector = np.array(p_vector)
    return p_vector


def Simulate_type(df_y, type_btstrp):
    y_star = list()
    Model, Residuals = Regress_AR(df_y)
    y_hat = Model.fittedvalues
    for k in range(df_y.shape[0]):
        if type_btstrp == "np":
            vType = 1
            y_star_i = y_hat[k] + random.choice(Residuals) * vType
            y_star.append(y_star_i)
        elif type_btstrp == "wild":
            vType = random.normalvariate(0, 1)
            y_star_i = y_hat[k] + random.choice(Residuals) * vType
            y_star.append(y_star_i)
    y_star = np.array(y_star)
    Tn = adfuller(y_star)
    return y_star, Tn


def Regress_AR(Dependent):
    Model = AutoReg(Dependent, lags=[2])
    Results = Model.fit()
    # Results.summary()
    return Results, Results.fittedvalues


if __name__ == "__main__":
    main()
