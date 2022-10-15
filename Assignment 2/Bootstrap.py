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
    # Exercise_1()
    # Exercise_2()
    return


def Exercise_2A():
    df_X, df_Y = Process_data_2()
    df_X, df_Y = df_X.to_numpy(), df_Y.to_numpy()
    return


def Exercise_1():
    df_X, df_Y = Process_data()
    df_X, df_Y = df_X.to_numpy(), df_Y.to_numpy()

    naive_rate = naiveTtest(df_X, df_Y)
    naive_pvalue = 1 - naive_rate
    print(f' Test rejects H0 is approximately: {naive_rate * 100:.2f}%')
    print(f' The p-value is approximately: {naive_pvalue:.2f}')

    NP_rate = type_bootstrap(df_Y, df_X, "np")
    NP_pvalue = 1 - NP_rate
    print(f' Test rejects H0 is approximately: {NP_rate:.2f}%')
    print(f' The p-value is approximately: {NP_pvalue:.2f}')

    Wild_rate = type_bootstrap(df_Y, df_X, "wild")
    Wild_pvalue = 1 - Wild_rate
    print(f' Test rejects H0 is approximately: {Wild_rate * 100:.2f}%')
    print(f' The p-value is approximately: {Wild_pvalue:.2f}')

    Pair_rate = pair_bootstrap(df_Y, df_X)
    Pair_pvalue = 1 - Pair_rate
    print(f' Test rejects H0 is approximately: {Pair_rate * 100:.2f}%')
    print(f' The p-value is approximately: {Pair_pvalue:.2f}')
    return


def Process_data_2():
    return pd.read_csv('Timeseries_het.txt', header=None), pd.read_csv('Timeseries_dep.txt', header=None)


def Process_data_1():
    return pd.read_csv('Regressors.txt', header=None), pd.read_csv('Observables.txt', header=None)


def naiveTtest(df_X, df_Y):
    rejected = 0
    n = df_Y.shape[1]

    for i in range(n):
        Tn = test_stat(df_Y[:, i], df_X)
        if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_Y) - 2)):
            rejected += 1

    result = (rejected / n)
    return result


def Regress_OLS(Dependent, Independent):
    Model = sm.OLS(Dependent, Independent)
    Results = Model.fit()
    # Results.summary()
    return Results, Results.resid


def pair_bootstrap(df_y, df_x):
    rejected = 0
    n = df_y.shape[1]

    for i in range(n):

        Tn = test_stat(df_y[i], df_x)
        if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_y) - 2)):
            rejected += 1

        for j in range(B):
            y_star = list()
            x_star = np.zeros(3, 0)

            for k in range(len(df_y)):
                index = random.randint(0, len(df_y) - 1)
                x_star_i = df_x.iloc[[index]].to_numpy()
                y_star_i = df_y[i][index]
                y_star.append(y_star_i)
                x_star = np.concatenate([x_star, x_star_i], axis=1)

            y_star = np.array(y_star)
            Tn = test_stat(y_star, x_star)

            if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_y) - 1)):
                rejected += 1

        print(f'{i} - Cycles Done')

    Rej_Rate = (rejected / (n * (B + 1)))
    return Rej_Rate


def type_bootstrap(df_y, df_x, bootstrap_type):
    rejected = 0
    n = df_y.shape[1]

    for i in range(n):
        Model, Residuals = Regress_OLS(df_y[:, i], df_x)
        y_hat = Model.fittedvalues
        vType = 1

        Tn = test_stat(df_y[:, i], df_x)
        if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_y) - 2)):
            rejected += 1

        for j in range(B):
            y_star = list()

            for k in range(len(y_hat)):

                if bootstrap_type == "np":
                    vType = 1
                elif bootstrap_type == "wild":
                    vType = random.normalvariate(0, 1)
                y_star_i = y_hat[k] + random.choice(Residuals) * vType
                y_star.append(y_star_i)

            y_star = np.array(y_star)
            Tn = test_stat(y_star, df_x)

            if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_y) - 1)):
                rejected += 1

        print(f'{i} cycle')

    Rej_Rate = (rejected / (n * (B + 1)))
    return Rej_Rate


def Dickey_Fuller(y_vector):
    DF_stat = adfuller(y_vector)
    return DF_stat


def test_stat(y_vector, x_vector):
    Model, y_res = Regress_OLS(y_vector, x_vector)
    Tn = Model.params[2] / np.sqrt(Model.cov_HC0[2][2])
    return Tn


if __name__ == "__main__":
    main()
