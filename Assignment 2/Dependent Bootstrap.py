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


def Process_data():
    return pd.read_csv('Timeseries_het.txt', header=None), pd.read_csv('Timeseries_dep.txt', header=None)


def naiveTtest(df_X, df_Y, c1, c2):
    rejected = 0
    n = df_Y.shape[1]

    for i in range(n):
        Tn = test_stat(df_Y[:, i], df_X)
        if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_Y) - 2)):
            rejected += 1

    result = (rejected / n)
    return result


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


def type_bootstrap(df_y, df_x, bootstrap_type, c1, c2):
    rejected = 0
    n = df_y.shape[1]

    for i in range(n):
        Model, y_hat = Regress_AR(df_y[:, i])
        vType = 1

        Tn = Dickey_Fuller(df_y[:, i], [2])
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
            Tn = Dickey_Fuller(y_star, [2])

            if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_y) - 1)):
                rejected += 1

        print(f'{i} cycle')

    Rej_Rate = (rejected / (n * (B + 1)))
    return Rej_Rate


def Dickey_Fuller(y_vector, lags_num):
    DF_stat = adfuller(y_vector, max_lags=lags_num)
    return DF_stat


def Regress_AR(Dependent):
    Model = AutoReg(Dependent, lags=[2])
    Results = Model.fit()
    # Results.summary()
    return Results, Results.fittedvalues


if __name__ == "__main__":
    main()
