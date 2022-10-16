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
    print(f'Naive Test\nTest rejects H0 is approximately: {naive_rate * 100:.2f}%')

    NP_rate = bootstrap(df_Y, df_X, "np")
    print(f'Non Paremetric Bootstrap\nThe rejection rate is on average: {np.average(NP_rate) * 100:.2f}%')

    Wild_rate = bootstrap(df_Y, df_X, "wild")
    print(f'Wild Bootstrap\nThe rejection rate is on average: {np.average(Wild_rate) * 100:.2f}%')

    Pair_rate = bootstrap(df_Y, df_X, "wild")
    print(f'Pairs Bootstrap\nThe rejection rate is on average: {np.average(Pair_rate) * 100:.2f}%')
    return


def Process_data():
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


def rejection_rate(t_star, df_y):
    rejected = 0
    for i in range(len(t_star)):
        if abs(t_star[i]) >= abs(t.ppf(alpha / 2, len(df_y) - 2)):
            rejected += 1
    return rejected / (len(t_star) + 1)


def Regress_OLS(Dependent, Independent):
    Model = sm.OLS(Dependent, Independent)
    Results = Model.fit()
    # Results.summary()
    return Results, Results.resid


def bootstrap(df_y, df_x, bootstrap_type):
    n = 200  # df_y.shape[1]
    p_vector = list()
    for i in range(n):

        Tn_column = list()
        Tn = test_stat(df_y[:, i], df_x)
        Tn_column.append(Tn)

        for j in range(B):
            y_star, Tn = Simulate_type(df_y[:, i], df_x, bootstrap_type)
            Tn_column.append(Tn)

        Tn_column = np.array(Tn_column).transpose()
        p_value = rejection_rate(Tn_column, df_y[:, i])
        p_vector.append(p_value)

    p_vector = np.array(p_vector)
    return p_vector


def Simulate_type(df_y, df_x, type_btstrp):
    y_star = list()
    if type_btstrp == "pair":
        x_star = list()
        for k in range(df_y.shape[0]):
            index = random.randint(0, len(df_y) - 1)
            x_star_i = df_x.iloc[[index]].tolist()
            y_star_i = df_y[i][index]
            y_star.append(y_star_i)
            x_star.append(x_star_i)
        x_star = np.array(x_star)
        y_star = np.array(y_star)
        Tn = test_stat(y_star, x_star)
    else:
        X_Res = df_x.copy()
        X_Res[:, 1] = 0
        Model, Residuals = Regress_OLS(df_y, X_Res)
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
        Tn = test_stat(y_star, df_x)
    return y_star, Tn


def test_stat(y_vector, x_vector):
    Model, y_res = Regress_OLS(y_vector, x_vector)
    Tn = Model.params[1] / np.sqrt(Model.cov_HC0[1][1])
    return Tn


if __name__ == "__main__":
    main()
