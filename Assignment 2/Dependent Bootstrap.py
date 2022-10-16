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
    Y = getVariable('Timeseries_het.txt')

    naive_rate = naiveTtest(Y)
    print(f'Naive Test\nTest rejects H0 is approximately: {naive_rate * 100:.2f}%')

    NP_rejection = bootstrap(Y, "np")
    print(f'Non Parametric Bootstrap\nThe rejection rate is on average: {np.average(NP_rejection) * 100:.2f}%')

    Wild_rejection = bootstrap(Y, "wild")
    print(f'Wild Bootstrap\nThe rejection rate is on average: {np.average(Wild_rejection) * 100:.2f}%')

    # Sieve_rejection = bootstrap(df_X, "wild")
    # print(f'Wild Bootstrap\nThe rejection rate is on average: {np.average(Sieve_rejection) * 100:.2f}%')
    #
    # Block_rejection = bootstrap(df_X, "wild")
    # print(f'Wild Bootstrap\nThe rejection rate is on average: {np.average(Block_rejection) * 100:.2f}%')
    return


def difference_operator(df):
    df_delta = df.copy()
    for i in range(df_delta.shape[0] - 1, 1, -1):
        df_delta[i] = df_delta[i] - df_delta[i - 1]
    return df_delta


def lag_operator(df):
    df_lagged = df.copy()
    for i in range(df_lagged.shape[0] - 1, 1, -1):
        df_lagged[i] = df_lagged[i - 1]
    return df_lagged


def naiveTtest(df_Y):
    rejected = 0
    n = df_Y.shape[1]

    for i in range(n):
        Tn = DF_manual(df_Y[:, i])
        if Tn < -1.95:
            rejected += 1

    return rejected / n


def rejection_rate(t_star):
    rejected = 0
    for i in range(t_star.shape[0]):
        if t_star[i] < -1.95:
            rejected += 1
    return rejected / (t_star.shape[0])


def bootstrap(df_y, bootstrap_type):
    n = 100  # df_y.shape[1]
    p_vector = list()
    for i in range(n):

        Tn_column = list()
        Tn = DF_manual(df_y[:, i])
        Tn_column.append(Tn)

        for j in range(B):
            y_star, Tn = Simulate_type(df_y[:, i], bootstrap_type)
            Tn_column.append(Tn)

        Tn_column = np.array(Tn_column).transpose()
        p_value = rejection_rate(Tn_column)
        p_vector.append(p_value)

    p_vector = np.array(p_vector)
    return p_vector


def Simulate_type(df_y, type_btstrp):
    y_star = list()
    Model, Residuals = Regress_AR(df_y)
    phi = Model.params[0]
    if type_btstrp == "np":
        vType = 1
        y_star.append(random.choice(Residuals) * vType)
        y_star.append((1 - phi) * y_star[0] + random.choice(Residuals) * vType)
    elif type_btstrp == "wild":
        vType = random.normalvariate(0, 1)
        y_star.append(random.choice(Residuals) * vType)
        y_star.append((1 - phi) * y_star[0] + random.choice(Residuals) * vType)
    for i in range(2, df_y.shape[0]):
        if type_btstrp == "np":
            vType = 1
            y_star_i = (1 - phi) * y_star[i - 1] + phi * y_star[i - 2] + random.choice(Residuals) * vType
            y_star.append(y_star_i)
        elif type_btstrp == "wild":
            vType = random.normalvariate(0, 1)
            y_star_i = (1 - phi) * y_star[i - 1] + phi * y_star[i - 2] + random.choice(Residuals) * vType
            y_star.append(y_star_i)
    y_star = np.array(y_star)
    Tn = DF_manual(y_star)
    return y_star, Tn


def Regress_AR(Dependent):
    Model = AutoReg(Dependent, lags=[1])
    Results = Model.fit()
    # Results.summary()
    return Results, Results.resid


def getVariable(pathfile):
    return pd.read_csv(pathfile, header=None).to_numpy()


def test_stat(deltaY, indep):
    Model, y_res = Regress_OLS(deltaY, indep)
    Tn = Model.params[0] / np.sqrt(Model.cov_HC0[0][0])
    return Tn


def Regress_OLS(deltaY, indep):
    Model = sm.OLS(deltaY, indep, hasconst=False)
    Results = Model.fit()
    return Results, Results.resid


def DF_manual(df_Y):
    delta_Y = difference_operator(df_Y)
    delta_Y1 = lag_operator(delta_Y)
    delta_Y1 = delta_Y1.reshape(len(delta_Y1), 1)
    df_Y1 = lag_operator(df_Y)
    df_Y1 = df_Y1.reshape(len(df_Y1), 1)
    indep = np.concatenate([df_Y1, delta_Y1], axis=1)
    return test_stat(delta_Y, indep)


if __name__ == "__main__":
    main()
