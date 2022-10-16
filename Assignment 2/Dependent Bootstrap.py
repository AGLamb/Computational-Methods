from recombinator.optimal_block_length import optimal_block_length
from recombinator.block_bootstrap import circular_block_bootstrap
from statsmodels.tsa.api import AutoReg, adfuller
from scipy.stats import norm, t
import statsmodels.api as sm
import random as rnd
import pandas as pd
import numpy as np
import random
import math


B = 99
alpha = 0.05


def main():
    Y = getVariable('Timeseries_het.txt')

    # naive_rate = naiveTtest(Y)
    # print(f'Naive Test\nTest rejects H0 approximately: {naive_rate * 100:.2f}%')
    #
    # NP_rejection = bootstrap(Y, "np")
    # print(f'Non Parametric Bootstrap\nThe rejection rate is on average: {np.average(NP_rejection) * 100:.2f}%')
    #
    # Wild_rejection = bootstrap(Y, "wild")
    # print(f'Wild Bootstrap\nThe rejection rate is on average: {np.average(Wild_rejection) * 100:.2f}%')
    # #
    # Sieve_rejection = bootstrap(Y, "sieve")
    # print(f'Sieve Bootstrap\nThe rejection rate is on average: {np.average(Sieve_rejection) * 100:.2f}%')

    Block_rejection = bootstrap(Y, "block")
    print(f'Block Bootstrap\nThe rejection rate is on average: {np.average(Block_rejection) * 100:.2f}%')
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
            Tn = Simulate_type(df_y[:, i], bootstrap_type)
            Tn_column.append(Tn)

        Tn_column = np.array(Tn_column).transpose()
        p_value = rejection_rate(Tn_column)
        p_vector.append(p_value)

    p_vector = np.array(p_vector)
    return p_vector


def Simulate_type(df_y, type_btstrp):
    Model, Residuals = Regress_AR(df_y, [1])
    phi = Model.params[0]
    if type_btstrp == "wild":
        y_star = wild_simulation(df_y, Residuals, phi)
    elif type_btstrp == "sieve":
        y_star = sieve_simulation(df_y, Residuals, phi)
    elif type_btstrp == "block":
        y_star = block_simulation(df_y, Residuals, phi)
    else:
        Residuals -= np.average(Residuals)
        y_star = np_simulation(df_y, Residuals, phi)
    Tn = DF_manual(y_star)
    return Tn


def sieve_simulation(df_y, Residuals, phi):
    y_star = list()
    eta = list()
    Model, Res = Regress_AR(Residuals, [1, 2, 3, 4, 5])
    Res -= np.average(Res)

    for k in range(5):
        eta_i = Res[k]
        eta.append(eta_i)

    for j in range(5, len(Res)):
        eta_i = (Model.params[0] * eta[j - 1]) + (Model.params[1] * eta[j - 2]) + (Model.params[2] * eta[j - 3]) + \
                (Model.params[3] * eta[j - 4]) + (Model.params[4] * eta[j - 5]) + random.choice(Res)
        eta.append(eta_i)

    y_star.append((random.choice(eta) * random.normalvariate(0, 1)))
    y_star_i = ((1 - phi) * y_star[0]) + (random.choice(eta) * random.normalvariate(0, 1))
    y_star.append(y_star_i)

    for i in range(2, df_y.shape[0]):
        vType = random.normalvariate(0, 1)
        y_star_i = ((1 - phi) * y_star[i - 1]) + (phi * y_star[i - 2]) + (random.choice(eta) * vType)
        y_star.append(y_star_i)

    y_star = np.array(y_star)
    return y_star


def block_simulation(df_y, Residuals, phi):
    b_star = optimal_block_length(Residuals)
    b_star_sb = b_star[0].b_star_sb
    b_star_cb = math.ceil(b_star[0].b_star_cb)
    print(b_star_cb)
    return


def wild_simulation(df_y, Residuals, phi):
    y_star = list()
    y_star.append((random.choice(Residuals) * random.normalvariate(0, 1)))
    y_star_i = ((1 - phi) * y_star[0]) + (random.choice(Residuals) * random.normalvariate(0, 1))
    y_star.append(y_star_i)
    for i in range(2, df_y.shape[0]):
        vType = random.normalvariate(0, 1)
        y_star_i = ((1 - phi) * y_star[i - 1]) + (phi * y_star[i - 2]) + (random.choice(Residuals) * vType)
        y_star.append(y_star_i)
    y_star = np.array(y_star)
    return y_star


def np_simulation(df_y, Residuals, phi):
    y_star = list()
    y_star.append(random.choice(Residuals))
    y_star_i = ((1 - phi) * y_star[0]) + random.choice(Residuals)
    y_star.append(y_star_i)
    for i in range(2, df_y.shape[0]):
        y_star_i = ((1 - phi) * y_star[i - 1]) + (phi * y_star[i - 2]) + random.choice(Residuals)
        y_star.append(y_star_i)
    y_star = np.array(y_star)
    return y_star


def Regress_AR(Dependent, num_lags):
    Model = AutoReg(Dependent, lags=num_lags, trend='n')
    Results = Model.fit()
    return Results, Results.resid


def getVariable(pathfile):
    return pd.read_csv(pathfile, header=None).to_numpy()


def test_stat(deltaY, indep):
    Model, y_res = Regress_OLS(deltaY, indep)
    Tn = Model.params[0] / np.sqrt(Model.cov_HC0[0][0])
    return Tn


def Regress_OLS(deltaY, indep):
    Model = sm.OLS(deltaY, indep)
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
