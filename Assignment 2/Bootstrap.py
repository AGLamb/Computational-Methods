from statsmodels.tsa.api import AutoReg, adfuller
from scipy.stats import norm, t
import statsmodels.api as sm
from tqdm import tqdm
import warnings as w
import pandas as pd
import numpy as np
import random


w.filterwarnings('ignore')
B = 99
alpha = 0.05


def main():
    df_X, df_Y = Process_data()
    df_X, df_Y = df_X.to_numpy(), df_Y.to_numpy()

    naive_rate = naiveTtest(df_X, df_Y)
    print(f'Naive Test\nTest rejects H0 is approximately: {naive_rate * 100:.2f}%')

    NP_rate, NP_pvalue = bootstrap(df_Y, df_X, "np")
    print(f'Non Paremetric Bootstrap\nThe rejection rate is on average: {np.average(NP_rate) * 100:.2f}%')
    print(f'The p-value is on average: {np.average(NP_pvalue):.2f}%')

    Wild_rate, Wild_pvalue = bootstrap(df_Y, df_X, "wild")
    print(f'Wild Bootstrap\nThe rejection rate is on average: {np.average(Wild_rate) * 100:.2f}%')
    print(f'The p-value is on average: {np.average(Wild_pvalue):.2f}%')

    Pair_rate, Pair_pvalue = bootstrap(df_Y, df_X, "wild")
    print(f'Pairs Bootstrap\nThe rejection rate is on average: {np.average(Pair_rate) * 100:.2f}%')
    print(f'The p-value is on average: {np.average(Pair_pvalue):.2f}%')
    return


def Monte_Carlo_pvalue(Tn_vector):
    t_hat = Tn_vector[0]
    counter = 0
    for i in range(1, Tn_vector.shape[0]):
        if Tn_vector[i] > t_hat:
            counter += 1
    return counter/B

def Process_data():
    return pd.read_csv('Assignment 2\Regressors.txt', header=None), pd.read_csv('Assignment 2\Observables.txt', header=None)


def naiveTtest(df_X, df_Y):
    rejected = 0
    n = df_Y.shape[1]

    for i in range(n):
        Tn = test_stat(df_Y[:, i], df_X)

        if abs(Tn) >= abs(t.ppf(alpha / 2, len(df_Y) - 2)):
            rejected += 1

    return rejected / n


def rejection_rate(t_star, df_y):
    rejected = 0

    for i in range(len(t_star)):

        if abs(t_star[i]) >= abs(t.ppf(alpha / 2, len(df_y) - 2)):
            rejected += 1

    return rejected / len(t_star)


def Regress_OLS(Dependent, Independent):
    Model = sm.OLS(Dependent, Independent)
    Results = Model.fit()
    # Results.summary()
    return Results, Results.resid


def bootstrap(df_y, df_x, bootstrap_type):
    n = df_y.shape[1]
    rr_vector = list()
    p_vector = list()

    for i in tqdm(range(n)):

        Tn_column = list()
        Tn = test_stat(df_y[:, i], df_x)
        Tn_column.append(Tn)

        for j in range(B):
            Tn = Simulate_type(df_y[:, i], df_x, bootstrap_type)
            Tn_column.append(Tn)

        Tn_column = np.array(Tn_column).transpose()
        rr_value = rejection_rate(Tn_column, df_y[:, i])
        rr_vector.append(rr_value)
        p_value = Monte_Carlo_pvalue(Tn_column)
        p_vector.append(p_value)

    rr_vector = np.array(rr_vector)
    p_vector = np.array(p_vector)
    return rr_vector, p_vector


def pair_simulation(df_y, df_x):
    y_star = list()
    x_star = list()
    for k in range(df_y.shape[0]):
        index = random.randint(0, len(df_y) - 1)
        x_star_i = df_x.iloc[[index]].tolist()
        y_star_i = df_y[k][index]
        y_star.append(y_star_i)
        x_star.append(x_star_i)

    x_star = np.array(x_star)
    y_star = np.array(y_star)
    Tn = test_stat(y_star, x_star)
    return  Tn 


def wild_simulation(df_y, df_x):
    y_star = list()
    X_Res = df_x.copy()
    X_Res[:, 1] = 0
    Model, Residuals = Regress_OLS(df_y, X_Res)
    y_hat = Model.fittedvalues

    for k in range(df_y.shape[0]):
        vType = random.normalvariate(0, 1)
        y_star_i = y_hat[k] + random.choice(Residuals) * vType
        y_star.append(y_star_i)

    y_star = np.array(y_star)
    Tn = test_stat(y_star, df_x)
    return Tn 


def np_simulation(df_y, df_x):
    y_star = list()
    X_Res = df_x.copy()
    X_Res[:, 1] = 0
    Model, Residuals = Regress_OLS(df_y, X_Res)
    y_hat = Model.fittedvalues

    for k in range(df_y.shape[0]):
        y_star_i = y_hat[k] + random.choice(Residuals)
        y_star.append(y_star_i)

    y_star = np.array(y_star)
    Tn = test_stat(y_star, df_x)
    return Tn 


def Simulate_type(df_y, df_x, type_btstrp):

    if type_btstrp == "pair":
        Tn = pair_simulation(df_y, df_x)
    elif type_btstrp == "wild":
        Tn = wild_simulation(df_y, df_x)
    else:
        Tn = np_simulation(df_y, df_x)

    return Tn


def test_stat(y_vector, x_vector):
    Model, y_res = Regress_OLS(y_vector, x_vector)
    Tn = Model.params[1] / np.sqrt(Model.cov_HC0[1][1])
    return Tn


if __name__ == "__main__":
    main()
