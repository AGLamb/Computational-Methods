from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
import numpy as np
import random as rnd


def main():
    df_X, df_Y = Process_data()  # get X and Y
    # naiveTtest(df_X, df_Y)
    print(np_res_bstrap(df_Y, df_X))
    return


def Process_data():  # X (25,3) and Y (25,10000) data
    return pd.read_csv('Regressors.txt', header=None), pd.read_csv('Observables.txt', header=None)


def naiveTtest(df_X, df_Y):  # Ex 2 t-test
    alpha = 0.05
    rejected = 0
    n = len(df_Y.columns)  # 10000 y column vectors
    for i in range(n):
        Model, y_res = Regress_OLS(df_Y[i], df_X)  # OLS on a column vector of y (25,1) and all of X (25,3)
        Tn = Model.params[2] / np.sqrt(Model.cov_HC0[2][2])  # t-stat w/ b_OLS,2 and s_2^2

        if abs(Tn) >= abs(norm.ppf(alpha / 2)):  # if Tn > inverse normal dist with alpha/2
            rejected += 1
    print(f' Test rejects H0 approximately: {(rejected / n) * 100:.2f}%')


def Regress_OLS(Dependent, Independent):
    Model = sm.OLS(Dependent, Independent)  # ordinary least squares
    Results = Model.fit()
    # Results.summary()
    return Results, Results.resid


def np_res_bstrap(df_y, df_x):  # Ex 3b
    eps_hat = df_y - df_x.dot(np.array(Regress_OLS(df_y, df_x)[0].params))  # eps_hat from beta_OLS, column vectors
    n = len(eps_hat.columns)
    df = pd.DataFrame()
    for i in range(n):
        eps_star = eps_hat[rnd.randint(0, n)]
        df = pd.concat([df, eps_star], axis=1)  # Haven't figured out how to put eps_star side by side
    return df


if __name__ == "__main__":
    main()
