from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
import numpy as np
import random


def main():
    df_X, df_Y = Process_data()
    naiveTtest(df_X, df_Y)

    return


def Process_data():
    return pd.read_csv('Regressors.txt', header=None), pd.read_csv('Observables.txt', header=None)


def naiveTtest(df_X, df_Y):
    alpha = 0.05
    rejected = 0
    n = len(df_Y.columns)
    for i in range(n):
        Model, y_res = Regress_OLS(df_Y[i], df_X)
        Tn = Model.params[2] / np.sqrt(Model.cov_HC0[2][2])

        if abs(Tn) >= abs(norm.ppf(alpha / 2)):
            rejected += 1
    print(f' Test rejects H0 approximately: {(rejected / n) * 100:.2f}%')


def Regress_OLS(Dependent, Independent):
    Model = sm.OLS(Dependent, Independent)
    Results = Model.fit()
    Results.summary()
    return Results, Results.resid


if __name__ == "__main__":
    main()
