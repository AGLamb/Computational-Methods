from scipy.stats import norm
import statsmodels.api as sm
import random as rnd
import pandas as pd
import numpy as np
import random


def main():
    df_X, df_Y = Process_data()
    naiveTtest(df_X, df_Y)

    bootstrap(df_Y, df_X, "np")
    bootstrap(df_Y, df_X, "wild")
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

    print(f' Test rejects H0 is approximately: {(rejected / n) * 100:.2f}%')


def Regress_OLS(Dependent, Independent):
    Model = sm.OLS(Dependent, Independent)
    Results = Model.fit()
    # Results.summary()
    return Results, Results.resid


def bootstrap(df_y, df_x, bootstrap_type, B=99):
    df_Tn = pd.DataFrame()
    alpha = 0.05
    rejected = 0
    n = len(df_y.columns)

    for i in range(n):
        Model, Residuals = Regress_OLS(df_y[i], df_x)
        y_hat = Model.fittedvalues
        Tn_vector = list()
        vType = 1

        for j in range(B):
            y_star = list()

            for k in range(len(y_hat)):
                if bootstrap_type == "np":
                    vType = 1
                elif bootstrap_type == "wild":
                    vType = random.normalvariate(0, 1)
                y_star_i = y_hat[k] + random.choice(Residuals) * vType
                y_star.append(y_star_i)

            y_star = pd.DataFrame(y_star)
            Tn = test_stat(y_star, df_x)

            if abs(Tn) >= abs(norm.ppf(alpha / 2)):
                rejected += 1

            Tn_vector.append(Tn)

        Tn_vector = pd.DataFrame(Tn_vector)
        df_Tn = pd.concat([df_Tn, Tn_vector], axis=1)

    Rej_Rate = (rejected / n) * 100
    print(f' Test rejects H0 is approximately: {Rej_Rate:.2f}%')
    # print(df_Tn)
    return


def test_stat(y_vector, x_vector):
    Model, y_res = Regress_OLS(y_vector, x_vector)
    Tn = Model.params[2] / np.sqrt(Model.cov_HC0[2][2])
    return Tn


if __name__ == "__main__":
    main()
