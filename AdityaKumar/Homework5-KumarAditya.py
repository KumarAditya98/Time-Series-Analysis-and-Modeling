import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal


# Creating function to generate error
def gen_e(num,den,y):
    """
    :param num: The denominator to generate error from process
    :param den: The numerator to generate error from process
    :return: error process
    """
    np.random.seed(6313)
    num_e = np.r_[1,num].tolist()
    den_e = np.r_[1,den].tolist()
    system = (num_e, den_e, 1)
    t, e = signal.dlsim(system,y)
    return e


# Q1. Code for parameter estimation through Levenberg-Marquardt optimization. - Along with process generation.
def lm_param_estimate():
    """
    Includes ARMA (synthetic process) generation. - Used for testing.
    Then attempts to estimate the true coefficients of process through LM Algo.
    :return: None. Only prints in case of convergence or any error.
    """
    np.random.seed(6313)
    N = int(input("Enter the number of data samples:"))
    mean_e = int(input("Enter the mean of white noise:"))
    var_e = int(input("Enter the variance of white noise:"))
    na = int(input("Enter the AR portion order:"))
    nb = int(input("Enter the MA portion order:"))
    if na == 0 and nb == 0:
        print("This is just a white noise. Run program again.")
        return None
    print(
        'Enter the respective coefficients in the form:\n[y(t) + a1*y(t-1)) + a2*y(t-2) + ... = e (t) + b1*e(t-1) + b2*e(t-2) + ...]\n')
    ar_coeff = []
    for i in range(na):
        prompt = "Enter the coefficient for a" + str((i + 1))
        ar_coeff.append(float(input(prompt)))
    ma_coeff = []
    for i in range(nb):
        prompt = "Enter the coefficient for b" + str((i + 1))
        ma_coeff.append(float(input(prompt)))
    ar = np.r_[1, ar_coeff]
    ma = np.r_[1, ma_coeff]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    mean_y = mean_e * (1 + np.sum(ma_coeff)) / (1 + np.sum(ar_coeff))
    y = arma_process.generate_sample(N, scale=np.sqrt(var_e)) + mean_y
    if arma_process.isstationary:
        print('Process with given coefficients is Stationary.\n')
    else:
        print('Process with given coefficients is Non-Stationary.\n')
    max_iter = 100
    delta = 1e-6
    mu = 0.01
    mu_max = 1e10
    epsilon = 0.0001
    sse_plot = []
    iter = 0
    while iter < max_iter:
        #Initialize
        if iter == 0:
            error = np.array(y.copy()).reshape(len(y),1)
            theta = np.zeros(na+nb).reshape((na+nb),1)
        else:
            num = [0]*max(na,nb)
            den = [0]*max(na,nb)
            num[:na] = theta.ravel().tolist()[:na]
            den[:nb] = theta.ravel().tolist()[na:]
            error = gen_e(num,den,y)
        sse = np.dot(error.T, error)
        sse_plot.append(sse)
        big_x = np.empty(len(y)*(na+nb)).reshape(len(y),(na+nb))
        for i in range(len(theta)):
            theta_temp = theta.copy()
            theta_temp[i] = theta_temp[i] + delta
            num = [0]*max(na,nb)
            den = [0]*max(na,nb)
            num[:na] = theta_temp.ravel().tolist()[:na]
            den[:nb] = theta_temp.ravel().tolist()[na:]
            error_temp = gen_e(num,den,y)
            big_x[:,i] = (error.ravel() - error_temp.ravel())/delta
        hessian = np.dot(big_x.T,big_x)
        gradient = np.dot(big_x.T,error.reshape(len(y),1))
        identity = np.identity((na+nb))
        delta_theta = np.dot(np.linalg.inv((hessian+mu*identity)),gradient)
        theta_new = theta + delta_theta
        num = [0] * max(na, nb)
        den = [0] * max(na, nb)
        num[:na] = theta_new.ravel().tolist()[:na]
        den[:nb] = theta_new.ravel().tolist()[na:]
        error_new = gen_e(num, den, y)
        sse_new = np.dot(error_new.T,error_new)
        if sse_new < sse:
            if np.linalg.norm(delta_theta) < epsilon:
                theta = theta_new.copy()
                var_e = sse_new/(len(y)-(na+nb))
                covariance = var_e*np.linalg.inv(hessian)
                print("Algorithm has converged!!!")
                print(f"The Estimated Parameters are: {theta.ravel().tolist()}")
                print(f"The True Parameters are: {ar_coeff+ma_coeff}")
                print(f"The Covariance matrix is: {covariance}")
                print(f"The Variance of error is: {var_e.ravel()[0]}")
                num = theta.ravel().tolist()[:na]
                den = theta.ravel().tolist()[na:]
                temp = np.diag(covariance).tolist()
                num_temp = temp[:na]
                den_temp = temp[na:]
                if na != 0:
                    print(f"Confidence interval for AR parameters are:")
                for i in range(len(num)):
                    print(f"{num[i]-2*np.sqrt(num_temp[i])} < a{format(i+1)} < {num[i]+2*np.sqrt(num_temp[i])}")
                if nb != 0:
                    print(f"Confidence interval for MA parameters are:")
                for i in range(len(den)):
                    print(f"{den[i] - 2 * np.sqrt(den_temp[i])} < b{format(i + 1)} < {den[i] + 2 * np.sqrt(den_temp[i])}")
                if nb != 0:
                    print(f"The roots of the numerator are: {np.roots(np.r_[1,den].tolist())}")
                if na != 0:
                    print(f"The roots of the denominator are: {np.roots(np.r_[1,num].tolist())}")
                fig, ax = plt.subplots(figsize=(14,8))
                ax.plot(np.arange(len(sse_plot),dtype='int').tolist(),np.array(sse_plot).ravel(),label='Sum Square Error')
                plt.grid()
                plt.title("The Sum Square Error V/S Number of Iterations")
                plt.xlabel("Number of Iterations")
                plt.ylabel("Sum Square Error")
                plt.legend()
                plt.show()
                return None
            else:
                theta = theta_new.copy()
                mu = mu/10
        while sse_new >= sse:
            mu = mu*10
            if mu > mu_max:
                print(f"Error: Value of mu has become too large. Algorithm will become unstable if we proceed further.\t Printing Results:")
                theta = theta_new.copy()
                var_e = sse_new / (len(y) - (na + nb))
                covariance = var_e * np.linalg.inv(hessian)
                print(f"The Estimated Parameters are: {theta.ravel().tolist()}")
                print(f"The True Parameters are: {ar_coeff + ma_coeff}")
                print(f"The Covariance matrix is: {covariance}")
                print(f"The Variance of error is: {var_e.ravel()[0]}")
                num = theta.ravel().tolist()[:na]
                den = theta.ravel().tolist()[na:]
                temp = np.diag(covariance).tolist()
                num_temp = temp[:na]
                den_temp = temp[na:]
                if na != 0:
                    print(f"Confidence interval for AR parameters are:")
                for i in range(len(num)):
                    print(
                        f"{num[i] - 2 * np.sqrt(num_temp[i])} < a{format(i + 1)} < {num[i] + 2 * np.sqrt(num_temp[i])}")
                if nb != 0:
                    print(f"Confidence interval for MA parameters are:")
                for i in range(len(den)):
                    print(
                        f"{den[i] - 2 * np.sqrt(den_temp[i])} < b{format(i + 1)} < {den[i] + 2 * np.sqrt(den_temp[i])}")
                if nb != 0:
                    print(f"The roots of the numerator are: {np.roots(np.r_[1, den].tolist())}")
                if na != 0:
                    print(f"The roots of the denominator are: {np.roots(np.r_[1, num].tolist())}")
                return None
            delta_theta = np.dot(np.linalg.inv((hessian + mu * identity)), gradient)
            theta_new = theta + delta_theta
            num = [0] * max(na, nb)
            den = [0] * max(na, nb)
            num[:na] = theta_new.ravel().tolist()[:na]
            den[:nb] = theta_new.ravel().tolist()[na:]
            error_new = gen_e(num, den, y)
            sse_new = np.dot(error_new.T, error_new)
        iter = iter+1
        if iter > max_iter:
            print(
                f"Error: Value of iterations have become too large. Algorithm should've converged by now. Error in code logic.\t Printing Results:")
            theta = theta_new.copy()
            var_e = sse_new / (len(y) - (na + nb))
            covariance = var_e * np.linalg.inv(hessian)
            print(f"The Estimated Parameters are: {theta.ravel().tolist()}")
            print(f"The True Parameters are: {ar_coeff + ma_coeff}")
            print(f"The Covariance matrix is: {covariance}")
            print(f"The Variance of error is: {var_e.ravel()[0]}")
            num = theta.ravel().tolist()[:na]
            den = theta.ravel().tolist()[na:]
            temp = np.diag(covariance).tolist()
            num_temp = temp[:na]
            den_temp = temp[na:]
            if na != 0:
                print(f"Confidence interval for AR parameters are:")
            for i in range(len(num)):
                print(
                    f"{num[i] - 2 * np.sqrt(num_temp[i])} < a{format(i + 1)} < {num[i] + 2 * np.sqrt(num_temp[i])}")
            if nb != 0:
                print(f"Confidence interval for MA parameters are:")
            for i in range(len(den)):
                print(
                    f"{den[i] - 2 * np.sqrt(den_temp[i])} < b{format(i + 1)} < {den[i] + 2 * np.sqrt(den_temp[i])}")
            if nb != 0:
                print(f"The roots of the numerator are: {np.roots(np.r_[1,den].tolist())}")
            if na != 0:
                print(f"The roots of the denominator are: {np.roots(np.r_[1,num].tolist())}")
            return None
        theta = theta_new.copy()

lm_param_estimate()

