

import numpy as np
import matplotlib.pyplot as plt
import random

import scipy
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sb


class Samples_data:
    def __init__(self, points, noise, xlim = [-1,1]):
        self.points = points
        self.noise = noise
        self.xlim = xlim

# Compute weights for Basis Fucntion
def Compute_Weight_Analytical_BF(Samples, Model):
    N_samples = len(Samples[0])
    # Calculate design matrix Phi

    x = Samples[0]
    y = Samples[1]

    Phi = Model.Compute_DesignMatrix(x)
    # Calculate parameters w and alpha
    w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y
    #Calculate error
    error = sum((y - Phi @ w)**2) / len(y)
    # Weights
    w = np.expand_dims(w, axis=1)

    return w, error


def GP_Posterior(X,Y,X_test, kernel_func, noise):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    X_ = np.expand_dims(X, 1)
    # Kernel of the observations
    Σ11 = kernel_func(X_, X_)
    Q = Σ11 + noise*noise*np.identity(len(X))
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X_, X_test)
    # Solve
    #solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T

    solved = Σ12.T @ np.linalg.inv(Q)

    # Compute posterior mean
    μ2 = solved @ Y
    # Compute the posterior covariance
    Σ22 = kernel_func(X_test, X_test)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance

#GENERATE DATA-------------------------------------------------------------

def generate_GroundTruth_Function(option = 1,N_points = 200,
                                  f_domain = [-1.0,1.0]):

    f_range  = abs(f_domain[1]-f_domain[0])
    x = np.arange(start=f_domain[0], stop=f_domain[1], step=f_range/N_points)
    y = []
    if option == 1:#Sin
        y = np.sin(x)
    elif option == 2:#Polinomial
        y = 5*(x+0.3)**4-2.9*(x+0.3)**3-3.3*(x+0.3)**2+0.3*(x+0.3)+0.4
    elif option == 3:  # Polinomial first degree /for bayesian study
        y = -0.3 + 0.5*x;
    return x,y


def generate_Noise_SampleData(mu = 0, sigma = 0.05, N_samples = 10):
    n = np.random.normal(mu, sigma, N_samples)
    return n

def generate_Samples(ground_truth, N_samples,pick = "Random",
                     add_noise = False, mu = 0, sigma =0.2, option = 1,
                     sample_x_from_distribution = False):

    elements_GT = len(ground_truth[0])
    if pick == "Random":
        idx = random.sample(range(0, elements_GT), N_samples)
    elif pick == "Uniform":
        idx = np.round(np.linspace(0, elements_GT - 1, N_samples)).astype(int)
    elif pick == "Distribution":
        idx = np.round(np.linspace(0, elements_GT - 1, N_samples)).astype(int)



    x = ground_truth[0][idx]
    y = ground_truth[1][idx]





    if add_noise:
        y = y + generate_Noise_SampleData(mu,sigma,N_samples)

    samples = Samples_data([x,y],sigma)

    return samples

#GENERATE PLOTS DATA--------------------------------------------------------
def Plot_Truth_and_Samples_Predicted(True_function,Predicted, Samples):


    plt.scatter(Samples[0], Samples[1], color = "m",
                marker = "o", s = 30, label = "Samples")
    # plotting Ground Thruth
    plt.plot(True_function[0], True_function[1], color = "g",
             label = "Gound Truth")

    # plotting the regression line
    plt.plot(True_function[0], Predicted.T, color = "orange",label = "Predicted")


    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="upper right")

    # function to show plot
    plt.show()

# GENERATE PLOTS DATA--------------------------------------------------------
def Plot_Truth_and_Samples_Predicted(True_function, Samples, Predicted = None, error = 0):

    plt.figure()

    if (Predicted is None):
        plt.scatter(Samples[0], Samples[1], color="m",
                    marker="o", s=30, label="Samples")
        # plotting Ground Thruth
        plt.plot(True_function[0], True_function[1], color="g",
                 label="Ground Truth")

        # putting labels
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc="upper right")

        # function to show plot
        plt.show()


    if (Predicted is not None):
        plt.scatter(Samples[0], Samples[1], color="m",
                    marker="o", s=30, label="Samples")
        # plotting Ground Thruth
        plt.plot(True_function[0], True_function[1], color="g",
                 label="Ground Truth")

        # plotting the regression line
        plt.plot(True_function[0], Predicted.T, color="orange", label="Predicted (err = %.5f)"%error)

        # putting labels
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc="best")

        # function to show plot
        plt.show()

# Our 2-dimensional distribution will be over variables X and Y

def Plot_Covariance_Matrix(mu_matrix, cov_matrix, true_values = [-0.3, 0.5], lim_axis = False):

    mu = mu_matrix  # 2 dimensional mean vector
    cov = cov_matrix # 2x2 covariance matrix
    # taking 1000 samples from this 2D Gaussian
    sample = np.random.multivariate_normal(mu, cov, 1000)
    print('mean vector: ', mu)
    print('covariance matrix: ', '\n', cov)
    # plotting this 2 dimensional Gaussian distribution
    # using the seaborn library
    ax = sb.kdeplot(sample[:, 0], sample[:, 1], shade=True, cbar=True)
    if(lim_axis):
        plt.scatter(true_values[0], true_values[1], marker='x', s=20, color="red")

        ax.set(ylim=(-1, 1))
        ax.set(xlim=(-1, 1))
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')

    #plt.show()
    return ax

def ShowKernel(Σ,X,xlim, kernel_func):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    # Plot covariance matrix
    im = ax1.imshow(Σ, cmap=cm.YlGnBu)
    cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
    cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
    ax1.set_title((
        'Exponentiated quadratic \n'
        'example of covariance matrix'))
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('x', fontsize=13)
    ticks = list(range(xlim[0], xlim[1] + 1))
    ax1.set_xticks(np.linspace(0, len(X) - 1, len(ticks)))
    ax1.set_yticks(np.linspace(0, len(X) - 1, len(ticks)))
    ax1.set_xticklabels(ticks)
    ax1.set_yticklabels(ticks)
    ax1.grid(False)

    # Show covariance with X=0
    zero = np.array([[0]])
    Σ0 = kernel_func(X, zero)
    # Make the plots
    ax2.plot(X[:, 0], Σ0[:, 0], label='$k(x,0)$')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('covariance', fontsize=13)
    ax2.set_title((
        'Exponentiated quadratic  covariance\n'
        'between $x$ and $0$'))
    # ax2.set_ylim([0, 1.1])
    ax2.set_xlim(*xlim)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.show()

def PlotSampledFunctions(number_of_functions, X, ys,xlim):
    # Plot the sampled functions
    plt.figure(figsize=(6, 4))
    for i in range(number_of_functions):
        plt.plot(X, ys[i], linestyle='-', marker='o', markersize=3)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title((
        'Sampling'))
    plt.xlim(xlim)
    plt.show()


def Plot_Posterior_GP(Ground_Truth, X2,y2, μ2, Σ2, σ2, domain,X1,y1 ):
    # Plot the postior distribution and some samples
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 6))
    # Plot the distribution of the function (mean, covariance)
    ax1.plot(Ground_Truth[0], Ground_Truth[1], 'b--', label='$f(x)$')
    ax1.fill_between(X2.flat, μ2 - 2 * σ2, μ2 + 2 * σ2, color='red',
                     alpha=0.15, label='$2 \sigma_{2|1}$')
    ax1.plot(X2, μ2, 'r-', lw=2, label='$\mu_{2|1}$')
    ax1.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    ax1.set_xlabel('$x$', fontsize=13)
    ax1.set_ylabel('$y$', fontsize=13)
    ax1.set_title('Distribution of posterior and prior data.')
    ax1.axis([domain[0], domain[1], -3, 3])
    ax1.legend()
    # Plot some samples from this function
    ax2.plot(X2, y2.T, '-')
    ax2.plot(X1, y1, 'ko', linewidth=2, label='$(x_1, y_1)$')

    ax2.set_xlabel('$x$', fontsize=13)
    ax2.set_ylabel('$y$', fontsize=13)
    ax2.set_title('5 different function realizations from posterior')
    ax1.axis([domain[0], domain[1], -3, 3])
    ax2.set_xlim(domain)
    plt.tight_layout()
    plt.show()