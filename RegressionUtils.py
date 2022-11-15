

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sb



class Samples_data:
    def __init__(self, points, noise):
        self.points = points
        self.noise = noise

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

def Plot_Covariance_Matrix(mu_matrix, cov_matrix, true_values = [-0.3, 0.5]):

    mu = mu_matrix  # 2 dimensional mean vector
    cov = cov_matrix # 2x2 covariance matrix
    # taking 1000 samples from this 2D Gaussian
    sample = np.random.multivariate_normal(mu, cov, 1000)
    print('mean vector: ', mu)
    print('covariance matrix: ', '\n', cov)
    # plotting this 2 dimensional Gaussian distribution
    # using the seaborn library
    ax = sb.kdeplot(sample[:, 0], sample[:, 1], shade=True, cbar=True)
    ax.set(ylim=(-1, 1))
    ax.set(xlim=(-1, 1))
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')
    plt.scatter(true_values[0], true_values[1], marker='x', s=20, color = "red")

    #plt.show()
    return ax

