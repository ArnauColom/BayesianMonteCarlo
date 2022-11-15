
import RegressionUtils as Utils
import RegressionModels
import numpy as np
import matplotlib.pyplot as plt
import math

def Compute_Regression(True_function,Samples, Regression = "Bayesian_LR"):

    if(Regression == "LBFM"):#Linear Basis Function Models
        M = 6 #number of Parameters
        if(len(Samples.points[0])<M):
            raise ValueError('Work with M>N')
        #Generate Predicted function
        #model = RegressionModels.Polynomial(M, b= True)
        model = RegressionModels.Gaussian(M,Samples.points[0], b=True, gamma = 1, auto_mu= True,auto_sigma= False)
        w, error = Utils.Compute_Weight_Analytical_BF(Samples.points, model)
        Phi_cont = model.Compute_DesignMatrix(True_function[0])

        y_pred = w.T @ Phi_cont.T

        #Plot GT and Sampels
        Utils.Plot_Truth_and_Samples_Predicted(True_function, Samples.points,y_pred, error)
    elif(Regression == "Bayesian_LR"):
        CASE = 1;

        if(CASE == 1):
            ############################# First raw Implemetnation####################### BISHOP Patter recognition
            #Number of Basis function
            M = 2
            if (len(Samples.points[0]) < M):
                raise ValueError('Work with M>N')
            #Polynomial linear basis function, with bias
            model = RegressionModels.Polynomial(M, b = True);
            # Noise used in Gaussian added to the groun truth
            noise_std = Samples.noise
            sigma_n_2 = pow(1 / noise_std, 2);
            #Number of samples
            for i in range(5):
                #Number of samples to evaluate the weights
                num_samples = 10*i;
                #Covariance matrix for the weights (this particular case 2x2), bivariate-Gaussian
                mean_matrix_w = np.array([0.0,0.0]) #Mean
                cov_matrix_w = np.array([[0.2, 0.0], [0.0, 0.2]]) #Covarance matrix (independent)
                inv_cov = np.linalg.inv(cov_matrix_w) #Inverse covariance
                #Start iterative process-----------
                #Utils.Plot_Covariance_Matrix(mean_matrix_w,cov_matrix_w)
                Phi_cont = []
                if(num_samples == 0):
                    Phi_cont = 0
                    A = inv_cov
                else:
                    #Compute Design matrix
                    Phi_cont = model.Compute_DesignMatrix(Samples.points[0][0:num_samples])
                    #Create new covariance matrix for weights
                    A = sigma_n_2*Phi_cont.T@Phi_cont + inv_cov
                #Inverse of A
                A_inv = np.linalg.inv(A)
                #Compute new w mean
                if(num_samples == 0):
                    w_hat = mean_matrix_w
                else:
                    w_hat = A_inv @ (sigma_n_2 * Phi_cont.T @ Samples.points[1][0:num_samples] + mean_matrix_w @ inv_cov)
                #Sample 5 different weights
                W = np.random.multivariate_normal(w_hat, A_inv, 5)
                #plt.figure()
                plt.figure()
                fig, axs = plt.subplots(2, 1)
                for i in range(5):
                    w = np.expand_dims(W[i], axis=0)
                    x = np.expand_dims(True_function[0], axis=0)
                    Phi_ = model.Compute_DesignMatrix(True_function[0])
                    samples =w@Phi_.T
                    axs[0].plot(x.T,samples.T)

                axs[0].scatter(Samples.points[0],Samples.points[1])
                axs[0].set_xlim([-1,1])
                axs[0].set_ylim([-1,1])
                #plt.show()
                #plt.figure()
                fig = Utils.Plot_Covariance_Matrix(w_hat,A_inv)
                axs[1] = fig
                plt.show()
        if(CASE == 2):
            ############################# Ricardo Implementation#######################
            t= 0






def main():
    ##Clear environment
    plt.close('all')
    #GENERATE DATA---------------------------------------------------

    #Ground Truth Function
    True_function = Utils.generate_GroundTruth_Function(option = 3, f_domain = [-1,1])#f_domain = [-math.pi,math.pi]
    #Generate Samples
    # Number of samples
    N_samples = 50
    Samples = Utils.generate_Samples(True_function, N_samples,
                                     pick = "Random", add_noise=True, sigma = 0.2)
    if True:
        Utils.Plot_Truth_and_Samples_Predicted(True_function,Samples.points)
    #LEARNING ----------------------------
    
    Regression = "Bayesian_LR" #"LBFM" Bayesian_LR   Linear Basis Function Models
    # Calculate design matrix Phi
    Compute_Regression(True_function,Samples, Regression)
    
  
if __name__ == "__main__":
    main()


