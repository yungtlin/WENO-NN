# Trains WENO-NN by Stevens and Colonius 
### Libraries ###
#import tensorflow as tf
import numpy as np

### Data ###
def get_dataX(path):
    with open(path, "rb") as f:
        print("Loading data: %s"%path)
        dataX = np.load(f)
    X = dataX[:, :-1]
    y = dataX[:, -1]
    return X, y

### WENO ###
def WENO5_getC(f_tilde):
    # check dimension (r=3)
    r = 3
    if f_tilde.shape[1] != (2*r-1):
        raise ValueError("Data Dimension Must Be 5")

    omega = WENO5_omega(f_tilde)
    C = WENO5_C(omega)

    return C


def WENO5_omega(f, epsilon=1e-6):
    nX = f.shape[0]

    # beta
    beta = np.zeros((nX, 3))    
    beta[:, 0] = 13/12*(f[:,0] - 2*f[:,1] + f[:,2])**2\
        + 1/4*(f[:,0] - 4*f[:,1] + 3*f[:,2])**2
    beta[:, 1] = 13/12*(f[:,1] - 2*f[:,2] + f[:,3])**2\
        + 1/4*(f[:,1] - f[:,3])**2
    beta[:, 2] = 13/12*(f[:,2] - 2*f[:,3] + f[:,4])**2\
        + 1/4*(3*f[:,2] - 4*f[:,3] + f[:,4])**2

    # sigma
    gamma = np.array([1/10, 3/5, 3/10]).reshape((1, -1))
    sigma = gamma/(epsilon + beta)**2

    # omega
    sigma_sum = np.sum(sigma, axis=1).reshape((-1, 1))
    omega = sigma/sigma_sum

    return omega

def WENO5_C(omega):
    nX = omega.shape[0]

    C = np.zeros((nX, 5))

    C[:, 0] = 1/3*omega[:,0]
    C[:, 1] = -7/6*omega[:,0] - 1/6*omega[:,1]
    C[:, 2] = 11/6*omega[:,0] + 5/6*omega[:,1] + 1/3*omega[:,2]
    C[:, 3] = 1/3*omega[:,1] + 5/6*omega[:,2]
    C[:, 4] = -1/6*omega[:,2]

    return C


if __name__ == "__main__":
    folder = "training_data/"
    file = "data_SC_0.npy"
    X, y = get_dataX(folder+file)
    
    c_tilde = WENO5_getC(X)

    import matplotlib.pyplot as plt
    f_weno = np.sum(c_tilde*X, axis=1)
    plt.plot(y, f_weno, "ob", markersize=0.2, label="WENO5-JS")
    plt.xlabel("exact")
    plt.grid()
    plt.show()

