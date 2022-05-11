# Generates the data from the method of Stevens and Colonius
# Check WENO-NN report for technical details

### Libraries ###
import numpy as np
from scipy import spatial
import gen_data


### Functions ###
# u* = (u - u_min)/(u_max - u_min)
# U[:, :-1] = X[:, :-1]
# U[:, -1] = y
def normalize_SC(X, y):
    U = np.array(X)
    U[:, -1] = y.reshape(-1)
    u_max = np.max(U, axis=1)
    u_min = np.min(U, axis=1) 
    du = u_max - u_min

    ## remove constant extrapolation ##
    tolerance = 1e-10
    idx_0 = np.argwhere(du < tolerance)
    U = np.delete(U, idx_0, 0)

    # update and check
    u_max = np.max(U, axis=1).reshape((-1, 1))
    u_min = np.min(U, axis=1).reshape((-1, 1))
    du = u_max - u_min
    idx_0 = np.argwhere(du < tolerance)
    if len(idx_0) != 0:
        raise ValueError("Constants Are Still Existed After Removal")

    U_sc = (U - u_min)/(u_max - u_min)

    return U_sc

def gen_X_sc(x, r, func, args):
    X, y = gen_data.gen_Xy(x, r, func, args)
    X_sc = normalize_SC(X, y)
    return X_sc

### SC ####
def data_SC(path):
    r = 3 # WENO5 based

    x = np.linspace(-1, 1, 101)

    func = gen_data.f_sin
    omega = np.pi
    delta = 0
    args = (omega, delta)
    
    X = gen_X_sc(x, r, func, args)


# testing field
if __name__ == "__main__":
    # saving path
    folder = ""
    file = "data_SC.npy"
    path = folder + file

    # executing data generation process
    data_SC(path)


