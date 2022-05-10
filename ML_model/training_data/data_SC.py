# Generates the data from the method of Stevens and Colonius
# Check WENO-NN report for technical details

### Libraries ###
import numpy as np
from scipy import spatial
import gen_data


### Functions ###
# u* = (u - u_min)/(u_max - u_min)
def normalize():
    pass


### SC ####
def data_SC(path):
    r = 3 # WENO5 based

    x = np.linspace(-1, 1, 101)

    func = gen_data.f_sin
    omega = np.pi
    delta = 0
    args = (omega, delta)
    
    x, y = np.mgrid[0:5, 2:8]
    #X, y = gen_data.gen_Xy(x, r, func, args)
    #X = X[:, :-1]
    X = np.array([x.ravel(), y.ravel()]).T
    tree = spatial.KDTree(X)
    #gen_data.check_f(x, r, func, args)

    radius = 1
    

# testing field
if __name__ == "__main__":
    # saving path
    folder = ""
    file = "data_SC.npy"
    path = folder + file

    # executing data generation process
    data_SC(path)


