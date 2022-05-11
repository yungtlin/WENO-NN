# Data class to ensure each data entries are unique in term of Minkowski norm

### Libraries ###
import numpy as np
from scipy import spatial

### UData ###
class UData:
    ### Init ###
    def __init__(self, X, r_min, p=2):
        self.r_min = r_min
        self.p = p

        self.X = X
        self.check_Data()

    # update data and recursively remove data points that are too close to each other
    def check_Data(self):
        nX_0 = self.X.shape[0]
        self.X = unique_data(self.X, self.r_min, self.p)
        # update
        self.tree = spatial.cKDTree(self.X) 
        nX_1 = self.X.shape[0]
        print("%i checked, %i removed"%(nX_0, nX_0-nX_1))

    # add data 
    def add(self, X_new):
        nX_0 = X_new.shape[0]
        # makes X_new unique
        X_uni = unique_data(X_new, self.r_min, self.p)
        # removes unique X_new that is too close to existed points
        # return distance of nearst neighbor in X
        dist, idx = self.tree.query(X_uni, p=self.p) 
        idx_rm = np.argwhere(dist < self.r_min)
        X_uni = np.delete(X_uni, idx_rm, axis=0)

        nX_new = X_uni.shape[0]
        if nX_new > 0: # merge when X_new has at least one unique value
            self.X = np.append(self.X, X_uni, axis=0)
            self.tree = spatial.cKDTree(self.X)
        nX = self.X.shape[0]
        print("%i recieved, %i added, %i in total"%(nX_0, nX_new, nX))

# Makes input X unique data set
# Iteratively removes data point
def unique_data(X0, r_min, p):
    X = np.array(X0)
    tree = spatial.cKDTree(X)
    pair_list = list(tree.query_pairs(r_min, p=p))    

    while len(pair_list) != 0:
        pair_0 = pair_list[0]
        idx_rm = pair_0[1]
        X = np.delete(X, idx_rm, axis=0)
        tree = spatial.cKDTree(X)
        pair_list = list(tree.query_pairs(r_min, p=p))    
    return X

# testing field
if __name__ == "__main__":
    import gen_data
    from data_SC import gen_X_sc

    r = 3 # WENO5 based

    x = np.linspace(-1, 1, 101)
    func = gen_data.f_sin
    omega = np.pi
    delta = 0
    args = (omega, delta)
    X1 = gen_X_sc(x, r, func, args)

    r_min = 0.001 # 
    uData = UData(X1, r_min)

    x = np.linspace(-1, 1, 101)
    func = gen_data.f_sin
    omega = np.pi*1.02
    delta = 0
    args = (omega, delta)
    X2 = gen_X_sc(x, r, func, args)

    uData.add(X2)

    uData.check_Data()
