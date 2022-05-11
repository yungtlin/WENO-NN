# Data class to ensure each data entries are unique in term of Minkowski norm

### Libraries ###
import numpy as np
from scipy import spatial

### UData ###
class UData:
    ### Init ###
    def __init__(self, X, r_max, p=2):
        self.r_max = r_max
        self.p = p 
        self.tree = spatial.cKDTree(X)
        self.update_Data()


    # get # of unique data points
    def get_nData(self):
        pass

    # update data and remove 
    def update_Data(self):
        n_pairs = list(self.tree.query_pairs(self.r_max, p=self.p))

    # add data 
    def add(self, X_new):
        pass


    #
    #def 




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

    X = gen_X_sc(x, r, func, args)

    


