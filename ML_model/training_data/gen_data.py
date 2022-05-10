# Generates data with flux and cell averaged
# Check WENO-NN report for technical details

### Libraries ###
import numpy as np  

# x: cell average location x_{j-2:j+2}
# x_b: cell boundary location 
# x_f: flux location x_{j+1/2} 

# return x_b and x_f
def average_location(x, r):
    x_b = (x[:-1] + x[1:])/2 # mid points
    x_f = x_b[r+1: -r] # discards unusable flux points near boundaries
    return x_b, x_f

### Data Xy generation ###
# f_b: \bar{f} = cell average (x_{j})
# h = h_j
# f_h: \hat{f} = flux on edge (x_{j+1/2})

# X = (f_b, h), y = f_h
def gen_Xy(x, r, func, args):
    x_b, x_f = average_location(x, r)
    F_b, f_h = func(x_b, x_f, args)
    f_b, h = f_int2b(F_b, x_b)
    data_X, data_y = packto2D(f_b, h, f_h, r)

    return data_X, data_y

# convert F_b to f_b
def f_int2b(F_b, x_b):
    h = x_b[1:] - x_b[:-1]
    f_b = (F_b[1:] - F_b[:-1])/h

    return f_b, h

# pack sequntial data into a 2d data package
def packto2D(f_b, h, f_h, r):
    stencil = 2*r + 1
    ndata = f_h.shape[0]
    
    data_f_b = np.zeros((ndata, stencil))
    for i in range(stencil):
        data_f_b[:, i] = f_b[i:i+ndata]

    data_h = h[r:-r].reshape((-1, 1))

    data_X = np.append(data_f_b, data_h, axis=1)
    data_y = f_h.reshape((-1, 1))

    return data_X, data_y

### Candidate functions ###
# F_b: F at x_b (F = \int f)
# f_h: \hat{f} = flux on edge (x_{j+1/2})

# f = sin(omega*(x - phi))
def f_sin(x_b, x_f, args):
    (omega, phi) = args
    f_h = np.sin(omega*(x_f - phi))
    F_b = -np.cos(omega*(x_b - phi))/omega
    
    return F_b, f_h

# f = H(x - delta) (Heaviside step)
def f_step(x_b, x_f, args):
    delta = args
    f_h = 

    return x_b, x_f




if __name__ == "__main__":
    r = 2 # optimal accuracy/stencil: 2r+1

    L_l = 0 # left boundary
    L_r = 1 # right boundary
    nx = 101
    x = np.linspace(L_l, L_r, nx)

    func = f_step
    omega = 4*np.pi
    phi = 1
    args = (omega, phi)


    X, y = gen_Xy(x, r, func, args)
    





