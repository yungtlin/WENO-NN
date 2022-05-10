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
    x_f = x_b[r: -(r-1)] # discards unusable flux points near boundaries
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
    stencil = 2*r - 1
    ndata = f_h.shape[0]
    
    data_f_b = np.zeros((ndata, stencil))
    for i in range(stencil):
        data_f_b[:, i] = f_b[i:i+ndata]

    data_h = h[(r-1):-(r-1)].reshape((-1, 1))

    data_X = np.append(data_f_b, data_h, axis=1)
    data_y = f_h.reshape((-1, 1))

    return data_X, data_y

# checkinig validity of a candidate function through plotting
def check_f(x, r, func, args, n_exact=1001):
    x_b, x_f = average_location(x, r)
    F_b, f_h = func(x_b, x_f, args)
    f_b, h = f_int2b(F_b, x_b)

    # import matplot
    import matplotlib.pyplot as plt
    
    # exact solution
    x_exact = np.linspace(x_b[0], x_b[-1], n_exact)
    f_exact = func(x_b, x_exact, args)[1]
    plt.plot(x_exact, f_exact, "--k", label="exact")


    # cell average
    x_fb = x[1:-1]
    plt.plot(x_fb, f_b, "ob", label="cell average")

    # plotting
    plt.title("Check f", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

### Candidate functions ###
# F_b: F at x_b (F = \int f)
# f_h: \hat{f} = flux on edge (x_{j+1/2})

# f = sin(omega*(x - delta))
def f_sin(x_b, x_f, args):
    (omega, delta) = args
    f_h = np.sin(omega*(x_f - delta))
    F_b = -np.cos(omega*(x_b - delta))/omega
    
    return F_b, f_h

# f = H(x - delta) (Heaviside step)
def f_step(x_b, x_f, args):
    delta = args
    f_h = np.where((x_f-delta) < 0, 0, 1)
    F_b = np.where((x_b-delta) < 0, 0, (x_b-delta))

    return F_b, f_h

# f = (k*(x_f-delta))%1 -1/2 symmetry sawtooth \in [-1, 1] 
def f_sawtooth(x_b, x_f, args):
    k, delta = args
    f_h = (2*k*(x_f-delta+0.5))%2 - 1
    
    dx = (2*k*(x_b-delta+0.5))%2 - 1
    F_b = 1/4*dx**2

    return F_b, f_h

# f = tanh(k*x)
def f_tanh(x_b, x_f, args):
    k = args
    f_h = np.tanh(k*x_f)
    F_b = np.log(np.cosh(k*x_b))/k

    return F_b, f_h

# f = (x - delta)**n
def f_poly(x_b, x_f, args):
    n, delta = args
    f_h = (x_f - delta)**n
    F_b = ((x_b - delta)**(n+1))/(n+1)

    return F_b, f_h

# testing field
if __name__ == "__main__":
    r = 3 # optimal accuracy/stencil: 2r-1

    L_l = -1 # left boundary
    L_r = 1 # right boundary
    nx = 18
    x = np.linspace(L_l, L_r, nx)

    func = f_poly
    n = 5
    delta = 0
    args = (n, delta)

    x_b, x_f = average_location(x, r)
    F_b, f_h = func(x_b, x_f, args)
    check_f(x, r, func, args)
    
    X, y = gen_Xy(x, r, func, args)

