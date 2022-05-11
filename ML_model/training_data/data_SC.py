# Generates the data from the method of Stevens and Colonius
# Check WENO-NN report for technical details

### Libraries ###
import numpy as np
import gen_data
from UData import UData

### Data Process ###
def gen_X_sc(x, r, func, args):
    X, y = gen_data.gen_Xy(x, r, func, args)
    X_sc = normalize_SC(X, y)
    return X_sc

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

### Custom Candidate ###
# signed sawtooth
def sign_sawtooth(x, r, args):
    (sign, args_sw) = args
    F_b, f_h = gen_data.f_sawtooth(x, r, args_sw)
    F_b = sign*F_b
    f_h = sign*f_h
    return F_b, f_h

# signed step
def sign_step(x, r, args):
    (sign, args_stp) = args
    F_b, f_h = gen_data.f_step(x, r, args_stp)
    F_b = sign*F_b
    f_h = sign*f_h
    return F_b, f_h

# signed tanh
def sign_tanh(x, r, args):
    (sign, args_tanh) = args
    F_b, f_h = gen_data.f_tanh(x, r, args_tanh)
    F_b = sign*F_b
    f_h = sign*f_h
    return F_b, f_h

#
def mix_sin_stp(x, r, args):
    args_sin, args_sstp = args
    F_bsin, f_hsin = gen_data.f_sin(x, r, args_sin)
    F_bstp, f_hstp = sign_step(x, r, args_sstp)

    F_b = F_bsin + F_bstp
    f_h = f_hsin + f_hstp

    return F_b, f_h


### SC ####
def data_SC(path):
    r = 3 # WENO5 based
    r_min = 1e-3 # data minimum distance

    dataU = init_data_U(r, r_min)
    #data_SC_sin(dataU, r) # ~17,000 # sin
    #data_SC_sw(dataU, r) # ~9,000 # sawtooth (26,000)
    #data_SC_step(dataU, r) # ~1,000 # step (27,000)
    #data_SC_tanh(dataU, r) # ~8,000 # tanh (34,000)
    data_SC_sin_stp(dataU, r) # sin + step

# randomly add sinusoidal function initialize data_U
def init_data_U(r, r_min):
    func = gen_data.f_sin
    n_data = 100 # data per shift

    x = np.linspace(0, 1, n_data+2*r)
    exp_max = 3
    exp_min = -1
    exp_k = (exp_max-exp_min)*np.random.random_sample() + exp_min
    k = np.exp(exp_k)
    omega = 2*np.pi*k
    delta = np.random.random_sample()/k
    args = (omega, delta)

    X = gen_X_sc(x, r, func, args)
    dataU = UData(X, r_min)

    return dataU

## Sinusoidal ##
def data_SC_sin(dataU, r):
    func = gen_data.f_sin
    
    n_k = 50 # total for sin
    n_shift = 10 # shift per k
    n_data = 50 # data per shift
    x = np.linspace(0, 1, n_data+2*r)

    for k_cnt in range(n_k):
        exp_max = 3
        exp_min = -1
        exp_k = (exp_max-exp_min)*np.random.random_sample() + exp_min
        k = np.exp(exp_k)
        omega = 2*np.pi*k
        print("Sinusoidal (k:%.5f):"%k)

        for shift in range(n_shift):
            delta = np.random.random_sample()/k
            args = (omega, delta)
            
            X = gen_X_sc(x, r, func, args)
            dataU.add(X)


## Sawtooth ##
def data_SC_sw(dataU, r):
    func = sign_sawtooth

    n_k = 20 # total of wave numbers
    n_shift = 50 # shift per k
    n_data = 5 # data per shift (fixed for optimal sampling)
    x = np.linspace(-1, 0, n_data+2*r)
    dx = np.mean(x[1:] - x[:-1])/2
 
    # flipping sign
    for sign in [1, -1]:
        for k_cnt in range(n_k):
            exp_max = 1.4
            exp_min = -1.4
            exp_k = (exp_max-exp_min)*np.random.random_sample() + exp_min 
            k = np.exp(exp_k)

            print("Sawtooth (sign:%i, k:%.5f):"%(sign, k))
            for shift in range(n_shift):
                delta = 2*dx*np.random.random_sample() - dx
                args_sw = (k, delta)
                args = (sign, args_sw)

                X = gen_X_sc(x, r, func, args)
                dataU.add(X)

## Step ##
def data_SC_step(dataU, r):
    func = sign_step

    n_shift = 200 # shift per h
    n_data = 5 # data per shift (fixed for optimal sampling)
    x = np.linspace(-0.5, 0.5, n_data+2*r)
    dx = np.mean(x[1:] - x[:-1])/2

    # flipping sign
    for sign in [1, -1]:
        print("Step (sign:%i):"%(sign))
        for shift in range(n_shift):
            delta = 2*dx*np.random.random_sample() - dx
            args_stp = (delta)
            args = (sign, args_stp)

            X = gen_X_sc(x, r, func, args)
            dataU.add(X)

## Hyperbolic Tangent ##
def data_SC_tanh(dataU, r):
    func = sign_tanh

    n_k = 50 # total of scale number
    n_shift = 20 # shift per k
    n_data = 5 # data per shift (fixed for optimal sampling)
    x_base = np.linspace(-1, 1, n_data+2*r)
    dx = np.mean(x_base[1:] - x_base[:-1])
 
    # flipping sign
    for sign in [1, -1]:
        for k_cnt in range(n_k):
            exp_max = 0
            exp_min = 4
            exp_k = (exp_max-exp_min)*np.random.random_sample() + exp_min 
            k = np.exp(exp_k)

            print("Hyperbolic Tangent (sign:%i, k:%.5f):"%(sign, k))
            for shift in range(n_shift):
                delta = 2*dx*np.random.random_sample() - dx
                args_tanh = (k)
                args = (sign, args_tanh)
                x = x_base - delta
            
                X = gen_X_sc(x, r, func, args)
                dataU.add(X)

# Sinusoidal + Step
def data_SC_sin_stp(dataU, r):
    func = mix_sin_stp

    n_data = 5 # data per shift (fixed for optimal sampling)
    x = np.linspace(-0.5, 0.5, n_data+2*r)

    sign = 1

    omega = np.pi
    delta = 0

    args_sin = (omega, delta)
    args_sstp = (sign, (0))
    args = (args_sin, args_sstp)

    gen_data.check_f(x, r, func, args)


# testing field
if __name__ == "__main__":
    # saving path
    folder = ""
    file = "data_SC.npy"
    path = folder + file

    # executing data generation process
    data_SC(path)
