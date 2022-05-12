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

# sin + step
# f = alph*sin + step
def mix_sin_stp(x, r, args):
    alpha, args_sin, args_sstp = args
    F_bsin, f_hsin = gen_data.f_sin(x, r, args_sin)
    F_bstp, f_hstp = sign_step(x, r, args_sstp)

    F_b = alpha*F_bsin + F_bstp
    f_h = alpha*f_hsin + f_hstp

    return F_b, f_h

# sin + saw
# f = alph*sin + saw
def mix_sin_sw(x, r, args):
    alpha, args_sin, args_ssw = args
    F_bsin, f_hsin = gen_data.f_sin(x, r, args_sin)
    F_bsw, f_hsw = sign_sawtooth(x, r, args_ssw)

    F_b = alpha*F_bsin + F_bsw
    f_h = alpha*f_hsin + f_hsw

    return F_b, f_h

# sin + tanh
# f = alph*sin + tanh
def mix_sin_tanh(x, r, args):
    alpha, args_sin, args_stanh = args
    F_bsin, f_hsin = gen_data.f_sin(x, r, args_sin)
    F_btanh, f_htanh = sign_tanh(x, r, args_stanh)

    F_b = alpha*F_bsin + F_btanh
    f_h = alpha*f_hsin + f_htanh

    return F_b, f_h


### SC ####
def data_SC(path):
    r = 3 # WENO5 based
    r_min = 3e-3 # data minimum distance

    dataU = init_data_U(r, r_min)

    data_SC_sin(dataU, r)       # ~11,000 sin      (11,000)
    data_SC_sw(dataU, r)        # ~8,000  sawtooth (19,000)
    data_SC_step(dataU, r)      # ~1,000  step     (20,000)
    data_SC_tanh(dataU, r)      # ~5,000  tanh     (25,000)
    data_SC_sin_stp(dataU, r)   # ~10,000  sin+stp (35,000)
    data_SC_sin_sw(dataU, r)    # ~12,000 sin+saw  (47,000)
    data_SC_sin_tanh(dataU, r)  # ~34,000 sin+tanh (81,000)

    X_all = dataU.X
    print("Data generation completed! (total: %i)\n"%(X_all.shape[0]))

    with open(path, "wb") as f:
        print("Saving data: %s"%path)
        np.save(f, X_all)
    


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

        X_pack = np.zeros((0, 6))
        print("Sinusoidal (k:%.5f):"%k)
        for shift in range(n_shift):
            delta = np.random.random_sample()/k
            args = (omega, delta)
            
            X = gen_X_sc(x, r, func, args)
            X_pack = np.append(X_pack, X, axis=0)
        dataU.add(X_pack)

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

            X_pack = np.zeros((0, 6))
            print("Sawtooth (sign:%i, k:%.5f):"%(sign, k))
            for shift in range(n_shift):
                delta = 2*dx*np.random.random_sample() - dx
                args_sw = (k, delta)
                args = (sign, args_sw)

                X = gen_X_sc(x, r, func, args)
                X_pack = np.append(X_pack, X, axis=0)
            dataU.add(X_pack)

## Step ##
def data_SC_step(dataU, r):
    func = sign_step

    n_shift = 200 # shift per h
    n_data = 5 # data per shift (fixed for optimal sampling)
    x = np.linspace(-0.5, 0.5, n_data+2*r)
    dx = np.mean(x[1:] - x[:-1])/2

    # flipping sign
    for sign in [1, -1]:

        X_pack = np.zeros((0, 6))
        print("Step (sign:%i):"%(sign))
        for shift in range(n_shift):
            delta = 2*dx*np.random.random_sample() - dx
            args_stp = (delta)
            args = (sign, args_stp)

            X = gen_X_sc(x, r, func, args)
            X_pack = np.append(X_pack, X, axis=0)
        dataU.add(X_pack)

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

            X_pack = np.zeros((0, 6))
            print("Hyperbolic Tangent (sign:%i, k:%.5f):"%(sign, k))
            for shift in range(n_shift):
                delta = 2*dx*np.random.random_sample() - dx
                args_tanh = (k)
                args = (sign, args_tanh)
                x = x_base - delta
            
                X = gen_X_sc(x, r, func, args)
                X_pack = np.append(X_pack, X, axis=0)
            dataU.add(X_pack)

# Sinusoidal + Step
# f = alph*sin + step
def data_SC_sin_stp(dataU, r):
    func = mix_sin_stp
    L = 1
    n_data = 5 # data per shift (fixed for optimal sampling)
    x = np.linspace(-0.5*L, 0.5*L, n_data+2*r)
    dx = np.mean(x[1:] - x[:-1])/2

    n_alpha = 10
    n_k = 5
    n_delta = 5
    n_shift = 5

    for count_alpha in range(n_alpha):
        alp_max = -1
        alp_min = -5
        exp_alp = (alp_max-alp_min)*np.random.random_sample() + alp_min 
        alpha = np.exp(exp_alp)
        
        for count_k in range(n_k):
            k_max = 1.5
            k_min = -2
            exp_k = (k_max-k_min)*np.random.random_sample() + k_min 
            k = np.exp(exp_k)

            X_pack = np.zeros((0, 6))
            print("sin + step (alpha:%.5f, k:%.5f):"%(alpha, k))
            for count_delta in range(n_delta):
                omega = 2*np.pi*k
                delta = L/k*np.random.random_sample()
                args_sin = (omega, delta)

                for sign in [1, -1]:            
                    for count_shift in range(n_shift):
                        shift = 2*dx*np.random.random_sample() - dx
                        args_sstp = (sign, (shift))

                        args = (alpha, args_sin, args_sstp)
                        X = gen_X_sc(x, r, func, args)
                        X_pack = np.append(X_pack, X, axis=0)
            dataU.add(X_pack)

# Sinusoidal + Step
# f = alph*sin + step
def data_SC_sin_sw(dataU, r):
    func = mix_sin_sw
    L = 1
    n_data = 5
    x = np.linspace(-1*L, 0*L, n_data+2*r)
    dx = np.mean(x[1:] - x[:-1])/2

    n_alpha = 10
    n_omega = 5
    n_delta = 6
    n_k = 2
    n_shift = 2

    for count_alpha in range(n_alpha):
        alp_max = 1
        alp_min = -3
        exp_alp = (alp_max-alp_min)*np.random.random_sample() + alp_min 
        alpha = np.exp(exp_alp)
        
        for count_omega in range(n_omega):
            k_sin_max = 1
            k_sin_min = -1
            exp_k_sin = (k_sin_max-k_sin_min)*np.random.random_sample() + k_sin_min 
            k_sin = np.exp(exp_k_sin)
            
            X_pack = np.zeros((0, 6))
            print("sin + sawtooth (alpha:%.5f, k:%.5f):"%(alpha, k_sin))
            for count_delta in range(n_delta):
                omega = 2*np.pi*k_sin
                delta = L/k_sin*np.random.random_sample()
                args_sin = (omega, delta)
    
                for sign in [1, -1]:
                    for count_k in range(n_k):
                        k_sw_max = 0.7
                        k_sw_min = -2
                        exp_k_sw = (k_sw_max-k_sw_min)*np.random.random_sample() + k_sw_min 
                        k_sw = np.exp(exp_k_sw)

                        for count_shift in range(n_shift):
                            shift = 2*dx*np.random.random_sample() - dx
                            args_ssw = (sign, (k_sw, shift)) 
                
                            args = (alpha, args_sin, args_ssw)
                            X = gen_X_sc(x, r, func, args)
                            X_pack = np.append(X_pack, X, axis=0)
            dataU.add(X_pack)

# Sinusoidal + Hyperbolic Tangent
# f = alph*sin + tanh
def data_SC_sin_tanh(dataU, r):
    func = mix_sin_tanh

    L = 1
    n_data = 5
    x_base = np.linspace(-0.5*L, 0.5*L, n_data+2*r)
    dx = np.mean(x_base[1:] - x_base[:-1])/2

    n_alpha = 30
    n_omega = 5
    n_delta = 6
    n_k = 2
    n_shift = 2

    for count_alpha in range(n_alpha):
        alp_max = 1
        alp_min = -3
        exp_alp = (alp_max-alp_min)*np.random.random_sample() + alp_min 
        alpha = np.exp(exp_alp)

        for count_omega in range(n_omega):
            k_sin_max = 1.6
            k_sin_min = -1
            exp_k_sin = (k_sin_max-k_sin_min)*np.random.random_sample() + k_sin_min 
            k_sin = np.exp(exp_k_sin)
            
            X_pack = np.zeros((0, 6))
            print("sin + tanh (alpha:%.5f, k:%.5f):"%(alpha, k_sin))
            for count_delta in range(n_delta):
                omega = 2*np.pi*k_sin
                delta = L/k_sin*np.random.random_sample()
                args_sin = (omega, delta)

                for sign in [1, -1]:
                    for count_k in range(n_k):
                        k_tanh_max = 4
                        k_tanh_min = 0
                        exp_k_tanh = (k_tanh_max-k_tanh_min)*np.random.random_sample() + k_tanh_min 
                        k_tanh = np.exp(exp_k_tanh)

                        for count_shift in range(n_shift):
                            shift = 2*dx*np.random.random_sample() - dx

                            x = x_base + shift
                            args_stanh = (sign, (k_tanh))
                            
                            args = (alpha, args_sin, args_stanh)

                            X = gen_X_sc(x, r, func, args)
                            X_pack = np.append(X_pack, X, axis=0)
            dataU.add(X_pack)

# testing field
if __name__ == "__main__":
    # saving path
    folder = ""
    file = "data_SC_0.npy"
    path = folder + file

    # executing data generation process
    data_SC(path)

    # test by loading data
    #with open(path, "rb") as f:
    #    print("Loading data: %s"%path)
    #    X = np.load(f)
    #print(X.shape)