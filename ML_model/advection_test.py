# Linear advection test for evaluating model accuracy


### Libraries ###
import numpy as np
import matplotlib.pyplot as plt 

from train_model import *
from view_scatter import *

def exact_square(x, t, c, L):
    Phi = (x - c*t)%L
    
    f = np.zeros(x.shape)
    for i, phi in enumerate(Phi):
        # square wave
        if phi >= 0*L and phi < 1/2*L:
            f[i] = 1

    return f

def exact_sin(x, t, c, L):
    Phi = (x - c*t)%L

    k = 1
    omega = 2*np.pi*k
    f = np.sin(omega*Phi)
    
    return f


def exact_mixed(x, t, c, L):
    Phi = (x - c*t)%L

    f = np.zeros(x.shape)
    for i, phi in enumerate(Phi):
        # sinusoidal wave
        if phi >= 0*L and phi <= 1/4*L:
            k = 4
            omega = 2*np.pi*k
            f[i] = 1/2-1/2*np.cos(omega*phi)

        # square wave
        elif phi >= 2/4*L and phi < 3/4*L:
            f[i] = 1

    return f

# periodic boundary
def adv_get_f_bar(u, c):
    nx = u.shape[0]

    f = -c*u

    f_bar = np.zeros((nx, 5))

    f_bar[:, 0] = np.roll(f, 2)
    f_bar[:, 1] = np.roll(f, 1)
    f_bar[:, 2] = f
    f_bar[:, 3] = np.roll(f, -1)
    f_bar[:, 4] = np.roll(f, -2)

    return f_bar


def adv_RK3(adv_func, u0, x, c, CFL, T):
    dx = np.mean(x[1:]-x[:-1])
    dt = CFL*dx/c

    u = np.array(u0)

    t = 0
    iteration = 0
    while(t < T):
        if dt > T-t:
            dt = T-t

        dfdx = adv_dfdx(adv_func, u, c, dx)
        u1 = u + dfdx*dt

        dfdx = adv_dfdx(adv_func, u1, c, dx)
        u2 = 3/4*u + 1/4*u1 + 1/4*dfdx*dt

        dfdx = adv_dfdx(adv_func, u2, c, dx)
        u = 1/3*u + 2/3*u2 + 2/3*dfdx*dt

        iteration += 1
        t += dt
        #print("Iteration: %i, t: %.3e"%(iteration, t))

    return u

def adv_dfdx(adv_func, u, c, dx):
    f_bar = adv_get_f_bar(u, c)
    f_weno = adv_func(f_bar)

    f_r = f_weno
    f_l = np.roll(f_weno, 1)

    dfdx = (f_r - f_l)/dx

    return dfdx

def adv_RK3_NN(model_info, u0, x, c, CFL, T, u_max=1e10):
    dx = np.mean(x[1:]-x[:-1])
    dt = CFL*dx/c

    u = np.array(u0)

    t = 0
    iteration = 0
    while(t < T):
        if dt > T-t:
            dt = T-t

        dfdx = adv_dfdx_NN(model_info, u, c, dx)
        u1 = u + dfdx*dt

        dfdx = adv_dfdx_NN(model_info, u1, c, dx)
        u2 = 3/4*u + 1/4*u1 + 1/4*dfdx*dt

        dfdx = adv_dfdx_NN(model_info, u2, c, dx)
        u = 1/3*u + 2/3*u2 + 2/3*dfdx*dt

        iteration += 1
        t += dt
        #print("Iteration: %i, t: %.3e"%(iteration, t))

        if np.max(np.abs(u)) > u_max:
            print("NN Diverged!")
            break

    return u

def adv_dfdx_NN(model_info, u, c, dx):
    model_id, weights, data_func = model_info
    f_bar = adv_get_f_bar(u, c)
    X = data_func(f_bar)

    f_NN = model_predict(model_id, weights, X)

    f_r = f_NN
    f_l = np.roll(f_NN, 1)

    dfdx = (f_r - f_l)/dx

    return dfdx

# 1st-order upwind
def adv_upwind(u, c, dx):
    f = -c*u

    f_p = f
    f_m = np.roll(f, 1)
    dfdx = (f_p - f_m)/dx

    return dfdx   


# error
def rmse(a, b):
    dev = a - b
    err = np.sqrt(np.mean(dev**2))

    return err

# Total Variation
def compute_TV(u):
    u_p = u
    u_m = np.roll(u, 1)
    TV = np.sum(np.abs(u_p - u_m))

    return TV


## Testing template ##
def test_adv(adv_func, wave_func):
    # physics
    L = 1 # periodic boundary f(x+L) = f(x)
    c = 1

    # simulation parameter
    nI = 100
    CFL = 0.5
    T = 10

    # initialization
    x = np.linspace(0, L, nI+1)[:-1]
    u0 = wave_func(x, 0, c, L)

    # simulation
    u_sim = adv_RK3(adv_func, u0, x, c, CFL, T)

    # evalution
    u_exact = wave_func(x, T, c, L)
    err =  rmse(u_sim, u_exact)
    TV = compute_TV(u_sim)
    

    return err, TV

def test_adv_NN(model_info, wave_func):
    # physics
    L = 1 # periodic boundary f(x+L) = f(x)
    c = 1

    # simulation parameter
    nI = 100
    CFL = 0.5
    T = 10

    # initialization
    x = np.linspace(0, L, nI+1)[:-1]
    u0 = wave_func(x, 0, c, L)

    # simulation
    u_NN = adv_RK3_NN(model_info, u0, x, c, CFL, T)

    # evalution
    u_exact = wave_func(x, T, c, L)
    err =  rmse(u_NN, u_exact)
    TV = compute_TV(u_NN)

    return err, TV

def view_adv_NN(model_info, wave_func, T):
    # physics
    L = 1 # periodic boundary f(x+L) = f(x)
    c = 1

    # simulation parameter
    nI = 100
    CFL = 0.5

    # initialization
    x = np.linspace(0, L, nI+1)[:-1]
    u0 = wave_func(x, 0, c, L)

    # simulation
    u_NN = adv_RK3_NN(model_info, u0, x, c, CFL, T)
    u_weno = adv_RK3(WENO5_getf, u0, x, c, CFL, T)

    # evalution
    u_exact = wave_func(x, T, c, L)
    err =  rmse(u_NN, u_exact)
    TV = compute_TV(u_NN)
    print("error:", err)
    print("TV:", TV)

    
    x_exact =np.linspace(0, L, 1001)
    u_exact = wave_func(x_exact, T, c, L)

    plt.plot(x, u_NN, "ob", label="WENO-NN")
    plt.plot(x, u_weno, "sr", label="WENO5-JS")
    plt.plot(x_exact, u_exact, "--k")
    plt.legend(fontsize=14)
    plt.show()

    return err, TV


# Macro scripts
def test_WENO5_JS():
    print("Testing... (sin)")
    err_sin, TV_sin = test_adv(WENO5_getf, exact_sin)
    print("error:", err_sin)
    print("TV:", TV_sin)
    print()

    print("Testing... (square)")
    err_sq, TV_sq = test_adv(WENO5_getf, exact_square)
    print("error:", err_sq)
    print("TV:", TV_sq)
    print()

    return (err_sin, TV_sin), (err_sq, TV_sq)

def test_WENO_NN(model_path):
    model_info = read_model(model_path)

    print("Testing... (sin)")
    err_sin, TV_sin = test_adv_NN(model_info, exact_sin)
    print("error:", err_sin)
    print("TV:", TV_sin)
    print()
    
    print("Testing... (square)")
    err_sq, TV_sq = test_adv_NN(model_info, exact_square)
    print("error:", err_sq)
    print("TV:", TV_sq)
    print()

    return (err_sin, TV_sin), (err_sq, TV_sq)

def view_WENO_NN(model_path, T):
    model_info = read_model(model_path)

    print("Testing... (sin)")
    err_sin, TV_sin = view_adv_NN(model_info, exact_sin, T)
    print()
    
    print("Testing... (square)")
    err_sq, TV_sq = view_adv_NN(model_info, exact_square, T)
    print()

if __name__ == "__main__":
    model_folder = "test_batch/git_3_lamb_6/"
    model_name = "model_batch_11.bin"
    model_path = model_folder + model_name

    test_WENO5_JS()
    view_WENO_NN(model_path, 50)

