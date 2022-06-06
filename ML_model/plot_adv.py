### Libraries ###
import numpy as np
import matplotlib.pyplot as plt 

from advection_test import *

if __name__ == "__main__":
    T = 10

    # low TV 
    # x_TV = 2.0017
    # y_err = 0.145
    model_path = "model_TV_low.bin"
    x, u_NN = cal_WENO_NN(model_path, T, exact_square)
    plt.plot(x, u_NN, "-sr", label=r"Low TV")
    
    # high TV 
    # x_TV = 2.2056
    # y_err = 0.0777    
    model_path = "model_TV_high.bin"
    x, u_NN = cal_WENO_NN(model_path, T, exact_square)
    plt.plot(x, u_NN, "-dg", label="High TV")

    # medium TV
    # x_TV = 2.00949
    # y_err = 0.11075
    model_path = "model_TV_med.bin"
    x, u_NN = cal_WENO_NN(model_path, T, exact_square)
    plt.plot(x, u_NN, "-ob", label="Medium TV")


    x, u_weno = cal_WENO5_JS(T, exact_square)
    plt.plot(x, u_weno, "-m+", label="WENO5-JS")

    # Exact
    plot_exact(T, exact_square)

    plt.title("Linear Advection (T=%.1f)"%T, fontsize=18)
    plt.legend(fontsize=14)
    plt.ylabel("u", fontsize=14)
    plt.xlabel("x", fontsize=14)
    plt.grid()
    
    plt.savefig("plot_adv.png")
