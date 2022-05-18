# View the scatter plot of data and prediction

### Libraries ###
import numpy as np
import matplotlib.pyplot as plt
from train_SC import WENO5_getC, get_dataX

### Model Testing ###
def read_model_test(path):
    print("Reading model: %s"%path)
    file = open(path, "rb") 

    nn_count = readline(file, np.int32, 1)[0]
    
    weights = []

    for nn_idx in range(nn_count):
        dim = readline(file, np.int32, 2)
        n_w = dim[1] * dim[0]
        weight = readline(file, np.float32, n_w)
        weights += [weight.reshape(dim)]
    file.close()

    return weights

def readline(file, dtype, count):
    if dtype == np.float32:
        byte = 4
    elif dtype == np.int32:
        byte = 4 

    byte_arr = file.read(byte*count)
    return np.frombuffer(byte_arr, dtype=dtype, count=count)

#
def model_predict(weights, c_tilde, f_bar):
    # Neural Network
    nn_count = len(weights)

    # sigmoid->sigmoid->sigmoid->linear
    act_list = [actf_sigmoid, actf_sigmoid, actf_sigmoid, actf_linear]
    c_hat = c_tilde
    for nn_idx in range(nn_count):
        act_func = act_list[nn_idx]

        weight = weights[nn_idx]
        c_hat = np.matmul(c_hat, weight)
        c_hat = act_func(c_hat)

    # Affine Transformation 
    sum_c = np.sum(c_hat, axis=1).reshape((-1, 1))
    c_hat_s = c_hat - sum_c/5
    c = c_hat_s + c_tilde

    # Inner product
    f_NN = np.sum(c*f_bar, axis=1)
    return f_NN


# activation functions 
def actf_sigmoid(x):
    return 1/(1 + np.exp(-x))

def actf_linear(x):
    return x

def L2_norm(a, b):
    dev = a - b
    err_L2 = np.sqrt(np.mean(dev*dev))
    return err_L2


if __name__ == "__main__":
    folder = "training_data/"
    file = "test_SC_1.npy"
    f_bar, y = get_dataX(folder+file)

    # WENO5    
    c_tilde = WENO5_getC(f_bar)
    f_weno = np.sum(c_tilde*f_bar, axis=1)
    plt.plot(y, f_weno, "ob", markersize=0.2, label="WENO5-JS", alpha=1)
    print("WENO5-JS error: %.3e"%(L2_norm(f_weno, y)))

    # Read from bin file:
    path = "test_model.bin"
    weights = read_model_test(path)
    f_NN = model_predict(weights, c_tilde, f_bar )
    #plt.plot(y, f_NN, "or", markersize=0.2, label="WENO-NN1", alpha=1)
    print("WENO-NN error: %.3e"%(L2_norm(f_NN, y)))


    # slope = 1
    n_one = 101
    lims = [-0.3, 1.3]
    x = np.linspace(*lims, n_one)
    plt.plot(x, x, "--k", linewidth=1)
    
    # plot setting
    plt.title("Dataset: %s"%file, fontsize=16)
    plt.ylabel(r"predicted ($\hat{f}$)", fontsize=12)
    plt.xlabel(r"exact ($f$)", fontsize=12)
    plt.ylim(lims)
    plt.xlim(lims)
    # customize markersize in legend
    legend = plt.legend(fontsize=12, markerscale=20)

    plt.grid()
    plt.show()
    #plt.savefig("scatter_f.png")
