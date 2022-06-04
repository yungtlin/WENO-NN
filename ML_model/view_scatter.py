# View the scatter plot of data and prediction

### Libraries ###
import numpy as np
import matplotlib.pyplot as plt
from train_model import *

### Model Testing ###
def read_model(path):
    print("Reading model: %s"%path)
    file = open(path, "rb") 

    model_id = readline(file, np.int32, 1)[0]

    if model_id == 1:
        weights = read_model_SC(file)
        data_func = get_model_SC_data

    elif model_id == 2:
        weights = read_model_2(file)
        data_func = get_model_2_data

    file.close()

    return model_id, weights, data_func

def read_model_SC(file):
    nn_count = readline(file, np.int32, 1)[0]
    
    weights = []

    for nn_idx in range(nn_count):
        nW = readline(file, np.int32, 1)
        dim = readline(file, np.int32, 2)
        n_w = dim[1] * dim[0]
        # weight
        weight = readline(file, np.float32, n_w)
        weight = weight.reshape(dim)
        print(weight)
        # bias
        if nW == 2:
            bias = readline(file, np.float32, dim[1])
            print(bias)
        else:
            bias = []

        weights += [(weight, bias)]

    return weights


def read_model_2(file):
    nn_count = readline(file, np.int32, 1)[0]
    
    weights = []

    for nn_idx in range(nn_count):
        dim = readline(file, np.int32, 2)
        n_w = dim[1] * dim[0]
        weight = readline(file, np.float32, n_w)
        weights += [weight.reshape(dim)]

    return weights


def readline(file, dtype, count):
    if dtype == np.float32:
        byte = 4
    elif dtype == np.int32:
        byte = 4 

    byte_arr = file.read(byte*count)
    return np.frombuffer(byte_arr, dtype=dtype, count=count)

#
def model_predict(model_id, weights, X):
    if model_id == 1:
        c_tilde = X["x1"]
        f_bar = X["x2"]
        f_NN = model_SC(weights, c_tilde, f_bar)
    elif model_id == 2:
        omega = X["x1"]
        f_hat = X["x2"]
        f_NN = model_2(weights, omega, f_hat)

    return f_NN

# model ID: 1
def model_SC(weights, c_tilde, f_bar):
    # Neural Network
    nn_count = len(weights)

    # sigmoid->sigmoid->sigmoid->linear
    act_list = [actf_relu, actf_relu, actf_relu, actf_linear]
    c_hat = c_tilde
    for nn_idx in range(nn_count):
        act_func = act_list[nn_idx]

        weight, bias = weights[nn_idx]
        
        c_hat = np.matmul(c_hat, weight)
        
        if len(bias) != 0:
            bias = bias.reshape((1, -1))
            c_hat = c_hat + bias

        c_hat = act_func(c_hat)

    # Affine Transformation 
    sum_c = np.sum(c_hat, axis=1).reshape((-1, 1))
    c_hat_s = c_hat - sum_c/5
    c = c_hat_s + c_tilde

    # Inner product
    f_NN = np.sum(c*f_bar, axis=1)
    return f_NN

# model ID: 2
def model_2(weights, x1, f_hat):
    # Neural Network
    nn_count = len(weights)

    # sigmoid->sigmoid->sigmoid->linear
    act_list = [actf_relu, actf_relu, actf_relu, actf_softmax]
    omega_hat = x1
    for nn_idx in range(nn_count):
        act_func = act_list[nn_idx]

        weight = weights[nn_idx]
        omega_hat = np.matmul(omega_hat, weight)
        omega_hat = act_func(omega_hat)

    # Inner product
    f_NN = np.sum(omega_hat*f_hat, axis=1)
    
    return f_NN




# activation functions 
def actf_sigmoid(x):
    return 1/(1 + np.exp(-x))

def actf_linear(x):
    return x

def actf_relu(x):
    return np.where(x>0, x, 0)

def actf_softmax(x):
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=1).reshape((-1, 1))

    return exp_x/sum_x



def L2_norm(a, b):
    dev = a - b
    err_L2 = np.sqrt(np.mean(dev*dev))
    return err_L2



if __name__ == "__main__":
    folder = "training_data/"
    data_name = "test_SC_1.npy"
    #data_name = "data_github.npy"
    f_bar, y = get_dataX(folder+data_name)

    # WENO5    
    c_tilde = WENO5_getC(f_bar)
    f_weno = np.sum(c_tilde*f_bar, axis=1)

    plt.plot(y, f_weno, "ob", markersize=0.2, label="WENO5-JS", alpha=1)
    print("WENO5-JS error: %.3e"%(L2_norm(f_weno, y)))

    # Read from bin file:
    model_path = "model_batch_10.bin"
    model_id, weights, data_func = read_model(model_path)
    X = data_func(f_bar)
'''
    f_NN = model_predict(model_id, weights, X)
    
    plt.plot(y, f_NN, "or", markersize=0.2, label="WENO-NN", alpha=1)
    print("WENO-NN error: %.3e"%(L2_norm(f_NN, y)))

    # slope = 1
    n_one = 101
    lims = [-0.3, 1.3]
    x = np.linspace(*lims, n_one)
    plt.plot(x, x, "--k", linewidth=1)
    
    # plot setting
    plt.title("Dataset: %s"%(data_name), fontsize=16)
    plt.ylabel(r"predicted ($\hat{f}$)", fontsize=12)
    plt.xlabel(r"exact ($f$)", fontsize=12)
    plt.ylim(lims)
    plt.xlim(lims)
    # customize markersize in legend
    legend = plt.legend(fontsize=12, markerscale=20)

    plt.grid()
    #plt.show()
    
    #plt.savefig("test_scatter_1e-2.png")
    plt.savefig("scatter_f.png")
'''