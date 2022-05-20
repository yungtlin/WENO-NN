# Trains WENO-NN by Stevens and Colonius 
### Libraries ###
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import numpy as np

### Data ###
def get_dataX(path):
    with open(path, "rb") as f:
        print("Loading data: %s"%path)
        dataX = np.load(f)
    
    # shuffle data 
    np.random.shuffle(dataX)

    X = dataX[:, :-1]
    y = dataX[:, -1]

    return X, y

### WENO ###
# compute WENO5-JS coefficients
def WENO5_getC(f_tilde):
    # check dimension (r=3)
    r = 3
    if f_tilde.shape[1] != (2*r-1):
        raise ValueError("Data Dimension Must Be 5")

    omega = WENO5_omega(f_tilde)
    C = WENO5_C(omega)

    return C

# nonlinear weights
def WENO5_omega(f, epsilon=1e-6):
    nX = f.shape[0]

    # beta
    beta = np.zeros((nX, 3))    
    beta[:, 0] = 13/12*(f[:,0] - 2*f[:,1] + f[:,2])**2\
        + 1/4*(f[:,0] - 4*f[:,1] + 3*f[:,2])**2
    beta[:, 1] = 13/12*(f[:,1] - 2*f[:,2] + f[:,3])**2\
        + 1/4*(f[:,1] - f[:,3])**2
    beta[:, 2] = 13/12*(f[:,2] - 2*f[:,3] + f[:,4])**2\
        + 1/4*(3*f[:,2] - 4*f[:,3] + f[:,4])**2

    # sigma
    gamma = np.array([1/10, 3/5, 3/10]).reshape((1, -1))
    sigma = gamma/(epsilon + beta)**2

    # omega
    sigma_sum = np.sum(sigma, axis=1).reshape((-1, 1))
    omega = sigma/sigma_sum

    return omega

# WENO5 coefficients
def WENO5_C(omega):
    nX = omega.shape[0]

    C = np.zeros((nX, 5))

    C[:, 0] = 1/3*omega[:,0]
    C[:, 1] = -7/6*omega[:,0] - 1/6*omega[:,1]
    C[:, 2] = 11/6*omega[:,0] + 5/6*omega[:,1] + 1/3*omega[:,2]
    C[:, 3] = 1/3*omega[:,1] + 5/6*omega[:,2]
    C[:, 4] = -1/6*omega[:,2]

    return C

### model ###
def get_model(nf=5):
    ## model ##
    # ref: https://www.tensorflow.org/guide/keras/functional

    # WENO5-JS coefficients
    c_input = keras.Input(shape=(nf,), name="c_tilde")
    
    # Neural Network
    l2_lambda = 1e-1

    NN_h1 = layers.Dense(3, activation="sigmoid", use_bias=False,\
        kernel_regularizer=regularizers.L2(l2_lambda), name="hidden1")
    NN_h2 = layers.Dense(3, activation="sigmoid", use_bias=False,\
        kernel_regularizer=regularizers.L2(l2_lambda), name="hidden2")
    NN_h3 = layers.Dense(3, activation="sigmoid", use_bias=False,\
        kernel_regularizer=regularizers.L2(l2_lambda),name="hidden3")
    NN_out = layers.Dense(nf, activation="linear", use_bias=False,\
        kernel_regularizer=regularizers.L2(l2_lambda), name="dc_tilde")
    c_hat = NN_out(NN_h3(NN_h2(NN_h1(c_input))))

    # Affine Transformation 
    sum_c = tf.reduce_sum(c_hat, axis=1, keepdims=True)
    c_hat_s = c_hat - sum_c/nf # L2-optimal
    c = c_hat_s + c_input

    # Inner product
    f_input = keras.Input(shape=(nf,), name="f_bar")
    outputs = layers.Dot(axes=1)([c, f_input]) # WENO-NN output

    # model wrap up
    model = keras.Model(inputs=[c_input, f_input], outputs=outputs, name="WENO-NN")

    # set solver
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    model.summary()

    return model

# saves the neural networks weights in a binary file
def save_model(path, model):
    print("Saving model: %s"%path)
    file = open(path, "wb")

    # neural networks indices in the tf model
    nn_list = list(range(1, 5))

    # total number of layers (included output) 
    nn_count = len(nn_list)
    writeline(file, np.int32, nn_count)

    # loops over specified layers
    for nn_idx in nn_list:
        nn_layer = model.layers[nn_idx].get_weights()
        weight = nn_layer[0]
        w_dim = weight.shape
        # writes weights dimension
        writeline(file, np.int32, w_dim)
        # writes weights
        writeline(file, np.float32, weight)

    file.close()

# write data to the bin file
def writeline(file, dtype, data):
    data_np = np.array(data, dtype=dtype)
    file.write(data_np.tobytes())

if __name__ == "__main__":
    folder = "training_data/"
    file = "test_SC_1.npy"
    f_bar, y = get_dataX(folder+file)
    
    nf = f_bar.shape[1]
    c_tilde = WENO5_getC(f_bar)

    # model inputs
    X = {"c_tilde":c_tilde, "f_bar":f_bar},

    # get NN model
    model = get_model()

    # plot networks
    #keras.utils.plot_model(model, "test_WENO-NN.png")

    n_epochs = 50
    history = model.fit(X, y, batch_size=100, epochs=n_epochs, validation_split=0.2)

    path = "test_model_SC1.bin"
    #save_model(path, model)

    import matplotlib.pyplot as plt
    mse_train = history.history["root_mean_squared_error"]
    mse_valid = history.history["val_root_mean_squared_error"]

    x = np.arange(n_epochs) + 1
    plt.plot(x, mse_train, "b", label="training")
    plt.plot(x, mse_valid, "r", label="validation")
    
    plt.title("Training History", fontsize=16)
    plt.ylabel("Mean square Error", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig("test_training_epoch_%i.png"%n_epochs)
