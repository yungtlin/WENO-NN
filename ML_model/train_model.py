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
# weno5 flux
def WENO5_getf(f_bar):
    c_tilde = WENO5_getC(f_bar)
    f_weno = np.sum(c_tilde*f_bar, axis=1)
    return f_weno

# compute WENO5-JS coefficients
def WENO5_getC(f_bar):
    # check dimension (r=3)
    r = 3
    if f_bar.shape[1] != (2*r-1):
        raise ValueError("Data Dimension Must Be 5")

    omega = WENO5_omega(f_bar)
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

def WENO5_f_hat(f):
    nX = f.shape[0]

    f_hat = np.zeros((nX, 3))
    f_hat[:,0] = 1/3*f[:,0] -7/6*f[:,1] +11/6*f[:,2]
    f_hat[:,1] =-1/6*f[:,1] +5/6*f[:,2]  +1/3*f[:,3]
    f_hat[:,2] = 1/3*f[:,2] +5/6*f[:,3]  -1/6*f[:,4]

    return f_hat

def WENO_M2_f(f):
    nX = f.shape[0]

    
    f_weno = WENO5_getf(f)

    # for step
    f1st = 3/2*f[:,3] - 1/2*f[:,4]
    ones = np.ones(nX)

    # TVD bound
    f_min = np.min([f_weno, 0*ones], axis=0)
    f_max = np.max([f_weno, ones], axis=0)
    f1_clip = np.clip(f1st, f_min, f_max)

    f_hat = np.zeros((nX, 2))
    f_hat[:,0] = f_weno
    f_hat[:,1] = f1_clip

    return f_hat

### model ###
def get_model_SC(nf1=5, nf2=5):
    model_id = 1

    data_func = get_model_SC_data

    ## model ##
    # ref: https://www.tensorflow.org/guide/keras/functional

    # WENO5-JS coefficients
    c_input = keras.Input(shape=(nf1,), name="x1")

    # Neural Network
    bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.005, seed=None)

    l2_lambda = 6 + 10*np.random.rand()   

    np1 = np.random.randint(3) + 3
    np2 = np.random.randint(3) + 3
    np3 = np.random.randint(3) + 3


    NN_h1 = layers.Dense(np1, activation="relu",\
        bias_initializer=bias_init,\
        name="hidden1")
    NN_h2 = layers.Dense(np2, activation="relu",\
        bias_initializer=bias_init,\
        name="hidden2")
    NN_h3 = layers.Dense(np3, activation="relu",\
        bias_initializer=bias_init,\
        name="hidden3")

    ###
     
    NN_out = layers.Dense(nf2, activation="linear", use_bias=True,\
        bias_initializer=bias_init,\
        activity_regularizer=regularizers.L2(l2_lambda),\
        name="dc_tilde")
    c_hat = NN_out(NN_h3(NN_h2(NN_h1(c_input))))

    # Affine Transformation 
    sum_c = tf.reduce_sum(c_hat, axis=1, keepdims=True)
    c_hat_s = c_hat - sum_c/nf2 # L2-optimal
    c = c_hat_s + c_input

    # Inner product
    f_input = keras.Input(shape=(nf2,), name="x2")
    outputs = layers.Dot(axes=1)([c, f_input]) # WENO-NN output

    # model wrap up
    model = keras.Model(inputs=[c_input, f_input], outputs=outputs, name="WENO-NN")

    # set solver
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    #model.summary()

    return model, model_id, data_func

def get_model_SC_data(f_bar):
    c_tilde = WENO5_getC(f_bar)

    # model inputs
    X = {"x1":c_tilde, "x2":f_bar}
    return X


# Using x1 as input and predicting weights as neural networks output
# Then inner product the weights with x2
def get_model_2(nf1=3, nf2=2):
    model_id = 2
    data_func = get_model_2_data

    ## model ##
    # ref: https://www.tensorflow.org/guide/keras/functional

    # WENO5-JS coefficients
    omega_input = keras.Input(shape=(nf1,), name="x1")
    
    # Neural Network
    l2_lambda = 0.1

    NN_h1 = layers.Dense(3, activation="relu", use_bias=False,\
        kernel_regularizer=regularizers.L2(l2_lambda),\
        name="hidden1")
    
    NN_h2 = layers.Dense(3, activation="relu", use_bias=False,\
        kernel_regularizer=regularizers.L2(l2_lambda),\
        name="hidden2")
    
    NN_h3 = layers.Dense(3, activation="relu", use_bias=False,\
        kernel_regularizer=regularizers.L2(l2_lambda),\
        name="hidden3")
    
    NN_out = layers.Dense(nf2, activation="softmax", use_bias=False,\
        kernel_regularizer=regularizers.L2(l2_lambda),\
        name="omega_hat")

    omega_hat = NN_out(NN_h3(NN_h2(NN_h1(omega_input))))

    # Inner product
    f_input = keras.Input(shape=(nf2,), name="x2")
    outputs = layers.Dot(axes=1)([omega_hat, f_input]) # WENO-NN output

    # model wrap up
    model = keras.Model(inputs=[omega_input, f_input], outputs=outputs, name="WENO-NN")

    # set solver
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.1),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    model.summary()

    return model, model_id, data_func

def get_model_2_data(f_bar):
    omega = WENO5_omega(f_bar)
    f_hat = WENO_M2_f(f_bar)

    # model inputs
    X = {"x1":omega, "x2":f_hat}
    return X


### Model Saving ###
# saves the neural networks weights in a binary file
def save_model(path, model, model_id):
    print("Saving model: %s"%path)
    file = open(path, "wb")

    # save model_id
    writeline(file, np.int32, model_id)

    if model_id == 1:
        save_model1(file, model)
    elif model_id == 2:
        save_model2(file, model)

    file.close()

def save_model1(file, model):
    # neural networks indices in the tf model
    nn_list = list(range(1, 5))

    # total number of layers (included output) 
    nn_count = len(nn_list)
    writeline(file, np.int32, nn_count)

    # loops over specified layers
    for nn_idx in nn_list:
        nn_layer = model.layers[nn_idx].get_weights()

        nW = len(nn_layer) # nW = 1 (no bias), nw = 2 (biased)
        
        writeline(file, np.int32, nW)
        weight = nn_layer[0]
        w_dim = weight.shape
        # writes weights dimension
        writeline(file, np.int32, w_dim)
        # writes weights
        writeline(file, np.float32, weight)

        if nW == 2:
            bias = nn_layer[1]
            writeline(file, np.float32, bias)

def save_model2(file, model):
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


# write data to the bin file
def writeline(file, dtype, data):
    data_np = np.array(data, dtype=dtype)
    file.write(data_np.tobytes())


def plot_history(history):
    # Training history #
    import matplotlib.pyplot as plt
    mse_train = history.history["root_mean_squared_error"]
    mse_valid = history.history["val_root_mean_squared_error"]

    n_epochs = len(mse_train)

    x = np.arange(n_epochs) + 1
    plt.plot(x, mse_train, "b", label="training")
    plt.plot(x, mse_valid, "r", label="validation")
    
    plt.title("Training History", fontsize=16)
    plt.ylabel("Mean square Error", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig("test_training_epoch_%i.png"%n_epochs)


if __name__ == "__main__":
    folder = "training_data/"
    #file_data = "data_github.npy"
    file_data = "data_github.npy"
    f_bar, y = get_dataX(folder+file_data)
    
    model, model_id, data_func = get_model_SC()
    X = data_func(f_bar)    


    # plot networks
    #keras.utils.plot_model(model, "test_WENO-NN.png")

    
    n_epochs = 10
    history = model.fit(X, y, batch_size=80, epochs=n_epochs, validation_split=0.2)

    plot_history(history)

    model_path = "test_model_SC.bin"
    save_model(model_path, model, model_id)


    import advection_test
    #advection_test.test_WENO5_JS()
    advection_test.test_WENO_NN(model_path)
