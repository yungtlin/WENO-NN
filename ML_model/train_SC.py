# Trains WENO-NN by Stevens and Colonius 
### Libraries ###
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

### Data ###
def get_dataX(path):
    with open(path, "rb") as f:
        print("Loading data: %s"%path)
        dataX = np.load(f)
    X = dataX[:, :-1]
    y = dataX[:, -1]
    return X, y

### WENO ###
def WENO5_getC(f_tilde):
    # check dimension (r=3)
    r = 3
    if f_tilde.shape[1] != (2*r-1):
        raise ValueError("Data Dimension Must Be 5")

    omega = WENO5_omega(f_tilde)
    C = WENO5_C(omega)

    return C


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

def WENO5_C(omega):
    nX = omega.shape[0]

    C = np.zeros((nX, 5))

    C[:, 0] = 1/3*omega[:,0]
    C[:, 1] = -7/6*omega[:,0] - 1/6*omega[:,1]
    C[:, 2] = 11/6*omega[:,0] + 5/6*omega[:,1] + 1/3*omega[:,2]
    C[:, 3] = 1/3*omega[:,1] + 5/6*omega[:,2]
    C[:, 4] = -1/6*omega[:,2]

    return C


if __name__ == "__main__":
    folder = "training_data/"
    file = "data_SC_0.npy"
    f_bar, y = get_dataX(folder+file)
    
    nf = f_bar.shape[1]
    c_tilde = WENO5_getC(f_bar)



    # model inputs
    X = {"c_tilde":c_tilde, "f_bar":f_bar},

    ## model ##
    # ref: https://www.tensorflow.org/guide/keras/functional

    # WENO5-JS coefficients
    c_input = keras.Input(shape=(nf,), name="c_tilde")
    
    # Neural Network
    NN_h1 = layers.Dense(3, activation="sigmoid", name="hidden1")
    NN_h2 = layers.Dense(3, activation="sigmoid", name="hidden2")
    NN_h3 = layers.Dense(3, activation="sigmoid", name="hidden3")
    NN_out = layers.Dense(nf, activation="sigmoid", name="output")
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

    
    model.summary()
    # plot networks
    keras.utils.plot_model(model, "WENO-NN.png")

    
    # training solver
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    
    history = model.fit(X, y, batch_size=64, epochs=10, validation_split=0.2)

    # compare 
    test_scores = model.evaluate(X, y, verbose=2)
    f_weno = np.sum(c_tilde*f_bar, axis=1)
    dev = f_weno - y
    ref_loss = np.sqrt(np.mean(dev*dev))
    print("WENO5-JS rmse:", ref_loss)
