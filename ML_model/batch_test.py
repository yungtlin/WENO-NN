### Modules ###
import numpy as np 
import matplotlib.pyplot as plt 

from train_model import *
from advection_test import test_WENO_NN

def write_log(file, err_test, is_saved, model_save):
    if is_saved:
        file.write("SAVED model: %s\n"%model_save)

    err_test = (err_sin, TV_sin), (err_sq, TV_sq)    

    file.write("Advection (sin)\n")
    file.write("rmse: %.5e\n"%err_sin)
    file.write("TV: %.5f\n\n"%TV_sin)

    file.write("Advection (square)\n")
    file.write("rmse: %.5e\n"%err_sq)
    file.write("TV: %.5f\n\n\n"%TV_sq)


if __name__ == "__main__":
    n_epochs = 10
    idx = 0
    model_save = 0

    log_file = "test_log.txt"
    file = open(log_file, "w")

    data_folder = "training_data/"
    data_file = "data_github.npy"
    f_bar, y = get_dataX(data_folder + data_file)



    model, model_id, data_func = get_model_SC()
    X = data_func(f_bar)
    # model traning
    history = model.fit(X, y, batch_size=80, epochs=n_epochs, validation_split=0.2)

    mse_train = history.history["root_mean_squared_error"]

    model_folder = "test_batch/"
    model_temp = "model_batch_temp.bin"
    path_temp = model_folder + model_temp

    # save temp
    save_model(path_temp, model, model_id)

    # test temp
    (err_sin, TV_sin), (err_sq, TV_sq) = test_WENO_NN(path_temp)

    # Save condition
    is_TVB = (TV_sin <= 10 and TV_sq <= 6)
    # 0.2000 for data_github
    is_trained = mse_train[-1] < 0.2000 

    if is_TVB and is_trained:
        model_save = "model_batch_%i.bin"%idx
        path_save = model_folder + model_save
        save_model(path_save, model, model_id)
        idx += 1
        is_saved = True
    else:
        is_saved = False

    err_test = (err_sin, TV_sin), (err_sq, TV_sq)


    write_log(file, err_test, is_saved, model_save)

    file.close()
