### Modules ###
import numpy as np 
import matplotlib.pyplot as plt 

from train_model import *
from advection_test import test_WENO_NN

def write_log(file, err_test, is_saved, model_save):
    if is_saved:
        file.write("##### SAVED model: %s #####\n"%model_save)

    err_test = (err_sin, TV_sin), (err_sq, TV_sq)    

    file.write("Advection (sin)\n")
    file.write("rmse: %.5e\n"%err_sin)
    file.write("TV: %.5f\n"%TV_sin)

    file.write("Advection (square)\n")
    file.write("rmse: %.5e\n"%err_sq)
    file.write("TV: %.5f\n"%TV_sq)

    if is_saved:
            file.write("####### SAVED model END #######\n")

    file.write("\n\n")



if __name__ == "__main__":
    n_epochs = 10
    idx = 0
    model_save = 0

    log_file = "test_log.txt"
    file = open(log_file, "w")

    data_folder = "training_data/"
    data_file = "data_github.npy"
    #data_file = "test_SC_1.npy"
    f_bar, y = get_dataX(data_folder + data_file)

    err_list = []
    TV_list = []
    
    total_test = 100
    for i in range(total_test):
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

        err_list += [err_sq]
        TV_list += [TV_sq]

        # Save condition
        # WENO5-JS 
        # err_sin: 6.241e-05
        # TV_sin: 3.9996

        # err_sq: 0.11648111689410975
        # TV_sq: 2.006

        #is_sin = (err_sin < 4e-2 and TV_sin < 4.2)
        is_square = err_sq < 0.1165 
        is_TVB = (TV_sin < 4.1 and TV_sq < 2.03)

        # 0.2400 for data_github
        is_trained = mse_train[-1] < 0.2400 

        if True: #is_square and is_TVB and is_trained:
            model_save = "model_batch_%i.bin"%idx
            path_save = model_folder + model_save
            save_model(path_save, model, model_id)
            idx += 1
            is_saved = True
        else:
            is_saved = False

        err_test = (err_sin, TV_sin), (err_sq, TV_sq)


        write_log(file, err_test, is_saved, model_save)
        file.flush()

    file.close()


    TV_info = np.array([err_list, TV_list])
    TV_file = "git_3_lamb_24.npy"
    TV_path = model_folder + TV_file

    with open(TV_path, 'wb') as f:
        np.save(f, TV_info)
