# View the scatter plot of data and prediction

### Libraries ###
import numpy as np
import matplotlib.pyplot as plt
from train_SC import WENO5_getC, get_dataX
 

if __name__ == "__main__":
    folder = "training_data/"
    file = "data_SC_0.npy"
    X, y = get_dataX(folder+file)

    # WENO5    
    c_tilde = WENO5_getC(X)
    f_weno = np.sum(c_tilde*X, axis=1)
    plt.plot(y, f_weno, "ob", markersize=0.2, label="WENO5-JS", alpha=1)


        
    test_scores = model.evaluate(X, y, verbose=2)
    f_weno = np.sum(c_tilde*f_bar, axis=1)
    dev = f_weno - y
    ref_loss = np.sqrt(np.mean(dev*dev))
    print("WENO5-JS rmse:", ref_loss)


    # slope = 1
    n_one = 101
    plt.plot(x, x, "--k", linewidth=1)

    # plotting
    lims = [-0.5, 1.5]
    x = np.linspace(*lims, n_one)
    plt.title("Dataset: %s"%file, fontsize=16)
    plt.ylabel(r"predict ($\hat{f}$)", fontsize=12)
    plt.xlabel(r"exact ($f$)", fontsize=12)
    plt.ylim(lims)
    plt.xlim(lims)
    plt.legend(fontsize=14)
    plt.grid()
    plt.show()


