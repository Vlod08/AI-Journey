import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression

basedir = 'E:/AI/Problem-Sets Maths and Code/CS229-Machine-Learning-Stanford/PS1-2018/'


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(train_path, add_intercept=True)

    for t in tau_values:
        fig = plt.figure()
        LWR1 = LocallyWeightedLinearRegression(t)
        LWR1.fit(x_train,y_train)
        y_pred = LWR1.predict(x_eval)
        plt.scatter(x_train[:,1],y_train,c='blue')
        plt.scatter(x_eval[:,1],y_pred,c='red')
        plt.title(f"Locally Wighted Regression : tau = {t}")
        plt.savefig(basedir+'output/p05c_tau_'+str(t) + '.png')
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
