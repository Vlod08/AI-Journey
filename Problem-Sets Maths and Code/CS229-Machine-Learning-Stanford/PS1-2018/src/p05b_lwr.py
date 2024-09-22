import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    # Fit a LWR model
    
    LWR1 = LocallyWeightedLinearRegression(0.5)
    LWR1.fit(x_train,y_train)
    y_pred = LWR1.predict(x_eval)
    # Get MSE value on the validation set
    MSE = np.mean((y_pred - y_eval)**2)
    # Plot validation predictions on top of training set

    print(x_train.shape)

    fig = plt.figure()
    plt.scatter(x_train[:,1],y_train,c='blue')
    plt.scatter(x_eval[:,1],y_pred,c='red')
    #plt.show()
    print(MSE)


    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y 
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m_pred,n_pred = x.shape
        m_train,n_train = self.x.shape
        y_pred = np.zeros(m_pred)
        for p in range(m_pred):
            
            W = np.zeros(m_train)

            for i in range(m_train):
                norm2 = np.sum(self.x[i]-x[p])**2
                W[i] = np.exp(-norm2/(0.5 * self.tau**2))
            W = np.diag(W)
            teta = np.linalg.inv(np.dot(np.dot(self.x.T,W),self.x) )
            teta = np.dot(np.dot(np.dot(teta , self.x.T),W),self.y)
            y_pred[p] = np.dot(x[p],teta)
        return y_pred




        # *** END CODE HERE ***
