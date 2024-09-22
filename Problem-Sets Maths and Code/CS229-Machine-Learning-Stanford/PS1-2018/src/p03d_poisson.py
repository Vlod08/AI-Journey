import numpy as np
import util
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    P = PoissonRegression()
    # Fit a Poisson Regression model
    teta_tmp = P.fit(x_train,y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = P.predict(x_eval)
    #np.savetxt(fname=pred_path,X=y_pred)

    plt.figure()
    diff = y_pred-y_eval
    plt.hist(diff,bins=50,color='blue')

    plt.xlim(diff.min()-2, diff.max()+2)
    rectangle = Rectangle((0,0),1,1,color='blue', ec='k')
    plt.legend([rectangle],["Y_prediction - Y_test_label"])
    plt.savefig(pred_path.replace('txt','png'))

    
    
    
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***-
        m,n = x.shape
        self.teta = np.zeros(n)
        epsilon = 10e-5
        old_teta = np.ones(n)*np.inf 
        alpha = 10e-6
        nb_iterations = 10000
        iter = 0
        while np.linalg.norm(old_teta-self.teta,ord=1) > epsilon and iter < nb_iterations:
            old_teta = np.copy(self.teta)
            for i in range(m):
                #print((y[i]-np.exp(np.dot(self.teta,x[i]))))
                self.teta += (alpha/m)*(y[i]-np.exp(np.dot(self.teta,x[i])))*x[i]
            iter+=1
            #print(iter)
        return self.teta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.teta)) ## canonical response function 
        # *** END CODE HERE ***
