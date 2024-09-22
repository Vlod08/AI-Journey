import numpy as np
import util
import matplotlib.pyplot as plt
from os import path
    # *** START CODE HERE ***
 
    ## helper functions  ###
def sigmoid(z):
    return 1/(1+np.exp(-z))

def get_dataset_name(str_path):
    path.basename(str_path)

    # *** END CODE HERE ***

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    LR1 = LogisticRegression()
    teta = LR1.fit(x_train,y_train)
    
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    pred_path = open(pred_path, 'w')
    prediction = LR1.predict(x_eval)
    for i in range(y_eval.shape[0]):
        pred_path.write( f" {prediction[i] } : { y_eval[i] } \n" )

    fig = plt.figure()

    util.plot(x_eval, y_eval, LR1.teta, 'E:/AI/Problem-Sets Maths and Code/CS229-Machine-Learning-Stanford/PS1-2018/output/p01b_' + path.basename(eval_path) + '.png')

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        epsilon = 1e-5
 

        m,n = np.shape(x)
        self.teta = np.zeros(n)
        teta_old = np.ones(n)*np.inf
        
        while True:

            hox  = sigmoid(np.matmul(x,self.teta)) 

            #JO = np.sum(y * np.log(hox) + (1 - y) * np.log(1 - hox)) / m
        
            DJDO = np.matmul(x.T,hox-y)/m

            H = (1/m) * np.matmul((x.T * (hox * (1-hox))),x)

            teta_old = np.copy(self.teta) 

            self.teta -= np.matmul(DJDO,np.linalg.inv(H))

            
            if (np.linalg.norm(teta_old-self.teta,ord=1) <= epsilon ):
                break

        return self.teta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return np.round(sigmoid(np.dot(x, self.teta)))
        
        # *** END CODE HERE ***
