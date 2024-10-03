import numpy as np
import util
from os import path
from linear_model import LinearModel

def sigmoid(z):
    return 1/(1+np.exp(-z))

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***

    GDA1 = GDA()
    GDA1.fit(x_train,y_train)
    
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    pred_path = open(pred_path, 'w')
    
    prediction = GDA1.predict(x_eval)

    for i in range(y_eval.shape[0]):
        pred_path.write( f" {prediction[i] } : { y_eval[i] } \n" )

    util.plot(x_eval, y_eval, GDA1.teta_final , 'E:/AI/Problem-Sets Maths and Code/CS229-Machine-Learning-Stanford/PS1-2018/output/p01e_' + path.basename(eval_path) + '.png')

    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        x = np.array(x)
        y = np.array(y)
        m,n = np.shape(x)
        self.teta = 0 
        self.mu0 = np.zeros(n) 
        self.mu1 = np.zeros(n)
        self.sigma = np.zeros((n,n))  
        for i in range(m):
            if(int(y[i]) == 0 ):
                self.mu0 += x[i]
            elif(int(y[i]) == 1):
                self.mu1 += x[i]
                self.teta += 1
        
        self.mu0 /= (m - self.teta)
        self.mu1 /= self.teta
        self.teta /= m 

        for i in range(m):
            if(y[i] == 0 ):
                self.sigma += np.outer((x[i] - self.mu0),(x[i] - self.mu0))
            elif(y[i] == 1):
                self.sigma += np.outer((x[i] - self.mu1),(x[i] - self.mu1))
        self.sigma /= m
        return [self.teta, self.mu0, self.mu1, self.sigma]
    
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        inv_sigma = np.linalg.inv(self.sigma)
        O   = np.dot(inv_sigma,(self.mu1 - self.mu0)) 
        Oo = np.log(self.teta/(1-self.teta)) + ( 0.5 * ( np.dot(np.dot(np.transpose(self.mu0 + self.mu1),inv_sigma), (self.mu0 - self.mu1 ) ) ) )
        self.teta_final=np.insert(O,0,Oo)
        return sigmoid( np.dot(x, O) + Oo)
        # *** END CODE HERE
