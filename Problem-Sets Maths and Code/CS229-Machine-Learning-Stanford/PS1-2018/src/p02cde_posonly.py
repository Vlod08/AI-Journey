import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'
basedir = 'E:/AI/Problem-Sets Maths and Code/CS229-Machine-Learning-Stanford/PS1-2018/'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    pred_path_c_png = pred_path_c.replace('txt', 'png')
    pred_path_d_png = pred_path_d.replace('txt', 'png')
    pred_path_e_png = pred_path_e.replace('txt', 'png')


    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train,t_train = util.load_dataset(train_path,label_col='t',add_intercept=True)
    x_test,t_test = util.load_dataset(test_path,label_col='t',add_intercept=True)

    #print(x_train[:5])
    #print(t_train[:5])
    LRc = LogisticRegression()
    LRc.fit(x=x_train,y=t_train)
    prediction = LRc.predict(x_test)
    pred_path_c_file = open(pred_path_c,'w')
    pred_path_c_file.write( f" prediction :  t_test \n" )
    for i in range(prediction.shape[0]):
        pred_path_c_file.write( f" {prediction[i] } : { t_test[i] } \n" )
    util.plot(x_test,t_test,LRc.teta,pred_path_c_png)


    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train,y_train = util.load_dataset(train_path,label_col='y',add_intercept=True)
    x_test,y_test = util.load_dataset(test_path,label_col='y',add_intercept=True)

    LRd = LogisticRegression()
    LRd.fit(x=x_train,y=y_train)
    prediction = LRd.predict(x_test)
    pred_path_d_file = open(pred_path_d,'w')
    pred_path_d_file.write( f" prediction :  y_test \n" )
    for i in range(prediction.shape[0]):
        pred_path_d_file.write( f" {prediction[i] } : { y_test[i] } \n" )
    util.plot(x_test,y_test,LRd.teta,pred_path_d_png)

    

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    



    # *** END CODER HERE
