import numpy as np

def present_scores( s , algorithm='method' ):
        print(30*'-')
        print( algorithm + ' accuracy in 10-fold cross validation:' )
        print('mean: ' + str( np.mean(s) ))
        print('std: ' + str( np.std(s) ))
        print('median: ' + str( np.median(s) ))

def binary_accuracy( y_true , y_pred ):
    bin_pred = np.array( y_pred >= 0.5 ).astype(int)
    return np.sum( y_true == bin_pred ) / y_true.size