class RegressionLoss:
    '''
    A class containing regression loss functions.
    '''

    @staticmethod
    def squared_error(y_i, a_i):
        '''
        Brief: Compute the squared error loss between true 
        values and predictions.

        Arguments:
        * y_i (float): True target value.
        * a_i (float): Predicted value.

        Returns:
        float: Squared error loss.
        '''
        return (y_i - a_i) ** 2 

    @staticmethod
    def absolute_error(y_i, a_i):
        '''
        Brief: Compute the absolute error loss between true 
        values and predictions.

        Arguments:
        * y_i (float): True target value.
        * a_i (float): Predicted value.

        Returns:
        float: Absolute error loss.
        '''
        return np.abs(y_i - a_i)

    @staticmethod
    def huber(y_i, a_i, threshold):
        '''
        Brief: Compute the Huber loss between true values and
        predictions with a specified threshold.

        Arguments:
        * y_i (float): True target value.
        * a_i (float): Predicted value.
        * threshold (float): Threshold for switching between 
        squared error and absolute error loss.

        Returns:
        float: Huber loss.
        '''
        if absolute_error(y_i, a_i) <= threshold:
            return 0.5 * squared_error(y_i, a_i)
        else:
            return (threshold * absolute_error(y_i, a_i)) - (0.5 * threshold ** 2)


class ClassificationLoss:
    '''
    A class containing classification loss functions.
    '''

    @staticmethod
    def log_loss(y_i, a_i):
        '''
        Brief: Compute the binary log loss between true labels
        and predicted probabilities.

        Arguments:
        * y_i (float): True binary label (0 or 1).
        * a_i (float): Predicted probability of class 1.

        Returns:
        float: Binary log loss.
        '''
        return - ((y_i * np.log(a_i)) + ((1 - y_i) * np.log(1 - a_i)))

    @staticmethod
    def categorical_cross_entropy(y_i, a_i):
        '''
        Brief: Compute the categorical cross-entropy loss between
        true class distributions and predicted class probabilities.

        Arguments:
        * y_i (array-like): True class distribution.
        * a_i (array-like): Predicted class probabilities.

        Returns:
        float: Categorical cross-entropy loss.
        '''
        return np.sum(y_i * np.log(a_i))

    @staticmethod
    def sparse_categorical_loss(y_i_index, a_i):
        '''
        Brief: Compute the sparse categorical cross-entropy loss 
        for a specific class index and predicted class probabilities.

        Arguments:
        * y_i_index (int): Index of the true class.
        * a_i (array-like): Predicted class probabilities.

        Returns:
        float: Sparse categorical cross-entropy loss for the specified class.
        '''
        a_i /= np.sum(a_i)
        return - np.log(a_i[y_i_index])
