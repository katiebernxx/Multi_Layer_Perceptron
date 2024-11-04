'''
Constructs, trains, tests single layer neural network with softmax activation function.
Katie Bernard
'''
import numpy as np


class SoftmaxLayer:
    '''SoftmaxLayer is a class for single layer networks with softmax activation and cross-entropy loss
    in the output layer.
    '''
    def __init__(self, num_output_units):
        '''SoftmaxLayer constructor

        Parameters:
        -----------
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        # Network weights
        self.wts = None
        # Bias
        self.b = None
        # Number of data classes C
        self.num_output_units = num_output_units

    def accuracy(self, y, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        acc = ((y_pred == y).sum())/len(y)
        return acc

    def net_in(self, features):
        '''Computes the net input (net weighted sum)

        Parameters:
        -----------
        features: ndarray. input data. shape=(num images (in mini-batch), num features)
        i.e. shape=(N, M)

        Note: shape of self.wts = (M, C), for C output neurons

        Returns:
        -----------
        net_input: ndarray. shape=(N, C)
        '''
        net_in = features @ self.wts + self.b
        return net_in

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,C-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
        '''
        y_one_hot = np.zeros((len(y), num_classes))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot

        

    def fit(self, features, y, n_epochs=100, lr=0.0001, mini_batch_sz=256, reg=0, r_seed=None, verbose=2):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features M)
        y: ndarray. int-coded class assignments of training samples. 0,...,numClasses-1
        n_epochs: int. Number of training epochs
        lr: float. Learning rate
        mini_batch_sz: int. Batch size per training iteration.
            i.e. Chunk this many data samples together to process with the model on each training
            iteration. Then we do gradient descent and update the wts. NOT the same thing as an epoch.
        reg: float. Regularization strength used when computing the loss and gradient.
        r_seed: None or int. Random seed for weight and bias initialization.
        verbose: int. 0 means no print outs. Any value > 0 prints Current iteration number and
            training loss every 100 iterations.

        Returns:
        -----------
        loss_history: Python list of floats. Recorded training loss on every mini-batch / training
            iteration.
        '''
        rng = np.random.default_rng(r_seed)        

        self.wts = rng.normal(loc=0, scale=0.01, size=(features.shape[1],self.num_output_units))

        self.b = rng.normal(loc=0, scale=0.01, size=(self.num_output_units,))

        loss_history = []

        N, M = features.shape

        ### training loop
        iter = 0
        for epoch in range(n_epochs):

            # mini-batches
            for i in range(0, N, mini_batch_sz):
                indices = rng.integers(low=0, high=N, size=(mini_batch_sz,))
                batch_features = features[indices]
                batch_labels = y[indices]
              
                batch_one_hot_labels = self.one_hot(batch_labels,self.num_output_units)

                # net in 
                net_in = self.net_in(batch_features)

                # net act
                net_act = self.activation(net_in)

                # loss
                loss = self.loss(net_act, batch_labels, reg)
                loss_history.append(loss)


                # backprop update wts
                grad_wts, grad_b = self.gradient(batch_features, net_act, batch_one_hot_labels, reg)

                # update wts and bias
                self.wts -= lr * grad_wts
                self.b -= lr * grad_b

                if verbose > 0 and (iter % 100) == 0:
                    print(f" Iter {iter}, Loss: {loss:.4f}")

                iter += 1

        return loss_history



    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments from net_in (i.e. you dont
            need to apply the net activation function — it will not affect the most active neuron).
        '''
        net_in = features @ self.wts + self.b
        pred = np.argmax(net_in, axis =1)
        return pred

    def activation(self, net_in):
        '''Applies the softmax activation function on the net_in.

        Parameters:
        -----------
        net_in: ndarray. net in. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)

        Returns:
        -----------
        f_z: ndarray. net_act transformed by softmax function. shape=(N, C)

        '''
        net_in_adjusted = net_in - np.max(net_in, axis=1, keepdims=True)

        #net_act
        f_z = np.exp(net_in_adjusted) / np.sum(np.exp(net_in_adjusted), axis=1, keepdims=True)
        
        return f_z

    def loss(self, net_act, y, reg=0):
        '''Computes the cross-entropy loss

        Parameters:
        -----------
        net_act: ndarray. softmax net activation. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)
        y: ndarray. correct class values, int-coded. shape=(mini-batch size,)
        reg: float. Regularization strength

        Returns:
        -----------
        loss: float. Regularized (!!!!) average loss over the mini batch

        '''

        #correct classes in net_act
        correct_classes = net_act[np.arange(net_act.shape[0]), y]

        #cross-entropy loss
        ce_loss = -np.mean(np.log(correct_classes))

        #regularization term
        reg_loss = 0.5 * reg * np.sum(self.wts ** 2)

        #total loss
        total_loss = ce_loss + reg_loss

        #return loss 
        return total_loss
    

    def gradient(self, features, net_act, y, reg=0):
        '''Computes the gradient of the softmax version of the net

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        net_act: ndarray. net outputs. shape=(mini-batch-size, C)
            In the softmax network, net_act for each input has the interpretation that
            it is a probability that the input belongs to each of the C output classes.
        y: ndarray. one-hot coded class labels. shape=(mini-batch-size, Num output neurons)
        reg: float. regularization strength.

        Returns:
        -----------
        grad_wts: ndarray. Weight gradient. shape=(Num features, C)
        grad_b: ndarray. Bias gradient. shape=(C,)

        '''
       
        #grad wts
        grad_wts = (features.T @ (net_act - y)) / features.shape[0]

        #add regularization term
        grad_wts += reg * self.wts

        #grad bias
        grad_b = np.mean((net_act - y), axis = 0)

        return grad_wts, grad_b



    def test_loss(self, wts, b, features, labels):
        ''' Tester method for net_in and loss
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in shape={net_in.shape}, min={net_in.min()}, max={net_in.max()}')

        net_act = self.activation(net_in)
        print(f'net act shape={net_act.shape}, min={net_act.min()}, max={net_act.max()}')
        return self.loss(net_act, labels, 0), self.loss(net_act, labels, 0.5)

    def test_gradient(self, wts, b, features, labels, num_unique_classes, reg=0):
        ''' Tester method for gradient
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in: {net_in.shape}, {net_in.min()}, {net_in.max()}')
        print(f'net in 1st few values of 1st input are:\n{net_in[0, :5]}')

        net_act = self.activation(net_in)
        print(f'net act 1st few values of 1st input are:\n{net_act[0, :5]}')

        labels_one_hot = self.one_hot(labels, num_unique_classes)
        print(f'y one hot: {labels_one_hot.shape}, sum is {np.sum(labels_one_hot)}.')

        return self.gradient(features, net_act, labels_one_hot, reg=reg)
