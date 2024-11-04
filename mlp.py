'''
Constructs, trains, tests 2 layer multilayer layer perceptron networks
Katie Bernard
'''
import numpy as np


class MLP:
    '''MLP is a class for multilayer perceptron network.

    The structure of the MLP will be:

    Input layer (X units) ->
    Hidden layer (Y units) with Rectified Linear activation (ReLu) ->
    Output layer (Z units) with softmax activation

    Due to the softmax, activation of output neuron i represents the probability that the current input sample belongs
    to class i.
    '''
    def __init__(self, num_input_units, num_hidden_units, num_output_units):
        '''Constructor to build the model structure and intialize the weights. There are 3 layers:
        input layer, hidden layer, and output layer. Since the input layer represents each input
        sample, we don't learn weights for it.

        Parameters:
        -----------
        num_input_units: int. Num input features
        num_hidden_units: int. Num hidden units
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units

        self.initialize_wts(num_input_units, num_hidden_units, num_output_units)

    def get_y_wts(self):
        '''Returns a copy of the hidden layer wts'''
        return self.y_wts.copy()

    def initialize_wts(self, M, H, C, std=0.1, r_seed=None):
        ''' Randomly initialize the hidden and output layer weights and bias term

        Parameters:
        -----------
        M: int. Num input features
        H: int. Num hidden units
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights
        r_seed: None or int. Random seed for weight and bias initialization.

        Returns:
        -----------
        No return
        '''
        rng = np.random.default_rng(r_seed)
        
        #hidden
        self.y_wts = rng.normal(0, std, (M, H))
        self.y_b = rng.normal(0, std, (H,))
        
        #output
        self.z_wts = rng.normal(0, std, (H, C))  
        self.z_b = rng.normal(0, std, (C,)) 

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
        return np.mean(y == y_pred)

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
        '''
        N = y.shape[0] #num of samps
        y_one_hot = np.zeros((N, num_classes))
        y_one_hot[np.arange(N), y] = 1
        return y_one_hot

    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
        '''
        #HIDDEN LAYER
        z_hidden = np.dot(features, self.y_wts) + self.y_b
        hidden_activations = np.maximum(0, z_hidden) #reLu acts
        
        #OUTPUT LAYER
        z_output = np.dot(hidden_activations, self.z_wts) + self.z_b
        y_pred = np.argmax(z_output, axis=1) #index of max act
        return y_pred

    def forward(self, features, y, reg=0):
        '''Performs a forward pass of the net (input -> hidden -> output).
        This should start with the features and progate the activity to the output layer, ending with the cross-entropy
        loss computation.

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size N, Num features M)
        y: ndarray. int coded class labels. shape=(mini-batch-size N,)
        reg: float. regularization strength.

        Returns:
        -----------
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        loss: float. REGULARIZED loss derived from output layer, averaged over all input samples

        '''
        #HIDDEN LAYER
        y_net_in = np.dot(features, self.y_wts) + self.y_b
        y_net_act = np.maximum(0, y_net_in)  #relu acts

        # OUTPUT LAYER
        z_net_in = np.dot(y_net_act, self.z_wts) + self.z_b
        norm_ins = z_net_in - np.max(z_net_in, axis=1, keepdims=True)
        z_net_act = np.exp(norm_ins) / np.sum(np.exp(norm_ins), axis=1, keepdims=True)

        #cross entropy loss
        N = features.shape[0]
        
        correct_log_probs = -np.log(z_net_act[range(N), y])
        cross_e_loss = np.sum(correct_log_probs) / N

        #regularization 
        reg_loss = 0.5 * reg * (np.sum(self.y_wts ** 2) + np.sum(self.z_wts ** 2))
        loss = cross_e_loss + reg_loss

        return y_net_in, y_net_act, z_net_in, z_net_act, loss

    def backward(self, features, y, y_net_in, y_net_act, z_net_in, z_net_act, reg=0):
        '''Performs a backward pass (output -> hidden -> input) during training to update the weights. This function
        implements the backpropogation algorithm.

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        y: ndarray. int coded class labels. shape=(mini-batch-size,)
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        reg: float. regularization strength.

        Returns:
        -----------
        dy_wts, dy_b, dz_wts, dz_b: The following backwards gradients
        (1) hidden wts, (2) hidden bias, (3) output weights, (4) output bias
        Shapes should match the respective wt/bias instance vars.

        '''
        y_one_hot = self.one_hot(y,self.num_output_units) 
        dz_netAct = -1/(features.shape[0]*z_net_act)
        dz_netIn = dz_netAct * (z_net_act * (y_one_hot - z_net_act))
        dz_wts = (dz_netIn.T @ y_net_act).T
        dz_b = np.sum(dz_netIn, axis = 0)
        dy_netAct = dz_netIn @ self.z_wts.T
        dy_netIn = dy_netAct * np.maximum(0, np.sign(y_net_in))
        dy_wts = (dy_netIn.T @ features).T
        dy_b = np.sum(dy_netIn, axis = 0)

        # Add regularization to hidden weights
        dy_wts += reg * self.y_wts
        dz_wts += reg * self.z_wts

        return dy_wts, dy_b, dz_wts, dz_b




    def fit(self, features, y, x_validation, y_validation, n_epochs=500, lr=0.0001, mini_batch_sz=256, reg=0,
            r_seed=None, verbose=2, print_every=100):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features).
            Features over N inputs.
        y: ndarray.
            int-coded class assignments of training samples. 0,...,numClasses-1
        x_validation: ndarray. shape=(Num samples in validation set, num features).
            This is used for computing/printing the accuracy on the validation set at the end of each epoch.
        y_validation: ndarray.
            int-coded class assignments of validation samples. 0,...,numClasses-1
        n_epochs: int.
            Number of training epochs
        lr: float.
            Learning rate
        mini_batch_sz: int.
            Batch size per epoch. i.e. How many samples we draw from features to pass through the model per training epoch
            before we do gradient descent and update the wts.
        reg: float.
            Regularization strength used when computing the loss and gradient.
        r_seed: None or int.
            Random seed for weight and bias initialization.
        verbose: int.
            0 means no print outs. Any value > 0 prints Current epoch number and training loss every
            `print_every` (e.g. 100) epochs.
        print_every: int.
            If verbose > 0, print out the training loss and validation accuracy over the last epoch
            every `print_every` epochs.
            Example: If there are 20 epochs and `print_every` = 5 then you print-outs happen on
            after completing epochs 0, 5, 10, and 15 (or 1, 6, 11, and 16 if counting from 1).

        Returns:
        -----------
        loss_history: Python list of floats. len=`n_epochs * n_iter_per_epoch`.
            Recorded training loss for each mini-batch of training.
        train_acc_history: Python list of floats. len=`n_epochs`.
            Recorded accuracy on every epoch on the training set.
        validation_acc_history: Python list of floats. len=`n_epochs`.
            Recorded accuracy on every epoch on the validation set.
        '''

        rng = np.random.default_rng(r_seed)        
        self.wts = rng.normal(loc=0, scale=0.01, size=(features.shape[1],self.num_output_units))
        self.b = rng.normal(loc=0, scale=0.01, size=(self.num_output_units,))
        loss_history = []
        train_acc_history = []
        validation_acc_history = []

        N, M = features.shape

        ### training loop
        for epoch in range(n_epochs):
            
            # mini-batches
            for i in range(0, N, mini_batch_sz):
                # shuffle/select the mini-batch
                indices = rng.integers(low=0, high=N, size=(mini_batch_sz,))
                batch_features = features[indices]
                batch_labels = y[indices]
                
                #forward pass
                y_net_in, y_net_act, z_net_in, z_net_act, loss = self.forward(batch_features, batch_labels, reg)
                loss_history.append(loss)

                #backward pass 
                dy_wts, dy_b, dz_wts, dz_b = self.backward(batch_features, batch_labels, y_net_in, y_net_act, z_net_in, z_net_act, reg)

                # update wts and bias
                self.y_wts -= lr * dy_wts
                self.y_b -= lr * dy_b
                self.z_wts -= lr * dz_wts
                self.z_b -= lr * dz_b

            # calc and print acc after each epoch
            train_acc = self.accuracy(y, self.predict(features))
            train_acc_history.append(train_acc)
            val_acc = self.accuracy(y_validation, self.predict(x_validation))
            validation_acc_history.append(val_acc)

            if verbose > 0 and epoch % print_every == 0:
                print(f'Epoch {epoch}/{n_epochs}: Loss = {loss:.4f}, Train Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}')

        if verbose > 0:
            print('Training complete.')
            
        return loss_history, train_acc_history, validation_acc_history

