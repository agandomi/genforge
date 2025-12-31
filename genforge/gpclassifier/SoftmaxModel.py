# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize
defaults = {
    'xtrain': None,
    'ytrain': None,
    'yonehot_train': None,
    'xval': None,
    'yval': None,
    'yonehot_val': None,
    'num_class': None,
    'optimizer': 'adam',
    'optimizer_param': {},
    'initializer': 'glorot_uniform',
    'regularization': None,
    'regularization_rate': 0.01,
    'batch_size': 32,
    'epochs': 10,
    'patience': 10,
    'random_seed': None,
    'buffer_size': None,
    'shuffle': True,
    'verbose': 1,
    }

class SoftmaxModel:
    def __init__(self, **params):
        self.params = {**defaults, **params}
        self.rng = []
        self.config()
        self.weight = []
        self.bias = []
        self.initialize_weight()
    #### __init__ 
    ##############################################################################
    #### compiler     
    @classmethod
    def compiler(cls, **params):
        return cls(**params)
    #### End of compiler
    ##############################################################################
    #### config
    def config(self):
        random_seed = self.params['random_seed']
        # --- RNG unification ---
        if random_seed is None:
            self.rng = np.random.default_rng()                 # entropy-based new generator
        elif isinstance(random_seed, (int, np.integer)):
            self.rng = np.random.default_rng(int(random_seed)) # reproducible generator
        elif isinstance(random_seed, np.random.Generator):
            self.rng = random_seed                             # use as-is (shared rng)
        else:
            raise TypeError(
                f"random_seed must be None, int, or np.random.Generator; got {type(random_seed)}"
            )
        
        optim_par = {}
        if self.params['optimizer'] == 'sgd':
            # Stochastic Gradient Descent
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.01)
            optim_par['momentum'] = self.params['optimizer_param'].get('momentum', 0.0)  # Default is 0 if not provided
            optim_par['nesterov'] = self.params['optimizer_param'].get('nesterov', False)  # Use Nesterov momentum if True
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)  # L2 regularization
            optim_par['dampening'] = self.params['optimizer_param'].get('dampening', 0.0)  # Dampening for momentum
            optim_par['decay'] = self.params['optimizer_param'].get('decay', 0.0)  # Learning rate decay
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)  # Global norm clipping
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)  # Value clipping
            #####################################
        elif self.params['optimizer'] == 'adam':
            # Adam
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.001)
            optim_par['beta1'] = self.params['optimizer_param'].get('beta1', 0.9)
            optim_par['beta2'] = self.params['optimizer_param'].get('beta2', 0.999)
            optim_par['epsilon'] = self.params['optimizer_param'].get('epsilon', 1e-7)
            optim_par['amsgrad'] = self.params['optimizer_param'].get('amsgrad', False) 
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'rmsprop':
            # RMSprop
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.001)
            optim_par['rho'] = self.params['optimizer_param'].get('rho', 0.9)
            optim_par['momentum'] = self.params['optimizer_param'].get('momentum', 0)
            optim_par['epsilon'] = self.params['optimizer_param'].get('epsilon', 1e-7)
            optim_par['centered'] = self.params['optimizer_param'].get('centered', False) 
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'adamw':
            # adamW
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.001)
            optim_par['beta1'] = self.params['optimizer_param'].get('beta1', 0.9)
            optim_par['beta2'] = self.params['optimizer_param'].get('beta2', 0.999)
            optim_par['epsilon'] = self.params['optimizer_param'].get('epsilon', 1e-8)
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.01)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'nadam':
            # Nesterov-accelerated Adaptive Moment Estimation
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.002)
            optim_par['beta1'] = self.params['optimizer_param'].get('beta1', 0.9)
            optim_par['beta2'] = self.params['optimizer_param'].get('beta2', 0.999)
            optim_par['epsilon'] = self.params['optimizer_param'].get('epsilon', 1e-7)
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'adagrad':
            # Adaptive Gradient Algorithm
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.01)
            optim_par['epsilon'] = self.params['optimizer_param'].get('epsilon', 1e-7)
            optim_par['initial_accumulator_value'] = self.params['optimizer_param'].get('initial_accumulator_value', 0.1) 
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'adadelta':
            # Adadelta
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 1)
            optim_par['rho'] = self.params['optimizer_param'].get('rho', 0.95)
            optim_par['epsilon'] = self.params['optimizer_param'].get('epsilon', 1e-7)
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'ftrl':
            # Follow-the-Regularized-Leader
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.001)
            optim_par['learning_rate_power'] = self.params['optimizer_param'].get('learning_rate_power', -0.5)
            optim_par['initial_accumulator_value'] = self.params['optimizer_param'].get('initial_accumulator_value', 0.1)
            optim_par['l1_regularization_strength'] = self.params['optimizer_param'].get('l1_regularization_strength', 0)
            optim_par['l2_regularization_strength'] = self.params['optimizer_param'].get('l2_regularization_strength', 0)
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'adamax':
            # AdaMax
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.002)
            optim_par['beta1'] = self.params['optimizer_param'].get('beta1', 0.9)
            optim_par['beta2'] = self.params['optimizer_param'].get('beta2', 0.999)
            optim_par['epsilon'] = self.params['optimizer_param'].get('epsilon', 1e-7)
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'sgdnm':
            # SGD with Nesterov Momentum
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 0.01)
            optim_par['momentum'] = self.params['optimizer_param'].get('momentum', 0.9)
            optim_par['nesterov'] = self.params['optimizer_param'].get('nesterov', True)
            optim_par['weight_decay'] = self.params['optimizer_param'].get('weight_decay', 0.0)
            optim_par['dampening'] = self.params['optimizer_param'].get('dampening', 0.0)
            optim_par['decay'] = self.params['optimizer_param'].get('decay', 0.0)
            optim_par['clipnorm'] = self.params['optimizer_param'].get('clipnorm', None)
            optim_par['clipvalue'] = self.params['optimizer_param'].get('clipvalue', None)
            #######################################
        elif self.params['optimizer'] == 'lbgfs':
            # Limited-memory Broyden–Fletcher–Goldfarb–Shanno
            optim_par['learning_rate'] = self.params['optimizer_param'].get('learning_rate', 1)
            optim_par['max_iter'] = self.params['optimizer_param'].get('max_iter', 20)
            optim_par['max_eval'] = self.params['optimizer_param'].get('max_eval', None)
            optim_par['tolerance_grad'] = self.params['optimizer_param'].get('tolerance_grad', 1e-5)
            optim_par['tolerance_change'] = self.params['optimizer_param'].get('tolerance_change', 1e-9)
            optim_par['history_size'] = self.params['optimizer_param'].get('history_size', 100)
            optim_par['line_search_fn'] = self.params['optimizer_param'].get('line_search_fn', 'strong_wolfe')
            #######################################
            
        self.params['optimizer_param'] = optim_par
    #### End of config
    ##############################################################################
    #### initialize_weight 
    def initialize_weight(self):
        """Initialize weights based on the given initializer."""
        input_dim = self.params['xtrain'].shape[1]
        output_dim = self.params['num_class']
        initializer = self.params['initializer']
        rng = self.rng
        
            
        # Initialize weights
        if initializer == 'he_normal':
            weights = rng.standard_normal(size=(output_dim, input_dim)) * np.sqrt(2.0 / input_dim)
            biases = rng.standard_normal(size=(output_dim, 1)) * np.sqrt(2.0 / input_dim)  # He initialization for biases
        elif initializer == 'random_normal':
            weights = rng.standard_normal(size=(output_dim, input_dim))
            biases = rng.standard_normal(size=(output_dim, 1))
        elif initializer == 'glorot_uniform':
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            weights = rng.uniform(-limit, limit, size=(output_dim, input_dim))
            biases = rng.uniform(-limit, limit, size=(output_dim, 1))  # Glorot uniform for biases
        else:
            weights = rng.standard_normal(size=(output_dim, input_dim))
            biases = np.zeros((output_dim, 1))  # Default zero initialization for biases
        
        
        self.weight = weights
        self.bias = biases
    #### End of initialize_weight
    ##############################################################################
    #### batcher    
    def batcher(self, X, Y=None, batch_size=32, shuffle=False, buffer_size=None):
        n_samples = X.shape[0]
        X_batch = []
        Y_batch = []
        index_batch = []  # List to store the positional indices of instances in each batch
        rng = self.rng
        
        
        # Shuffle the dataset with the specified buffer size
        indices = np.arange(n_samples)
        
        if shuffle and (buffer_size is None or buffer_size >= n_samples):
            # If no buffer size or buffer size is greater than the dataset, shuffle the entire dataset
            rng.shuffle(indices)
        elif shuffle:
            # Shuffle with a buffer of size `buffer_size`
            for i in range(0, n_samples, buffer_size):
                end = min(i + buffer_size, n_samples)
                buffer_indices = indices[i:end]
                rng.shuffle(buffer_indices)  # Shuffle only within the buffer
                indices[i:end] = buffer_indices
        
        # Apply the shuffled indices to the dataset
        X = X[indices, :]
        if Y is not None:
            Y = Y[indices, :]
        
        # Create batches
        for i in range(0, n_samples, batch_size):
            X_batch.append(X[i:i + batch_size, :])
            if Y is not None:
                Y_batch.append(Y[i:i + batch_size, :])
            index_batch.append(indices[i:i + batch_size])  # Append the corresponding indices
        
        
        return X_batch, Y_batch, index_batch  # Return the list of positional indices as well
    #### End of batcher
    #######################################################################################
    #### logit
    def logit(self, X):
        z1 = np.dot(self.weight, X.T) + self.bias
        z = z1.T
        return z
    #### End of logit
    #######################################################################################
    #### softmaxfun
    def softmaxfun(self, z):
        """Softmax activation function."""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Stability improvement
        prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return prob
    #### End of softmaxfun
    #######################################################################################
    #### predict
    def predict(self, X, batch_size = 32, index_batch = None, Training = False):
        if Training:
            z = self.logit(X)
            prob = self.softmaxfun(z)
        elif index_batch is not None:
            len_all = 0
            for i in range(len(X)):
                len_all += X[i].shape[0]
            prob = np.zeros((len_all, self.weight.shape[0]))
            for i in range(len(X)):
                z = self.logit(X[i])
                prob[index_batch[i], :] = self.softmaxfun(z)
        else:
            x_batch, _, index_batch = self.batcher(X, batch_size = batch_size)
            prob = np.zeros((X.shape[0], self.weight.shape[0]))
            for i in range(len(x_batch)):
                z = self.logit(x_batch[i])
                prob[index_batch[i], :] = self.softmaxfun(z)
        return prob
    #### End of predict
    #######################################################################################
    #### loss
    def loss(self, prob, Y):
        log_likelihood = -np.sum(Y * np.log(prob + 1e-9), axis=1)
        loss = np.mean(log_likelihood)
        return loss
    #### End of loss
    #######################################################################################
    #### Regularization loss
    def regularization_loss(self):
        if self.params['regularization'] == 'l1':
            # L1 regularization (Lasso)
            reg_loss = self.params['regularization_rate'] * np.sum(np.abs(self.weight))
        elif self.params['regularization'] == 'l2':
            # L2 regularization (Ridge)
            reg_loss = self.params['regularization_rate'] * np.sum(self.weight ** 2)
        elif self.params['regularization'] == 'hybrid':
            # Hybrid L1 and L2 regularization
            reg_loss = self.params['regularization_rate'][0] * np.sum(np.abs(self.weight)) +\
                self.params['regularization_rate'][1] * np.sum(self.weight ** 2)
        else:
            reg_loss = 0
        return reg_loss
    #### End of Regularization loss
    #######################################################################################
    #### Regularization gradient
    def regularization_gradient(self, batch_size):
        """
        Compute the gradient of the regularization term with respect to the weights.
        """
        reg_grad = np.zeros_like(self.weight)
        if self.params['regularization'] == 'l1':
            # L1 Regularization Gradient (Lasso)
            reg_grad = (self.params['regularization_rate'] * np.sign(self.weight)) / batch_size
        elif self.params['regularization'] == 'l2':
            # L2 Regularization Gradient (Ridge)
            reg_grad = (2 * self.params['regularization_rate'] * self.weight) / batch_size
        elif self.params['regularization'] == 'hybrid':
            # Hybrid Regularization Gradient (L1 + L2)
            l1_rate = self.params['regularization_rate'][0]
            l2_rate = self.params['regularization_rate'][1]
            reg_grad = (l1_rate * np.sign(self.weight) + 2 * l2_rate * self.weight) / batch_size
        else:
            # No Regularization
            reg_grad = np.zeros_like(self.weight)  # No regularization applied
        return reg_grad

    #### End of Regularization gradient
    #######################################################################################
    #### sgd
    def sgd(self, dw, db, v_dw, v_db, epoch):
        """
        Stochastic Gradient Descent with Momentum and Nesterov acceleration (if enabled).
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            v_dw: Velocity term for weights.
            v_db: Velocity term for biases.
            epoch: Current epoch for learning rate decay.
            
        Returns:
            Updated velocity terms for the next iteration.
        """
        # Retrieve optimizer parameters
        learning_rate = self.params['optimizer_param']['learning_rate']
        momentum = self.params['optimizer_param']['momentum']
        nesterov = self.params['optimizer_param']['nesterov']
        weight_decay = self.params['optimizer_param']['weight_decay']
        dampening = self.params['optimizer_param']['dampening']
        decay = self.params['optimizer_param']['decay']
        clipnorm = self.params['optimizer_param']['clipnorm']
        clipvalue = self.params['optimizer_param']['clipvalue']
    
        # Apply gradient clipping by value
        if clipvalue is not None:
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
    
        # Apply gradient clipping by norm
        if clipnorm is not None:
            grad_norm_w = np.linalg.norm(dw)
            if grad_norm_w > clipnorm:
                dw = dw * clipnorm / grad_norm_w
            grad_norm_b = np.linalg.norm(db)
            if grad_norm_b > clipnorm:
                db = db * clipnorm / grad_norm_b
    
        # Update velocity for momentum (with optional dampening)
        v_dw = momentum * v_dw + (1 - dampening) * dw  # Dampening controls momentum contribution
        v_db = momentum * v_db + (1 - dampening) * db
    
        if nesterov:
            # Nesterov accelerated gradient
            self.weight -= learning_rate * (momentum * v_dw + (1 - dampening) * dw)
            self.bias -= learning_rate * (momentum * v_db + (1 - dampening) * db)
        else:
            # Standard momentum SGD
            self.weight -= learning_rate * v_dw
            self.bias -= learning_rate * v_db
    
        # Apply decoupled weight decay (L2 regularization) to the weights directly (not multiplied by learning rate)
        if weight_decay != 0:
            self.weight -= weight_decay * self.weight
    
        # Apply learning rate decay (if enabled)
        learning_rate *= (1. / (1. + decay * epoch))
        
        # Save updated learning rate back to parameters (if you want to track the decaying rate)
        self.params['optimizer_param']['learning_rate'] = learning_rate
    
        return v_dw, v_db
    #### End of sgd
    #######################################################################################
    #### adam
    def adam(self, dw, db, v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max, epoch):
        """
        Adam optimizer for updating weights and biases with all parameters, including amsgrad, weight decay, and gradient clipping.
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            m_dw: First moment (mean) of the weights gradient.
            m_db: First moment (mean) of the biases gradient.
            v_dw: Second moment (variance) of the weights gradient.
            v_db: Second moment (variance) of the biases gradient.
            v_dw_max: Max past squared gradients for weights (used in AMSGrad).
            v_db_max: Max past squared gradients for biases (used in AMSGrad).
            epoch: Current epoch for computing bias-corrected estimates.
        """
        # Retrieve Adam parameters from self.params
        learning_rate = self.params['optimizer_param']['learning_rate']
        beta1 = self.params['optimizer_param']['beta1']  # First moment decay
        beta2 = self.params['optimizer_param']['beta2']  # Second moment decay
        epsilon = self.params['optimizer_param']['epsilon']  # Small constant to prevent division by zero
        weight_decay = self.params['optimizer_param']['weight_decay']  # L2 regularization factor
        amsgrad = self.params['optimizer_param']['amsgrad']  # Boolean flag to use AMSGrad variant of Adam
        clipnorm = self.params['optimizer_param']['clipnorm']  # Clip gradients by global norm
        clipvalue = self.params['optimizer_param']['clipvalue']  # Clip gradients by value
        
        # Increment timestep (Adam uses epoch + 1 for time correction)
        t = epoch + 1
        
        # Update biased first moment estimate (m_dw and m_db)
        m_dw = beta1 * m_dw + (1 - beta1) * dw
        m_db = beta1 * m_db + (1 - beta1) * db
        
        # Update biased second moment estimate (v_dw and v_db)
        v_dw = beta2 * v_dw + (1 - beta2) * np.square(dw)
        v_db = beta2 * v_db + (1 - beta2) * np.square(db)
        
        # Compute bias-corrected first moment estimate (m_dw_corr and m_db_corr)
        m_dw_corr = m_dw / (1 - beta1 ** t)
        m_db_corr = m_db / (1 - beta1 ** t)
        
        # Compute bias-corrected second moment estimate (v_dw_corr and v_db_corr)
        v_dw_corr = v_dw / (1 - beta2 ** t)
        v_db_corr = v_db / (1 - beta2 ** t)
        
        # Apply AMSGrad variant if required (keep track of max past squared gradients)
        if amsgrad:
            v_dw_max = np.maximum(v_dw_max, v_dw_corr)
            v_db_max = np.maximum(v_db_max, v_db_corr)
            v_dw_corr = v_dw_max
            v_db_corr = v_db_max
        
        # Apply gradient clipping
        if clipnorm is not None:
            # Clip by global norm
            norm = np.sqrt(np.sum(np.square(dw)) + np.sum(np.square(db)))
            if norm > clipnorm:
                dw = dw * clipnorm / norm
                db = db * clipnorm / norm
        if clipvalue is not None:
            # Clip by value
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
        
        # Apply weight decay (decoupled from gradients)
        if weight_decay > 0:
            self.weight -= weight_decay * self.weight  # Decoupled weight decay, without learning_rate multiplication
        
        # Compute the updates for weights and biases (excluding weight decay)
        weight_update = learning_rate * (m_dw_corr / (np.sqrt(v_dw_corr) + epsilon))
        bias_update = learning_rate * (m_db_corr / (np.sqrt(v_db_corr) + epsilon))
        
        # Update weights and biases
        self.weight -= weight_update
        self.bias -= bias_update
        
        # Return updated momentum and variance terms for the next iteration
        return v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max
    #### End of adam
    #######################################################################################
    #### rmsprop
    def rmsprop(self, dw, db, v_dw, v_db, m_dw, m_db, mean_dw, mean_db, epoch):
        """
        RMSprop optimizer for updating weights and biases with all parameters, including
        weight decay, gradient clipping, and centered RMSprop, with momentum support.
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            v_dw: Exponentially weighted average of the squared gradient for weights.
            v_db: Exponentially weighted average of the squared gradient for biases.
            m_dw: Momentum term for weights.
            m_db: Momentum term for biases.
            mean_dw: Running mean of the gradients for weights (used in centered RMSprop).
            mean_db: Running mean of the gradients for biases (used in centered RMSprop).
            epoch: Current epoch for learning rate decay.
        """
        # Retrieve RMSprop parameters from self.params
        learning_rate = self.params['optimizer_param']['learning_rate']
        rho = self.params['optimizer_param']['rho']  # Decay rate for the moving average
        epsilon = self.params['optimizer_param']['epsilon']  # Small constant to prevent division by zero
        momentum = self.params['optimizer_param']['momentum']  # Momentum factor
        weight_decay = self.params['optimizer_param']['weight_decay']  # L2 regularization factor
        clipnorm = self.params['optimizer_param']['clipnorm']  # Clip gradients by global norm
        clipvalue = self.params['optimizer_param']['clipvalue']  # Clip gradients by value
        centered = self.params['optimizer_param']['centered']  # Use the centered variant if True
    
        # Update running average of squared gradient (v_dw and v_db)
        v_dw = rho * v_dw + (1 - rho) * np.square(dw)
        v_db = rho * v_db + (1 - rho) * np.square(db)
    
        # If centered RMSprop is used, compute the mean of the gradients (mean_dw and mean_db)
        if centered:
            mean_dw = rho * mean_dw + (1 - rho) * dw
            mean_db = rho * mean_db + (1 - rho) * db
            v_dw_corr = v_dw - np.square(mean_dw)
            v_db_corr = v_db - np.square(mean_db)
        else:
            v_dw_corr = v_dw
            v_db_corr = v_db
    
        # Apply momentum to the gradients (RMSprop with momentum)
        m_dw = momentum * m_dw + dw / (np.sqrt(v_dw_corr) + epsilon)
        m_db = momentum * m_db + db / (np.sqrt(v_db_corr) + epsilon)
    
        # Apply gradient clipping
        if clipnorm is not None:
            # Clip by global norm
            norm = np.sqrt(np.sum(np.square(dw)) + np.sum(np.square(db)))
            if norm > clipnorm:
                dw = dw * clipnorm / norm
                db = db * clipnorm / norm
        if clipvalue is not None:
            # Clip by value
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
    
        # Apply L2 weight decay separately (modern decoupled weight decay)
        if weight_decay > 0:
            self.weight -= weight_decay * self.weight  # No multiplication by learning_rate
    
        # Update weights and biases
        self.weight -= learning_rate * m_dw
        self.bias -= learning_rate * m_db
    
        # Return updated momentum and variance terms for the next iteration
        return v_dw, v_db, m_dw, m_db, mean_dw, mean_db
    #### End of rmsprop
    #######################################################################################
    #### adamw
    def adamw(self, dw, db, v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max, epoch):
        """
        AdamW optimizer for updating weights and biases with all parameters, including weight decay, gradient clipping, 
        and AMSGrad variant.
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            m_dw: First moment (mean) of the weights gradient.
            m_db: First moment (mean) of the biases gradient.
            v_dw: Second moment (variance) of the weights gradient.
            v_db: Second moment (variance) of the biases gradient.
            v_dw_max: Maximum of second moment for weights (used in AMSGrad).
            v_db_max: Maximum of second moment for biases (used in AMSGrad).
            epoch: Current epoch for bias correction in Adam.
            
        Returns:
            Updated first and second moments for next iteration.
        """
        # Retrieve AdamW parameters from self.params
        learning_rate = self.params['optimizer_param']['learning_rate']
        beta1 = self.params['optimizer_param']['beta1']  # First moment decay
        beta2 = self.params['optimizer_param']['beta2']  # Second moment decay
        epsilon = self.params['optimizer_param']['epsilon']  # Small constant to prevent division by zero
        weight_decay = self.params['optimizer_param']['weight_decay']  # L2 regularization factor
        clipnorm = self.params['optimizer_param']['clipnorm']  # Clip gradients by global norm
        clipvalue = self.params['optimizer_param']['clipvalue']  # Clip gradients by value
        
        # Increment timestep (Adam uses epoch + 1 for bias correction)
        t = epoch + 1
        
        # Update biased first moment estimate (m_dw and m_db)
        m_dw = beta1 * m_dw + (1 - beta1) * dw
        m_db = beta1 * m_db + (1 - beta1) * db
        
        # Update biased second moment estimate (v_dw and v_db)
        v_dw = beta2 * v_dw + (1 - beta2) * np.square(dw)
        v_db = beta2 * v_db + (1 - beta2) * np.square(db)
        
        # Compute bias-corrected first moment estimate (m_dw_corr and m_db_corr)
        m_dw_corr = m_dw / (1 - beta1 ** t)
        m_db_corr = m_db / (1 - beta1 ** t)
        
        # Compute bias-corrected second moment estimate (v_dw_corr and v_db_corr)
        v_dw_corr = v_dw / (1 - beta2 ** t)
        v_db_corr = v_db / (1 - beta2 ** t)
        
        # Apply AMSGrad variant if needed (keep track of max past squared gradients)
        if v_dw_max is not None and v_db_max is not None:
            v_dw_max = np.maximum(v_dw_max, v_dw_corr)
            v_db_max = np.maximum(v_db_max, v_db_corr)
            v_dw_corr = v_dw_max
            v_db_corr = v_db_max
        
        # Apply gradient clipping by norm
        if clipnorm is not None:
            # Compute global norm of the gradients (L2 norm)
            grad_norm = np.sqrt(np.sum(np.square(dw)) + np.sum(np.square(db)))
            if grad_norm > clipnorm:
                dw = dw * clipnorm / grad_norm
                db = db * clipnorm / grad_norm
        
        # Apply gradient clipping by value
        if clipvalue is not None:
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
        
        # Compute the updates for weights and biases (excluding weight decay)
        weight_update = learning_rate * (m_dw_corr / (np.sqrt(v_dw_corr) + epsilon))
        bias_update = learning_rate * (m_db_corr / (np.sqrt(v_db_corr) + epsilon))
        
        # Apply decoupled weight decay (modern AdamW)
        if weight_decay > 0:
            self.weight -= weight_decay * self.weight  # Apply weight decay directly to the weights (not scaled by learning rate)
        
        # Update weights and biases
        self.weight -= weight_update
        self.bias -= bias_update
        
        # Return updated moments for the next iteration
        return v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max
    #### End of adamw
    #######################################################################################
    #### nadam
    def nadam(self, dw, db, v_dw, v_db, m_dw, m_db, epoch):
        """
        Nadam optimizer for updating weights and biases with all parameters, including weight decay, gradient clipping, 
        and Nesterov momentum variant.
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            m_dw: First moment (mean) of the weights gradient.
            m_db: First moment (mean) of the biases gradient.
            v_dw: Second moment (variance) of the weights gradient.
            v_db: Second moment (variance) of the biases gradient.
            epoch: Current epoch for bias correction in Nadam.
            
        Returns:
            Updated first and second moments for the next iteration.
        """
        # Retrieve Nadam parameters from self.params
        learning_rate = self.params['optimizer_param']['learning_rate']
        beta1 = self.params['optimizer_param']['beta1']  # First moment decay
        beta2 = self.params['optimizer_param']['beta2']  # Second moment decay
        epsilon = self.params['optimizer_param']['epsilon']  # Small constant to prevent division by zero
        weight_decay = self.params['optimizer_param']['weight_decay']  # L2 regularization factor
        clipnorm = self.params['optimizer_param']['clipnorm']  # Clip gradients by global norm
        clipvalue = self.params['optimizer_param']['clipvalue']  # Clip gradients by value
        
        # Increment timestep (Nadam uses epoch + 1 for bias correction)
        t = epoch + 1
    
        # Nadam correction factor for the moving averages
        mu_t = beta1 * (1 - 0.5 * (0.96 ** (t / 250)))
        mu_t_plus_1 = beta1 * (1 - 0.5 * (0.96 ** ((t + 1) / 250)))
        
        # Nadam learning rate schedule
        learning_rate = learning_rate * (1 - 0.5 * (0.96 ** (t / 250)))
    
        # Update biased first moment estimate (m_dw and m_db)
        m_dw = beta1 * m_dw + (1 - beta1) * dw
        m_db = beta1 * m_db + (1 - beta1) * db
        
        # Update biased second moment estimate (v_dw and v_db)
        v_dw = beta2 * v_dw + (1 - beta2) * np.square(dw)
        v_db = beta2 * v_db + (1 - beta2) * np.square(db)
        
        # Compute bias-corrected first moment estimate (m_dw_corr and m_db_corr)
        m_dw_corr = m_dw / (1 - beta1 ** t)
        m_db_corr = m_db / (1 - beta1 ** t)
        
        # Compute bias-corrected second moment estimate (v_dw_corr and v_db_corr)
        v_dw_corr = v_dw / (1 - beta2 ** t)
        v_db_corr = v_db / (1 - beta2 ** t)
    
        # Nesterov momentum correction applied in Nadam
        m_dw_prime = (mu_t_plus_1 * dw + (1 - mu_t_plus_1) * m_dw_corr) / (np.sqrt(v_dw_corr) + epsilon)
        m_db_prime = (mu_t_plus_1 * db + (1 - mu_t_plus_1) * m_db_corr) / (np.sqrt(v_db_corr) + epsilon)
        
        # Apply gradient clipping by norm
        if clipnorm is not None:
            grad_norm = np.sqrt(np.sum(np.square(dw)) + np.sum(np.square(db)))
            if grad_norm > clipnorm:
                dw = dw * clipnorm / grad_norm
                db = db * clipnorm / grad_norm
        
        # Apply gradient clipping by value
        if clipvalue is not None:
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
        
        # Compute the updates for weights and biases (excluding weight decay)
        weight_update = learning_rate * m_dw_prime
        bias_update = learning_rate * m_db_prime
        
        # Apply decoupled weight decay (modern Nadam)
        if weight_decay > 0:
            self.weight -= weight_decay * self.weight  # Apply weight decay directly to the weights (not scaled by learning rate)
        
        # Update weights and biases
        self.weight -= weight_update
        self.bias -= bias_update
        
        # Return updated moments for the next iteration
        return v_dw, v_db, m_dw, m_db
    #### End of nadam
    #######################################################################################
    #### adagrad
    def adagrad(self, dw, db, v_dw, v_db, epoch):
        """
        Adagrad optimizer for updating weights and biases with all parameters, including weight decay and gradient clipping.
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            v_dw: Running sum of squared gradients for weights.
            v_db: Running sum of squared gradients for biases.
            epoch: Current epoch for tracking purposes (optional but included for consistency).
        """
        # Retrieve Adagrad parameters from self.params
        learning_rate = self.params['optimizer_param']['learning_rate']
        epsilon = self.params['optimizer_param']['epsilon']  # Small constant to prevent division by zero
        weight_decay = self.params['optimizer_param']['weight_decay']  # L2 regularization factor
        clipnorm = self.params['optimizer_param']['clipnorm']  # Clip gradients by global norm
        clipvalue = self.params['optimizer_param']['clipvalue']  # Clip gradients by value
        
        # Accumulate the squared gradients for weights and biases
        v_dw += np.square(dw)
        v_db += np.square(db)
        
        # Apply gradient clipping by norm
        if clipnorm is not None:
            grad_norm = np.sqrt(np.sum(np.square(dw)) + np.sum(np.square(db)))
            if grad_norm > clipnorm:
                dw = dw * clipnorm / grad_norm
                db = db * clipnorm / grad_norm
        
        # Apply gradient clipping by value
        if clipvalue is not None:
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
        
        # Compute the updates for weights and biases
        weight_update = learning_rate * (dw / (np.sqrt(v_dw) + epsilon))
        bias_update = learning_rate * (db / (np.sqrt(v_db) + epsilon))
        
        # Apply decoupled weight decay (modern Adagrad)
        if weight_decay > 0:
            self.weight -= weight_decay * self.weight  # Apply weight decay directly to the weights (not scaled by learning rate)
        
        # Update weights and biases
        self.weight -= weight_update
        self.bias -= bias_update
        
        # Return updated running sum of squared gradients
        return v_dw, v_db
    #### End of adagrad
    #######################################################################################
    #### adadelta
    def adadelta(self, dw, db, v_dw, v_db, epoch):
        """
        Adadelta optimizer for updating weights and biases with all parameters, including weight decay and gradient clipping.
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            v_dw: Running average of squared gradients for weights.
            v_db: Running average of squared gradients for biases.
            epoch: Current epoch for tracking purposes (optional but included for consistency).
        """
        # Retrieve Adadelta parameters from self.params
        learning_rate = self.params['optimizer_param']['learning_rate']
        rho = self.params['optimizer_param']['rho']  # Decay rate for the running average
        epsilon = self.params['optimizer_param']['epsilon']  # Small constant to prevent division by zero
        weight_decay = self.params['optimizer_param']['weight_decay']  # L2 regularization factor
        clipnorm = self.params['optimizer_param']['clipnorm']  # Clip gradients by global norm
        clipvalue = self.params['optimizer_param']['clipvalue']  # Clip gradients by value
        
        # Apply gradient clipping by norm
        if clipnorm is not None:
            grad_norm = np.sqrt(np.sum(np.square(dw)) + np.sum(np.square(db)))
            if grad_norm > clipnorm:
                dw = dw * clipnorm / grad_norm
                db = db * clipnorm / grad_norm
        
        # Apply gradient clipping by value
        if clipvalue is not None:
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
        
        # Accumulate the squared gradients for weights and biases (running average)
        v_dw = rho * v_dw + (1 - rho) * np.square(dw)
        v_db = rho * v_db + (1 - rho) * np.square(db)
        
        # Compute the updates for weights and biases (root mean square of accumulated gradients)
        weight_update = (np.sqrt(v_dw + epsilon)) * dw
        bias_update = (np.sqrt(v_db + epsilon)) * db
        
        # Apply decoupled weight decay (modern Adadelta)
        if weight_decay > 0:
            self.weight -= weight_decay * self.weight  # Apply weight decay directly to the weights (not scaled by learning rate)
        
        # Update weights and biases
        self.weight -= weight_update
        self.bias -= bias_update
        
        # Return updated running average of squared gradients
        return v_dw, v_db
    #### End of adadelta
    #######################################################################################
    #### ftrl
    def ftrl(self, dw, db, z_dw, z_db, n_dw, n_db, epoch):
        """
        FTRL optimizer for updating weights and biases with L1 and L2 regularization, weight decay, and gradient clipping.
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            z_dw: Accumulated z vector for weights.
            z_db: Accumulated z vector for biases.
            n_dw: Accumulated squared gradients for weights.
            n_db: Accumulated squared gradients for biases.
            epoch: Current epoch for tracking purposes (optional but included for consistency).
        """
        # Retrieve FTRL parameters from self.params
        learning_rate = self.params['optimizer_param']['learning_rate']
        learning_rate_power = self.params['optimizer_param']['learning_rate_power']  # Exponent for learning rate scaling
        initial_accumulator_value = self.params['optimizer_param']['initial_accumulator_value']  # Starting value for accumulators
        l1_strength = self.params['optimizer_param']['l1_regularization_strength']  # L1 regularization (sparse regularization)
        l2_strength = self.params['optimizer_param']['l2_regularization_strength']  # L2 regularization
        weight_decay = self.params['optimizer_param']['weight_decay']  # L2 regularization factor (decoupled weight decay)
        clipnorm = self.params['optimizer_param']['clipnorm']  # Clip gradients by global norm
        clipvalue = self.params['optimizer_param']['clipvalue']  # Clip gradients by value
        
        # Apply gradient clipping by norm
        if clipnorm is not None:
            grad_norm = np.sqrt(np.sum(np.square(dw)) + np.sum(np.square(db)))
            if grad_norm > clipnorm:
                dw = dw * clipnorm / grad_norm
                db = db * clipnorm / grad_norm
        
        # Apply gradient clipping by value
        if clipvalue is not None:
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
        
        # Accumulate squared gradients (n_dw and n_db)
        n_dw += np.square(dw)
        n_db += np.square(db)
        
        # Apply the update for z vector (FTRL specific update)
        z_dw += dw - ((np.sqrt(n_dw) - np.sqrt(n_dw - np.square(dw))) / learning_rate) * self.weight
        z_db += db - ((np.sqrt(n_db) - np.sqrt(n_db - np.square(db))) / learning_rate) * self.bias
        
        # Compute the learning rate scaling
        lr_dw = learning_rate / (np.power(n_dw + initial_accumulator_value, learning_rate_power))
        lr_db = learning_rate / (np.power(n_db + initial_accumulator_value, learning_rate_power))
        
        # Apply L1 and L2 regularization (L1 creates sparsity, L2 applies ridge regularization)
        weight_update = np.where(np.abs(z_dw) > l1_strength,
                                 -(z_dw - np.sign(z_dw) * l1_strength) / ((1 / lr_dw) + 2 * l2_strength),
                                 0.0)
        bias_update = np.where(np.abs(z_db) > l1_strength,
                               -(z_db - np.sign(z_db) * l1_strength) / ((1 / lr_db) + 2 * l2_strength),
                               0.0)
        
        # Apply weight decay (modern decoupled weight decay)
        if weight_decay > 0:
            self.weight -= weight_decay * self.weight
        
        # Update weights and biases
        self.weight += weight_update
        self.bias += bias_update
        
        # Return updated z and n values for tracking
        return z_dw, z_db, n_dw, n_db
    #### End of ftrl
    #######################################################################################
    #### adamax
    def adamax(self, dw, db, m_dw, m_db, u_dw, u_db, epoch):
        """
        Adamax optimizer for updating weights and biases with decoupled weight decay, gradient clipping, and the Adamax variant of Adam.
        
        Args:
            dw: Gradient of the loss w.r.t weights.
            db: Gradient of the loss w.r.t biases.
            m_dw: First moment (mean) of the weights gradient.
            m_db: First moment (mean) of the biases gradient.
            u_dw: Infinity norm (maximum absolute gradient) for weights.
            u_db: Infinity norm (maximum absolute gradient) for biases.
            epoch: Current epoch for bias-correction.
        """
        # Retrieve Adamax parameters from self.params
        learning_rate = self.params['optimizer_param']['learning_rate']
        beta1 = self.params['optimizer_param']['beta1']  # First moment decay
        beta2 = self.params['optimizer_param']['beta2']  # Decay for infinity norm
        epsilon = self.params['optimizer_param']['epsilon']  # Small constant to prevent division by zero
        weight_decay = self.params['optimizer_param']['weight_decay']  # L2 regularization factor (decoupled weight decay)
        clipnorm = self.params['optimizer_param']['clipnorm']  # Clip gradients by global norm
        clipvalue = self.params['optimizer_param']['clipvalue']  # Clip gradients by value
        
        # Increment timestep (Adamax uses epoch + 1 for time correction)
        t = epoch + 1
        
        # Update biased first moment estimate (m_dw and m_db)
        m_dw = beta1 * m_dw + (1 - beta1) * dw
        m_db = beta1 * m_db + (1 - beta1) * db
        
        # Update infinity norm (u_dw and u_db)
        u_dw = np.maximum(beta2 * u_dw, np.abs(dw))
        u_db = np.maximum(beta2 * u_db, np.abs(db))
        
        # Apply gradient clipping by norm
        if clipnorm is not None:
            grad_norm = np.sqrt(np.sum(np.square(dw)) + np.sum(np.square(db)))
            if grad_norm > clipnorm:
                dw = dw * clipnorm / grad_norm
                db = db * clipnorm / grad_norm
        
        # Apply gradient clipping by value
        if clipvalue is not None:
            dw = np.clip(dw, -clipvalue, clipvalue)
            db = np.clip(db, -clipvalue, clipvalue)
        
        # Compute the bias-corrected first moment estimate (m_dw_corr and m_db_corr)
        m_dw_corr = m_dw / (1 - beta1 ** t)
        m_db_corr = m_db / (1 - beta1 ** t)
        
        # Compute the updates for weights and biases (excluding weight decay)
        weight_update = learning_rate * (m_dw_corr / (u_dw + epsilon))
        bias_update = learning_rate * (m_db_corr / (u_db + epsilon))
        
        # Apply weight decay (decoupled from gradients)
        if weight_decay > 0:
            self.weight -= weight_decay * self.weight
        
        # Update weights and biases
        self.weight -= weight_update
        self.bias -= bias_update
        
        # Return updated m and u terms for the next iteration
        return m_dw, m_db, u_dw, u_db
    #### End of adamax
    #######################################################################################
    #### loss_and_gradient
    def loss_and_gradient(self, W):
        """
        Compute both the loss and the gradient for L-BFGS.
        Args:
            W: Flattened weight and bias array (concatenated).
        Returns:
            loss: Scalar loss value.
            grad: Flattened gradient (concatenated).
        """
        # Reshape W to extract weights and biases
        num_features = self.params['xtrain'].shape[1]
        num_classes = self.params['num_class']
        
        weight = W[:num_features * num_classes].reshape(num_classes, num_features)
        bias = W[num_features * num_classes:].reshape(num_classes, 1)
    
        # Forward pass
        Z = np.dot(self.params['xtrain'], weight.T) + bias.T
        prob = self.softmaxfun(Z)
        
        # Compute the loss
        log_likelihood = -np.sum(self.params['yonehot_train'] * np.log(prob + 1e-9), axis=1)
        loss = np.mean(log_likelihood)
        
        # Add regularization (L2 weight decay)
        if self.params['optimizer_param']['weight_decay'] > 0:
            loss += 0.5 * self.params['optimizer_param']['weight_decay'] * np.sum(weight ** 2)
        
        # Backward pass: Compute gradient
        dz = prob - self.params['yonehot_train']
        dw = np.dot(dz.T, self.params['xtrain']) / self.params['xtrain'].shape[0]
        db = np.mean(dz, axis=0).reshape(num_classes, 1)
        
        # Add weight decay to gradients
        if self.params['optimizer_param']['weight_decay'] > 0:
            dw += self.params['optimizer_param']['weight_decay'] * weight
        
        # Flatten and concatenate gradients for L-BFGS
        grad = np.concatenate([dw.flatten(), db.flatten()])
        
        return loss, grad
    #### End of loss_and_gradient
    #######################################################################################
# class EarlyStoppingException(Exception):
#     """Custom exception for early stopping."""
#     pass

# def lbfgs(self):
#     """
#     L-BFGS optimizer with early stopping based on validation loss.
#     """
#     # Initialize weights and biases (flattened for L-BFGS)
#     num_features = self.params['xtrain'].shape[1]
#     num_classes = self.params['num_class']
#     initial_weight = self.weight.flatten()
#     initial_bias = self.bias.flatten()
#     initial_params = np.concatenate([initial_weight, initial_bias])

#     # Retrieve optimizer parameters
#     learning_rate = self.params['optimizer_param']['learning_rate']
#     max_iter = self.params['optimizer_param']['max_iter']
#     max_eval = self.params['optimizer_param']['max_eval']
#     tolerance_grad = self.params['optimizer_param']['tolerance_grad']
#     tolerance_change = self.params['optimizer_param']['tolerance_change']
#     history_size = self.params['optimizer_param']['history_size']
#     line_search_fn = self.params['optimizer_param']['line_search_fn']

#     # Variables for early stopping
#     best_val_loss = float('inf')  # Initialize to a large number
#     patience_counter = 0  # Keep track of how long the validation loss has been worsening
#     patience = self.params['patience']  # Early stopping patience (epochs)

#     # L-BFGS callback to track progress and implement early stopping
#     def callback(W):
#         nonlocal best_val_loss, patience_counter
        
#         # Reshape weights and biases from flattened array
#         weight = W[:num_features * num_classes].reshape(num_classes, num_features)
#         bias = W[num_features * num_classes:].reshape(num_classes, 1)
        
#         # Save the current state of weights and biases in the model
#         self.weight = weight
#         self.bias = bias

#         # Calculate training loss (used for printing)
#         train_loss, _ = self.loss_and_gradient(W)
#         print(f"Current Training Loss: {train_loss}")

#         # Check validation loss (if validation data is available)
#         if self.params['xval'] is not None and self.params['yonehot_val'] is not None:
#             # Forward pass on validation set
#             val_Z = np.dot(self.params['xval'], weight.T) + bias.T
#             val_prob = self.softmaxfun(val_Z)
#             val_log_likelihood = -np.sum(self.params['yonehot_val'] * np.log(val_prob + 1e-9), axis=1)
#             val_loss = np.mean(val_log_likelihood)
#             print(f"Validation Loss: {val_loss}")

#             # Early stopping condition: Check if validation loss increased
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss  # Update best validation loss
#                 patience_counter = 0  # Reset patience counter
#             else:
#                 patience_counter += 1  # Increment patience counter
#                 if patience_counter >= patience:
#                     print(f"Early stopping triggered. Validation loss increased for {patience} epochs.")
#                     raise EarlyStoppingException  # Stop the optimization by raising an exception

#     # Minimize the loss using L-BFGS
#     try:
#         result = minimize(
#             fun=self.loss_and_gradient,  # Loss and gradient function
#             x0=initial_params,  # Initial parameters (weights and biases)
#             method='L-BFGS-B',
#             jac=True,  # Indicates that the function returns both loss and gradient
#             options={
#                 'maxiter': max_iter,
#                 'maxfun': max_eval,
#                 'ftol': tolerance_change,
#                 'gtol': tolerance_grad,
#                 'disp': True,  # Display optimization progress
#                 'maxls': history_size  # Set history size for line search
#             },
#             callback=callback  # Callback function to track progress
#         )
        
#         # Final update of weights and biases
#         final_params = result.x
#         self.weight = final_params[:num_features * num_classes].reshape(num_classes, num_features)
#         self.bias = final_params[num_features * num_classes:].reshape(num_classes, 1)

#         print("Optimization finished. Final loss:", result.fun)
#         return result

#     except EarlyStoppingException:
#         print("Early stopping activated. Optimization stopped.")
#         return None

    #######################################################################################
    #### Fit
    def fit(self):
        """
        Train the model using stochastic gradient descent.
        """
        X_train = self.params['xtrain']
        y_train = self.params['yonehot_train']
        X_val = self.params['xval']
        y_val = self.params['yonehot_val']
        batch_size = self.params['batch_size']
        random_seed = self.params['random_seed']
        buffer_size = self.params['buffer_size']
        shuffle = self.params['shuffle']
        epochs = self.params['epochs']
        patience = self.params['patience']
        optimizer_type = self.params['optimizer']
        verbose = self.params['verbose']
        
        # Optimizer's params
        if optimizer_type == 'sgd' or optimizer_type == 'sgdnm':
            v_dw = np.zeros_like(self.weight)
            v_db = np.zeros_like(self.bias)
        elif optimizer_type == 'adam':
            v_dw = np.zeros_like(self.weight)
            v_db = np.zeros_like(self.bias)
            m_dw = np.zeros_like(self.weight)
            m_db = np.zeros_like(self.bias)
            v_dw_max = np.zeros_like(self.weight)
            v_db_max = np.zeros_like(self.bias)
        elif optimizer_type == 'rmsprop':
            v_dw = np.zeros_like(self.weight)
            v_db = np.zeros_like(self.bias)
            m_dw = np.zeros_like(self.weight)
            m_db = np.zeros_like(self.bias)
            mean_dw = np.zeros_like(self.weight)
            mean_db = np.zeros_like(self.bias)
        elif optimizer_type == 'adamw':
            v_dw = np.zeros_like(self.weight)
            v_db = np.zeros_like(self.bias)
            m_dw = np.zeros_like(self.weight)
            m_db = np.zeros_like(self.bias)
            v_dw_max = np.zeros_like(self.weight)
            v_db_max = np.zeros_like(self.bias)
        elif optimizer_type == 'nadam':
            v_dw = np.zeros_like(self.weight)
            v_db = np.zeros_like(self.bias)
            m_dw = np.zeros_like(self.weight)
            m_db = np.zeros_like(self.bias)
        elif optimizer_type == 'adagrad':
            v_dw = np.zeros_like(self.weight)
            v_db = np.zeros_like(self.bias)
        elif optimizer_type == 'adadelta':
            v_dw = np.zeros_like(self.weight)
            v_db = np.zeros_like(self.bias)
        elif optimizer_type == 'ftrl':
            z_dw = np.zeros_like(self.weight)
            z_db = np.zeros_like(self.bias)
            n_dw = np.zeros_like(self.weight)
            n_db = np.zeros_like(self.bias)
        elif optimizer_type == 'adamax':
            m_dw = np.zeros_like(self.weight)
            m_db = np.zeros_like(self.bias)
            u_dw = np.zeros_like(self.weight)
            u_db = np.zeros_like(self.bias)
        # elif optimizer_type == 'lbgfs':
            
        # Initialize the best validation loss for early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_weight = np.copy(self.weight)  # Save best weights
        best_bias = np.copy(self.bias)  # Save best biases

        for epoch in range(epochs):
            # Batch the training data
            X_batches, y_batches, index_batches = self.batcher(X_train, y_train, batch_size=batch_size,\
                            shuffle=shuffle, buffer_size=buffer_size)
            
            # Training loop
            for X_batch, y_batch in zip(X_batches, y_batches):
                # Forward pass: compute probabilities
                prob = self.predict(X_batch, Training = True)
                
                # Backward pass: compute gradients
                dz = prob - y_batch
                dw = np.dot(dz.T, X_batch) / batch_size + self.regularization_gradient(batch_size)
                db = np.sum(dz, axis=0, keepdims=True).T / batch_size
                
                # Update weights and biases based on optimizer type
                if optimizer_type == 'sgd' or optimizer_type == 'sgdnm':
                    v_dw, v_db = self.sgd(dw, db, v_dw, v_db, epoch)
                elif optimizer_type == 'adam':
                    v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max = \
                        self.adam(dw, db, v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max, epoch)
                elif optimizer_type == 'rmsprop':
                    v_dw, v_db, m_dw, m_db, mean_dw, mean_db = \
                        self.rmsprop(dw, db, v_dw, v_db, m_dw, m_db, mean_dw, mean_db, epoch)
                elif optimizer_type == 'adamw':
                    v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max = \
                        self.adamw(dw, db, v_dw, v_db, m_dw, m_db, v_dw_max, v_db_max, epoch)
                elif optimizer_type == 'nadam':
                    v_dw, v_db, m_dw, m_db = self.nadam(dw, db, v_dw, v_db, m_dw, m_db, epoch)
                elif optimizer_type == 'adagrad':
                    v_dw, v_db = self.adagrad(dw, db, v_dw, v_db, epoch)
                elif optimizer_type == 'adadelta':
                    v_dw, v_db = self.adadelta(dw, db, v_dw, v_db, epoch)
                elif optimizer_type == 'ftrl':
                    z_dw, z_db, n_dw, n_db = self.ftrl(dw, db, z_dw, z_db, n_dw, n_db, epoch)
                elif optimizer_type == 'adamax':
                    m_dw, m_db, u_dw, u_db = self.adamax(dw, db, m_dw, m_db, u_dw, u_db, epoch)
                # elif optimizer_type == 'lbgfs':
                    
            prob = self.predict(X_batches, index_batch = index_batches, Training = False)
            # Calculate the loss and add regularization
            loss_value = self.loss(prob, y_train) + self.regularization_loss()
            
            # Validation loss for early stopping (if validation data is provided)
            if X_val is not None and y_val is not None:
                val_prob = self.predict(X_val, batch_size = batch_size, Training = False)
                val_loss = self.loss(val_prob, y_val) + self.regularization_loss()
            else:
                val_loss = None
                
            if val_loss is None and loss_value < best_loss:
                best_loss = loss_value
                best_weight = np.copy(self.weight)  # Save current best weights
                best_bias = np.copy(self.bias)  # Save current best biases
                patience_counter = 0
            elif val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                best_weight = np.copy(self.weight)  # Save current best weights
                best_bias = np.copy(self.bias)  # Save current best biases
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose != 0 and (epoch % verbose) == 0:
                        print(f"Early stopping at epoch {epoch}, Best Loss: {best_loss:.6f}")
                    break
            
            if verbose != 0 and (epoch % verbose) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value:.6f}, Val Loss: {val_loss:.6f}, Patience: {patience_counter}")

        # Always restore the best weights and biases, even if early stopping wasn't triggered
        if verbose != 0:
            print("Restoring the best weights and biases from the best epoch.")
        self.weight = np.copy(best_weight)
        self.bias = np.copy(best_bias)
        #### End of fit
        #######################################################################################

    
    
    
        
        
        