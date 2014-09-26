import numpy as np
import random

def sigmoid(z):
    return 1./(1.+np.exp(-z))
    
sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
    
sigmoid_prime_vec = np.vectorize(sigmoid_prime)
class CrossEntropyCost:
    @staticmethod
    def fn(a,y):
        return np.nan_to_num(np.sum(-y*np.log(a)-(1-y)*np.log(1-a)))
    @staticmethod
    def delta(z,a,y):
        return (a-y)
class Network():
    """
    implement the feedward 
    """
    def __init__(self,size,cost=CrossEntropyCost):
        """
        size is a list indicating the number of nodes in each layer
        """
        self.num_layers = len(size)
        self.sizes = size
        self.cost = cost
        self.default_weight_initializer()
        
    def default_weight_initializer(self):
        self.bias = [np.random.randn(y,1) for y in self.size[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.size[:-1],self.size[1:])]
    
    def large_weight_initializer(self):
        """
        need notice the difference with default_weight_initializer()
        """
        self.bias = [np.random.randn(y,1) for y in self.size[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.size[:-1],self.size[1:])]
    
    def feedforward(self,a):
        global sigmoid_vec
        for b,w in zip(self.bias, self.weights):
            a = sigmoid_vec(np.dot(w,a) + b)
        return a
      
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
             evaluation_data = None,
             monitor_evaluation_cost = False,
             monitor_evaluation_accuracy = False,
             monitor_training_cost = False,
             monitor_training_accuracy = False):
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [],[]
        training_cost, training_accuracy = [],[]
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta,lmbda, len(training_data))
            
            print 'Epoch %s training complet' % j
            if monitor_training_cost:
                cost = self.total_cost(training_data,
lmbda)
                training_cost.append(cost)
                print 'cost on training data: {}'.format(cost)

    def update_mini_batch(self,mini_batch, eta,lmbda,n):  
         nabla_b = [np.zeros(b.shape) for b in self.bias]
         nabla_w = [np.zeros(w.shape) for w in self.weights]
         for x,y in mini_batch:
             delta_nabla_b, delta_nabla_w = self.back_prop(x,y)
             nabla_b = [nb+dnb for nb, dnb in zip(nabla_b,delta_nabla_b)]
             nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
         self.weights =  [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
         self.bias = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.bias,nabla_b)]
        
    def back_prop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations = [x]
        activation = x
        zs = []
        for b,w in zip(self.bias,self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        
        delta = (self.cost).delta(zs[-1],activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*spv
            nabla_b = delta
            nabla_w = np.dot(delta, activations[-l-1].transpose())
        
        return nabla_b,nabla_w
        
        
        
        
                
             