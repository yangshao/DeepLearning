import numpy as np
import random
import sys
import json
import time
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1./(1.+np.exp(-z))
    
sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
    
sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def vectorized_result(y):
    e = np.zeros((10,1))
    e[y] = 1.0
    return e
    
def load(filename):
    f = open(filename,'r')
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__],data['cost'])
    net = Network(data['sizes'],cost=cost)
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    
class LinearOutputLayerActivation:
    @staticmethod
    def fn(z):
        return z
    @staticmethod
    def prime(z):
        return np.ones(np.shape(z))
        
class SigmoidOutputLayerActivation:
    @staticmethod
    def fn(z):
        return sigmoid_vec(z)
    @staticmethod
    def prime(z):
        return sigmoid_prime_vec(z)
    
class CrossEntropyCost:
    @staticmethod
    def fn(a,y):
        return np.nan_to_num(np.sum(-y*np.log(a)-(1-y)*np.log(1-a)))
    @staticmethod
    def delta(z,a,y,func=sigmoid_prime_vec):
#        return (a-y)
        y = np.array(y,dtype='float')
        a = np.array(a,dtype='float')
        return ((1-y)/(1-a)-y/a)*func(z)

class QuadraticCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y,func=sigmoid_prime_vec):
        """Return the error delta from the output layer, fun is the prime function of the outputLayerFunction"""
        return (a-y) * func(z)

def classification_accuracy(net,data, convert=False):
        if convert:
            results = [(np.argmax(net.feedforward(x)), np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(net.feedforward(x)),y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)
def regression_accuracy(net,data):
    losses = []
    for i,(x,y) in enumerate(data):
        predict_y = net.feedforward(x)
        losses.append(np.sum((predict_y-y)**2))
    return np.sum(losses)/len(data)
class Network():
    """
    implement the feedward 
    """
    def __init__(self,size,cost=CrossEntropyCost,outputLayerFunc = SigmoidOutputLayerActivation,accuracy = classification_accuracy):
        """
        size is a list indicating the number of nodes in each layer
        """
        self.num_layers = len(size)
        self.sizes = size
        self.cost = cost
        self.ouputLayerFunc = outputLayerFunc
        self.accuracy = accuracy
        self.default_weight_initializer()
        
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
    
    def large_weight_initializer(self):
        """
        need notice the difference with default_weight_initializer()
        """
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
    
    def feedforward(self,a):
        global sigmoid_vec
        for i,(b,w) in enumerate(zip(self.biases, self.weights)):
            if i < len(self.biases)-1:
                a = sigmoid_vec(np.dot(w,a) + b)
        a = self.ouputLayerFunc.fn(np.dot(w,a)+b)
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
#            eta = 1.0/(j+1)
#            print eta
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta,lmbda, len(training_data))
            
            print 'Epoch %s training complete' % j
            if monitor_training_cost:
                cost = self.total_cost(training_data,lmbda)
                training_cost.append(cost)
                print 'cost on training data: {}'.format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(self,training_data)
                training_accuracy.append(accuracy)
                print 'Accuracy on training data: {}'.format(accuracy)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data,lmbda)
                evaluation_cost.append(cost)
                print 'Cost on evaluation data: {}'.format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(self,evaluation_data)
                evaluation_accuracy.append(accuracy)
                print 'Accuracy on evaluation data: {}'.format(accuracy)
            print
            
        self.performance_plot(training_cost,training_accuracy,evaluation_cost,evaluation_accuracy)
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
                
                

    def update_mini_batch(self,mini_batch, eta,lmbda,n):  
         nabla_b = [np.zeros(b.shape) for b in self.biases]
         nabla_w = [np.zeros(w.shape) for w in self.weights]
         for x,y in mini_batch:
             delta_nabla_b, delta_nabla_w = self.back_prop(x,y)
#             print np.shape(nabla_b[0]),np.shape(delta_nabla_b[0])
             nabla_b = [nb+dnb for nb, dnb in zip(nabla_b,delta_nabla_b)]
             nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
         self.weights =  [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
         self.biases = [b - (eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]
        
    def back_prop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
#        print np.shape(nabla_b[0])
        activations = [x]
        activation = x
        zs = []
        for i,(b,w) in enumerate(zip(self.biases,self.weights)):
            z = np.dot(w,activation) + b
            zs.append(z)
            if i < len(self.biases) - 1:
                activation = sigmoid_vec(z)
            else:
                activation = self.ouputLayerFunc.fn(z)
            activations.append(activation)
        
        delta = (self.cost).delta(zs[-1],activations[-1],y,self.ouputLayerFunc.prime)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)# all the neurons use the sigmoid function except the output layer
            delta = np.dot(self.weights[-l+1].transpose(),delta)*spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            
        
        return nabla_b,nabla_w
        
    def total_cost(self,data, lmbda, convert = False):
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        cost += 0.5 * (lmbda/len(data))*sum(np.linalg.norm(w) for w in self.weights)
        return cost
        
    
        
    def save(self,filename):
        data = {'sizes': self.sizes,
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'cost': str(self.cost.__name__)
        }
        f = open(filename,'w')
        json.dump(data,f)
        f.close()
        
    def performance_plot(self,training_cost,training_accuracy,evaluation_cost,evaluation_accuracy):
        print len(training_cost), len(training_accuracy),len(evaluation_cost),len(evaluation_accuracy)
        n = len(training_cost)
        train_x = [i for i in range(n)]
        m = len(evaluation_accuracy)
        evaluation_x = [i for i in range(m)]
        plt.plot(train_x,training_cost,color='r',label='training cost')
        plt.plot(evaluation_x,evaluation_cost,color='y',label='evaluation cost')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('cost')
        plt.show()
        
        plt.plot(train_x,training_accuracy,color='r',label='training accuracy')
        plt.plot(evaluation_x,evaluation_accuracy,color='y',label='evaluation accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.ylim(0,10000)
        plt.show()
        
        
        

#nSamples = 7
#X = np.linspace(0,10,nSamples).reshape((-1,1))
#T = 1.5 + 0.6 * X + 0.8 * np.sin(1.5*X)
#T[np.logical_and(X > 2, X < 4)] *= 3
#T[np.logical_and(X > 5, X < 7)] *= 3
#
#nSamples = 100
#Xtest = np.linspace(0,10,nSamples).reshape((-1,1)) + 10.0/nSamples/2
#Ttest = 1.5 + 0.6 * Xtest + 0.8 * np.sin(1.5*Xtest) + np.random.uniform(-2,2,size=(nSamples,1))
#Ttest[np.logical_and(Xtest > 2, Xtest < 4)] *= 3
#Ttest[np.logical_and(Xtest > 5,Xtest < 7)] *= 3
#print np.shape(X),np.shape(Xtest),np.shape(T),np.shape(Ttest)
#training_data = [(np.reshape(x,(-1,1)),np.reshape(y,(-1,1))) for (x,y) in zip(X,T)]
#x,y = training_data[0]
##print np.shape(x)
##print np.shape(y)
#test_data = [(np.reshape(x,(-1,1)),np.reshape(y,(-1,1))) for x,y in zip(Xtest,Ttest)]

from sklearn import datasets
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
train_set_size = 250
X_train = X[:train_set_size]  # selects first 250 rows (examples) for train set
X_test = X[train_set_size:]
y_train = y[:train_set_size]   # selects first 250 rows (targets) for train set
y_test = y[train_set_size:]


training_data = [(np.reshape(X_train[i],(-1,1)),np.reshape(y_train[i],(-1,1))) for i in range(len(y_train))]
test_data = [(np.reshape(X_test[i],(-1,1)),np.reshape(y_test[i],(-1,1))) for i in range(len(y_test))]
#print X_train[0]
#print training_data

        
#import mnist_loader
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()               
net = Network([10,1], cost=QuadraticCost, outputLayerFunc=LinearOutputLayerActivation,accuracy=regression_accuracy)
#net.large_weight_initializer()
start_time = time.time()
net.SGD(training_data, 100, 10, 0.2,  evaluation_data=test_data, lmbda = 50,monitor_training_cost=True,monitor_training_accuracy=True,monitor_evaluation_cost=True,monitor_evaluation_accuracy=True)
print time.time()-start_time