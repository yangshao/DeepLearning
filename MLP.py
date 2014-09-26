import numpy as np

class Network():
    """
    implement the feedward 
    """
    def __init__(self,size,cost=CrossEntropy):
        """
        size is a list indicating the number of nodes in each layer
        """
        self.num_layers = len(size)
        self.sizes = size
        self.cost = cost
        self.default_weight_initializer()
        
    def defaul_weight_initializer(self):
        self.bias = [np.random.randn(y,1) for y in self.size[1:]]
        self.weight = [np.random.randn(y,x) for x,y in zip(self.size[:-1],self.size[1:])]