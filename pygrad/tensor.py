#inpired by https://github.com/tinygrad/teenygrad/blob/main/teenygrad/tensor.py
from functools import partialmethod
import numpy as pn

#start with three base classes

class Context:
    def __init__(self, arg, *tensors):
       self.arg = arg
       self.parents = tensors
       self.saved_tensors = []

    def save_for_backward(self, *x):
       self.saved_tensors.extend(x)

class Tensor:
    def __init__(self, data):
       #print(type(data), data)
       if type(data) != np.ndarray:
         print("error contructing tensor with %r" % data)
         assert(False)
       self.data = data
       self.grad = None

       # variables used for autograd graph construction
       self._ctx = None

    def __str__(self):
        return "Tensor %r with grad %r" % (self.data, self.grad)

    def backward(self, allow_fill=True):
        #print("running backward on", self)
        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            #fill in the first grad with one
           assert(self.data.size == 1)
           self.grad = np.ones_like(self.data)

        assert(self.grad is not None)

        grads = self._ctx.arg.backward(self._ctx, self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for t,g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
               print("grad shape must match tensor shape in %r, %r != %r" % (self._ctx.arg,g.shape, t.data.shape))
               assert(False)
            t.grad = g
            t.backward(False)

    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum().mul(div)

class Function:
    def apply(self, arg, *x):
       ctx = Context(arg, self, *x)
       ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
       ret._ctx = ctx
       return ret

def register(name, fxn):
    setattr(Tensor, name, partialmethod(fxn.apply, fxn))

#Implement a few function

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x*y

    @staticmethod
    def backward(ctx, grad_ouput):
       x,y = ctx.saved_tensors
       return y*grad_output, x*grad_ouput
register('mul', Mul)

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
      return x+y
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_ouput
register('add', Add)




























