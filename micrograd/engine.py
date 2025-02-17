import numpy as np
import math

class Value:
    def __init__(self, data, _ch=(), _op='', label='') -> None:
        self.data = data
        self._prev = set(_ch)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None # find a local derivative, and apply the chain rule
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, term):
        term = term if isinstance(term, Value) else Value(term)
        out = Value(self.data + term.data, _ch=(self, term), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            term.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, term):
        return self + term

    def __neg__(self):
        return self * -1

    def __sub__(self, term):
        return self + (-term)
    
    def __mul__(self, factor):
        factor = factor if isinstance(factor, Value) else Value(factor)
        out = Value(self.data * factor.data, _ch=(self, factor), _op='*')
        def _backward():
            self.grad += factor.data * out.grad
            factor.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only int/float"
        out = Value(self.data**exponent, _ch=(self, ), _op=f'**{exponent}')

        def _backward():
            self.grad += exponent * (self.data**(exponent - 1)) * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, factor):
        return self * factor

    def __truediv__(self, denominator):
        return self * denominator**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), _ch=(self, ), _op="exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, _ch=(self, ), _op="ReLU")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for ch in v._prev:
                    build_topo(ch)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0 # base case
        for node in reversed(topo):
            node._backward()
        
