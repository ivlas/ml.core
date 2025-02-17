import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True) -> None:
        """
            nin - neuron inputs
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu() if self.nonlin else act
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self, nin, nout, nonlin) -> None:
        """
            nin - neuron inputs
            nout - number of Neurons in a single layer
        """
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __repr__(self):
            return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):

    def __init__(self, nin, nouts) -> None:
        """
            nin - neuron inputs
            nouts - list of nout (All layers that we want in MLP)
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin = i != len(nouts) - 1) for i in range(len(nouts))]
    
    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    @property
    def num_parameters(self):
        return len(self.parameters())