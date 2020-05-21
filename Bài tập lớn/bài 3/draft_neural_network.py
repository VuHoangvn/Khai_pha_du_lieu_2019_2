import numpy as np 
np.random.seed(42)

class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        # take input data and return output data
        return input

    def backward(self, input, grad_output):
        # calculate chain rule
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input)

class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        # compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
        return grad_output*relu_grad

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        # f(x) = W*x + b
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2/(input_units+output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):
        # f(x) = W*x + b
        # input shape: [batch, input_units]
        # output shape: [batch, output_units]

        return np.dow(input, self.weights) + self.biases
    
    def backward(self, input, grad_output):
        # compute df/dx = df/ddense * ddense/dx
        # where ddense/dx = weight transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]

        assert grad_weights.shape == self.weights.shape and 
                grad_biases.shape == self.biases.shape

        # here we perform a stochastic gradient descent step.
        self.weights = self.weights - self.learning_rate*grad_weights
        self.biases = self.biases - self.learning_rate*grad_biases

        return grad_input


def softmax_crossentropy_with_logits(logits, reference_answers):
    # compute crossentropy from logits [batch, n_classes] and ids of correct answers
    logits_for_answers = 
                logits[np.arange(len(logits)), reference_answers]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    # compute crossentropy gradient from logits [batch, n_classes] and ids of correct answers
    ones_for_answers = np.zeros_like(logits)

    ones_for_answers[np.arange(len(logits)), reference_answers] = 1
    softmax = np.exp(logits)/np.exp(logits).sum(axis=-1, keepdims=True)

    return (-ones_for_answers + softmax) / logits.shape[0]
    
     
