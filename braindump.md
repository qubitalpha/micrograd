# Brain Dump

* What is backpropagation?
    * Initializes backpropagation at the node g.
    * Backpropagation starts at g, and it recursively goes backwards by applying chain rule from calculus. Derivative of g w.r.t all internal nodes and also w.r.t a (which is a.grad) and g w.r.t. b (which is b.grad). If a.grad is 20, if we slightly nudge a, g will grow and the slope of that growth of g is 20.
    * ![image.png](attachment:f8c110c0-3232-4132-9f8c-ab9b1e91c3a7.png)
    * https://github.com/karpathy/micrograd/tree/master#example-usage
    * Backprop calculates gradients for all intermediate nodes w.r.t Loss (L). For example, computing dL/dw for an intermediate node w gives us a slope s, meaning if we slightly wiggle w, it affects L by that slope. The ultimate goal is finding the derivative of L w.r.t the NN weights. While leaf nodes consist of either these weights or input data, we only care about the weights; the input data is fixed and cannot be changed.

* What is slope? It is run over rise. Normalized change. Or (f(x + h) - f(x))/h. h is the change

* Some concepts from NN:
    * The input layer is not neurons, neurons start from 1st layer. Input layer simply holds raw data and contains no weights, biases, or activation functions.
    * Hidden and output layers contain the computational neurons that perform the actual math. Each neuron starting from Layer 1, get inputs from input layer. And these neurons hold weights and a bias. If we remove randomization of weights, every neuron in layer 1, will get the same value.
    * A neuron multiplies its inputs by specific weights, adds a bias, and applies a non-linear activation function.
    * Weights belong to the connections between layers, while each receiving neuron has exactly one bias.
    * Adding neurons increases the number of weights multiplicatively and the number of biases additively.
    * Activation functions (basically non-linear function like tanh, ReLU etc etc) introduce non-linearity, allowing the network to learn complex patterns instead of just drawing straight lines. In fact, we can have a neural network with just linear functions, but it wouldn't solve complex patterns.
    * An entire layer typically uses the same activation function to maximize hardware processing speed. However, different layers might have different activation function.
    * A network can be as small as a single neuron or as large as modern models like GPT-4, which have roughly 100 layers and over a trillion parameters.

* What is embedding?
    * It is basically representation of a character or word in the form of vector. It can be of any dimension.
    * For e.g. (referring makemore/one) if we want to represent letter 'a' with 26 dimensions, we can use one hot encoder. Basically 1000000...25zeros. Similarly, if we want to represent letter 'b' with 26 dimensions, it will be 01000...24zeros. And so on.
    * But if we want to represent letter 'a' in 2 dimensions, which we can (referring to makemore/two). We will randomly assign
    * However, in micrograd - we did not need to embed, because it was simple example and we had already taken integers as example. xs was 3 int element vector, ys was 1 or -1 (again integer).

* What is "activation"? Mathematical operation applied to a neuron's raw weighted sum ($\sum w_i x_i + b$) to transform its value before passing it forward.

* What is "softmax"? Is a type of activation function, used specifically on the last layer. Its entire job is to take a list of raw, unconstrained numbers and turn them into probabilities (Every single number must be between 0 and 1. All the numbers must add up to exactly 1.0 (or 100%). )