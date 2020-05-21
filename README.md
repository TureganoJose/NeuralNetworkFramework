# Neural Network Framework

Decided to create my own to dense neural networks. Now I have a better understanding of what's going on in the background as it is easy to rely on 3rd parties (keras, pytorch, tf or even matlab).
Definitely I've gained a lot of proficiency on dealing with large matrices in python (numpy). 

There were also some discoveries in SGD and GD in general and why non-convex optimization needs a little of stochasticity to avoid saddle points or local optimums.


## Dense Neural Networks

Implemented the following activations:
- Sigmoid
- tanh
- relu
- softmax (the derivative is a Jacobian)

Tested it with the usual MNIST and make moons datasets(from scikit)
Also added mini-batch and L2-regularization

Overfitting:
 
![No regularization](regu_0.gif)

L2 Regularization:
This is pretty much a least squares with Tikhonov regularization (aka Ridge). 
![Regularization 0.05](regu_0_05.gif)


## Resources
There is a lot of stuff out there, mostly low quality. It is difficult to find good material. Loads of crappy explanations in Medium and other blogs. Stick to academia.
- https://www.deeplearningbook.org/
- http://cs229.stanford.edu/notes/cs229-notes-deep_learning.pdf
- https://arxiv.org/pdf/1804.07612.pdf
- https://www.jefkine.com/
- https://d2l.ai/
- https://mlfromscratch.com/