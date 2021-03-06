Unlike statistics, ML is better w/ tends to deal with large, complex datasets (such as a dataset of millions of images, each consisting of tens of thousands of pixels) for which classical statistical analysis such as Bayesian analysis would be impractical. 


Machine learning requires three things:
    - Input data points—For instance, if the task is speech recognition, these data points could be sound files of people speaking. If the task is image tagging, they could be pictures.
    - Examples of the expected output—In a speech-recognition task, these could be human-generated transcripts of sound files. In an image etask, expected outputs could be tags such as “dog,” “cat,” and so on.
    - A way to measure whether the algorithm is doing a good job—This is necessary in order to determine the distance between the algorithm’s current output and its expected output. The measurement is used as a feedback signal to adjust the way the algorithm works. This adjustment step is what we call learning.


Machine-learning models are all about finding appropriate representations for their input data—transformations of the data that make it more amena- ble to the task at hand, such as a classification task.

Deep learning is a multistage way to learn data representations, and completely automates humans doing manual feature engineering.

### Anatomy of a neural network
- Layers, which are combined into a network (or model)
- The input data and corresponding targets
- The loss function, which defines the feedback signal used for learning
- The optimizer, which determines how learning proceeds

In Keras, 

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,)))
model.add(layers.Dense(32))

--
two-branch networks
multi-head networks
inception blocks

The topology of a network defines a hypothesis space

we defined machine learning as “searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal.”

1. Start by defining your network topology. 
2. Define your Loss function (objective function)

Models are networks of layers

### Types of loss functions and where to use them.
Use binary crossentropy for binary (two-class) classification
Use categorical crossentropy for many-class classification
Use mean-squared error for a (scalar) regression
Use Connectionist Temporal Classiciation (CTC) for a sequence learning problem.

## Types of networks and what they're good for
Convolutional networks are good for vision
Recurrent networks are good for sequence processing

## Typical Keras workflow
- Define your training data: input tensors and target tensors.
- Define a network of layers (or model) that maps your inputs to your targets.
- Configure the learning process by choosing a loss function, an optimizer, and some metrics to monitor.
- Iterate on your training data by calling the fit() method of your model.
