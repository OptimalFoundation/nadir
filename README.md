# Nadir

**Nadir** (pronounced _nay-di-ah_) is derived from the arabic word _nazir_, and means "the lowest point of a space". In optimisation problems, it is equivalent to the point of minimum. If you are a machine learning enthusiast, a data scientist or an AI practitioner, you know how important it is to use the best optimization algorithms to train your models. The purpose of this library is to help optimize machine learning models and enable them to reach the point of nadir in the appropriate context.

PyTorch is a popular machine learning framework that provides a flexible and efficient way of building and training deep neural networks. This library, Nadir, is built on top of PyTorch to provide high-performing general-purpose optimisation algorithms.  

At present, the objective is to merely get the ball rolling. We introduce Nadir with two optimization algorithms, namely Stochastic Gradient Descent (SGD) and Adaptive Momentum (AdaM). There are many variations of these two algorithms which can be explored. Synchronous and asynchronous, centered and uncentered or the use of the first or second moment in the adaptive term leads to algorithms with varying performances in different circumstances. In the future, we aim to add more of these to the mix.

## Supported Optimisers

| Optimiser 	| Paper 	|
|:---------:	|:-----:	|
|  **SGD**  	|       	|
|  **Adam** 	|       	|


## Installation

Currently, Nadir is not on the PyPi packaging index, so you would need to install it from source. To install Nadir into your python environment, paste the commands in your terminal:

```bash
$ pip install nadir
```

## Usage

```python
import nadir as nd

# some model setup here...
model = ...

# set up your Nadir optimiser
config = nd.SGDConfig(lr=learning_rate)
optimizer = nd.SGD(model.parameters(), config)

```

