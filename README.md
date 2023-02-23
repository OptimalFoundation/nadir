# Nadir

**Nadir** (pronounced _nay-d-ah_) is derived from the arabic word _nazir_, means the lowest point of a space and is the opposite of the word zenith. In optimisation terms, it is equivalent to the point of minima. And making the machine learning model reach that point of Nadir under optimisation is the purpose of this library.  

This library is built on top of PyTorch to provide high-performing general-purpose optimisation algorithms. 


## Supported Optimisers

| Optimiser 	| Paper 	|
|:---------:	|:-----:	|
|  **SGD**  	|       	|
|  **Adam** 	|       	|


## Installation

Currently, Nadir is not on the PyPi packaging index, so you would need to install it from source. 

To install Nadir into your python environment, paste the commands in your terminal:

```bash
$ pip install git+https://github.com/Dawn-Of-Eve/nadir.git
```

## Usage

```
import nadir as nd
config = nd.SGDConfig(lr=learning_rate)
optimizer = nd.SGD(model.parameters(), config)

```

