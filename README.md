![NADIRbanner2](https://user-images.githubusercontent.com/11348086/221370644-fcc05274-eb99-4237-a270-60dafd5ab69d.png)

# Nadir

![PyPI - Downloads](https://img.shields.io/pypi/dm/nadir)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Dawn-Of-Eve/nadir)
![GitHub Repo stars](https://img.shields.io/github/stars/Dawn-Of-Eve/nadir?style=social)

**Nadir** (pronounced _nay-di-ah_) is derived from the arabic word _nazir_, and means "the lowest point of a space". In optimisation problems, it is equivalent to the point of minimum. If you are a machine learning enthusiast, a data scientist or an AI practitioner, you know how important it is to use the best optimization algorithms to train your models. The purpose of this library is to help optimize machine learning models and enable them to reach the point of nadir in the appropriate context.

**Nadir** follows the principles of Simplicity, Modularity and Composabilty. Read more in the [Core Philosophy](#core-philosophy) section. 

## Table of Contents

- [Nadir](#nadir)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Simple Usage](#simple-usage)
- [Core Philosphy](#core-philosophy)
- [Supported Optimisers](#supported-optimisers)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)



## Installation

You can either choose to install from the PyPI index, in the following manner:

```bash
$ pip install nadir
```
or install from source, in the following manner:

```bash
$ pip install git+https://github.com/Dawn-Of-Eve/nadir.git
```
**Note:** Installing from source might lead to a breaking package. It is recommended that you install from PyPI itself.

## Simple Usage

```python
import nadir as nd

# some model setup here...
model = ...

# set up your Nadir optimiser
config = nd.SGDConfig(lr=learning_rate)
optimizer = nd.SGD(model.parameters(), config)

# Call the optimizer step
optimizer.step()
```

## Core Philosophy

`Nadir` was built to provide a sense of uniformity and integration that might be lacking in the optimisation community, based on the simple idea that optimisers are not islands. They are usually inheriting characteristics from other optimisers and they provide inspiration to other optimisers. So why not make optimisers inheritable, composible and modular objects? 

The core concepts that each optimiser in `Nadir` follows are:

1. **Simplicity** is of key importance. We prefer readability and simplicity over performance. Experiment, test and verify what works and what does not with Nadir. Optimise and write custom fused kernels for your favorite optimisers after, for performance. 

2. **Modularity** means that the each new optimiser should minimise on the extra new logic added by adding or editing only the parts that need editing. If you want to have a different momentum in Adam, you only change the function of Momentum after inheriting Adam. No need to write the entire code from scratch.

3. **Composibility** implies that we can take things from one optimiser and add them to another without much effort. You can build a optimiser that is the mix of RAdam and NAdam with the properties of AdaBelief, if you so desire! That's what makes this library really powerful. 

## Supported Optimisers

| Optimiser 	| Paper 	                                                 |
|:---------:	|:-----:	                                                 |
|  **SGD**  	| https://paperswithcode.com/method/sgd                      |
|  **Momentum** | https://paperswithcode.com/method/sgd-with-momentum        |
|  **NAG**      | https://jlmelville.github.io/mize/nesterov.html            |
|  **Adagrad** 	| https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf |
|  **RMSProp** 	| https://paperswithcode.com/method/rmsprop                  |
|  **Adam**     | https://arxiv.org/abs/1412.6980v9                          |
|  **Adamax**   | https://arxiv.org/abs/1412.6980v9                          |
|  **AdamW**    | https://arxiv.org/abs/1711.05101v3                         |
|  **Adadelta** | https://arxiv.org/abs/1212.5701v1                          |
|  **AMSGrad**  | https://arxiv.org/abs/1904.09237v1                         |
|  **RAdam**    | https://arxiv.org/abs/1908.03265v4                         |
|  **Lion**     | https://arxiv.org/abs/2302.06675                           |
|  **AdaBelief**| https://arxiv.org/pdf/2010.07468v5.pdf                     |
|  **NAdam**    | http://cs229.stanford.edu/proj2015/054_report.pdf          |

## Acknowledgements

We would like to thank all the amazing contributors of this project who spent so much effort making this repositary awesome! :heart:


## Citation

You can use the _Cite this repository_ button provided by Github or use the following bibtex:

```bibtex
@software{MinhasNadir,
    title        = {{Nadir: A Library for Bleeding-Edge Optimizers in PyTorch}},
    author       = {Minhas, Bhavnick and Kalathukunnel, Apsal},
    year         = 2023,
    month        = 3,
    version      = {0.0.2}
}
```
