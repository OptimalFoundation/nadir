# Literature Review 

## Optimisers

1. Sebastian Ruder (2016). An overview of gradient descent optimisation algorithms. arXiv preprint arXiv:1609.04747. https://arxiv.org/abs/1609.04747
    1. General article talking about all the additions to SGD pre-2016

```bibtex
@article{ruder2016overview,
  title={An overview of gradient descent optimization algorithms},
  author={Ruder, Sebastian},
  journal={arXiv preprint arXiv:1609.04747},
  year={2016}
}
```

2. John Chen (2020). An updated overview of recent gradient descent algorithms. https://johnchenresearch.github.io/demon/

3. Chandra, Kartik, et al. (2019) "Gradient descent: The ultimate optimizer."   https://arxiv.org/pdf/1909.13371.pdf
    1. When we train deep neural networks by gradient descent, we have to select a step size α for our optimizer. If α is too small, the optimizer runs            very  slowly, whereas if α is too large, the optimizer fails to converge. Choosing an appropriate α is thus itself an optimization task that                machine learning practitioners face every day. 
    2. To solve this issue, the authors suggest to apply gradient descent to find the optimal hyper-parameters, by computing the derivative of the loss            function not only with respect to the neural network’s weights, but also with respect to α.
    3. It is observed that automatic differentiation can be applied to optimize not only the hyperparameters, but also the hyper-hyperparameters, and the          hyper-hyper-hyperparameters, and so on. In fact, we can implement arbitrarily tall towers of recursive optimizers, which are increasingly robust to        the choice of initial hyperparameter.
    4. Experiments by the authors verify prediction of Baydin et al. (2018), that as the towers of hyperoptimizers grew taller, the resulting algorithms          would become less sensitive to the human-chosen hyperparameters.
```bibtex
@article{chandra2019gradient,
  title={Gradient descent: The ultimate optimizer},
  author={Chandra, Kartik and Meijer, Erik and Andow, Samantha and Arroyo-Fang, Emilio and Dea, Irene and George, Johann and Grueter, Melissa and Hosmer, Basil and Stumpos, Steffi and Tempest, Alanna and others},
  journal={arXiv preprint arXiv:1909.13371},
  year={2019}
}

```
4. Reddi, Sashank J., et al. (2019) "On the convergence of adam and beyond." https://arxiv.org/pdf/1904.09237.pdf
    1. The paper looks into the flaws made in the proof of Adam optimizer, and states that Adam does not correctly converge in all problems.
    2. This happens because of the exponential gradients which scale down the effect of large gradients, which maybe less frequent but in positive direction.
    3. Also the paper states that any adaptive method that is based on exponentially weighted moving averages (EWMA) of the gradients, including RMSProp, AdaDelta and NAdam, are also flawed.
    4. To solve this authors propose a new optimizer, AMSgrad, which introduces a hot-fix ${\hat v_{t+1}} = {\max(v_{t+1}, \hat v_t)}$.  
     Refer [notes](https://www.notion.so/On-the-Convergence-of-Adam-and-Beyond-fbadcd3f494243fab44b712e08e8dc73) for details.
```
@article{reddi2019convergence,
  title={On the convergence of adam and beyond},
  author={Reddi, Sashank J and Kale, Satyen and Kumar, Sanjiv},
  journal={arXiv preprint arXiv:1904.09237},
  year={2019}
}
```
