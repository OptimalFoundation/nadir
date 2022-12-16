![StableDiffusion image titled Dawn of Eve](/assets/dawn_of_eve2.png)

# Dawn Of Eve: Will Adam Optimizer Rest in Peace or Long Live and Prosper? 

Since ages (5 years) Adam has absolutely dominated the Large Language Modelling Sphere for optimization.\
Practically no other optimizer has come close since plain SGD to the prowess and general usability of Adam, to the point where even after considerable work showcasing improvements over Adam, it still rules. 

This repository is a **reseach** project, meant to serve as a benchmark for experiments on Optimizers and Language Modelling. 

## Core Research Problems

Research Problems this project is trying to tackle:

* "Where and how has Adam been utilized for large language modelling?"
    * There must be a reason why Adam is so dominant for large language modelling, and understanding that is the first step to understading the alternative options to Adam. 
* "What alternative options are there to Adam? Are they competitive?"
    * There are a number of considerations to make while choosing the optimizer for LLM training, like convergence speed, memory overload, training speed and more. For alternatives to be truely competitive, they should present significiant improvements in one or more of these factors.
    * Another important factor to understand whether Adam should be replaced for LLM training is that "can you live without the alternative?". If an alternative is insignificant in its benefit as compared to Adam, utilizing it would not make sense because of the deeprooted-ness of the Adam in the community and ease of use from being readily available in frameworks.
* "I see the claims of the alternatives but how well do they actually perform in practice?"
    * Without extensive experimentation and emperical (hard and cold) evidence to back the claims of improvement over Adam, nothing really matters. That's the beauty of the research community based on peer-review. 
    * Tragically, one of the reasons why most of the recent work on convex optimisation has not made it to the industry (at least for LLMing) is the lack of proper testing on language modelling objectives, used with transformer-like models. Most papers, if at all, test LLMing on LSTMs, which learn differently than Transformers and might have different results. 


## Citations
<!-- 
If you wish to cite this work, please use the following bibtex:
```bibtex
``` -->

For a list of citations regarding the papers used to make this repository, please refer to [citations.md](citations.md). If any citation is missing please inform the repository maintainer to get it included. 

## Misc.

Here's a poem written for the demise of Adam:
> Adam is Old  
> Adam is Tired  
> Adam has Back-Pain  
> Adam wants to Retire  
>                       ~Author, 2022
