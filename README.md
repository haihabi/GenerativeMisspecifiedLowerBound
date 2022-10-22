# Generative Misspecified Lower Bound

The Misspecified  Cram\'er-Rao lower bound (MCRB) provides a lower bound on the performance of any unbiased estimator of parameter vector $\theta$ under model misspecification. An approximation of the MCRB can be numerically evaluated using a set of i.i.d samples of the true distribution at $\theta$. However, obtaining a good approximation for multiple values of $\theta$ requires collocating an unrealistically large number of samples. In this paper, we present a method for approximating the MCRB using a Generative Model, referred to as a Generative Misspecified Lower Bound (GMLB), in which we train a generative model on data from the true measurement distribution. Then, the generative model can generate as many samples as required for any $\theta$, and therefore the GMLB can use a limited set of training data to achieve an excellent approximation of the MCRB for any parameter. We demonstrate the GMLB on two examples: a misspecified Linear Gaussian model; and a Non-Linear Truncated Gaussian model. In both cases, we empirically show the benefits of the GMLB in accuracy and sample complexity. In addition, we show the ability of the GMLB to approximate the MCRB on unseen parameters.

## Install

```
pip install pyresearchutils
pip install normflowpy
pip install -r requirements
```

## Run Experiment
```
python experiments/main_training.py
```

## References 
Habi, Hai Victor, Hagit Messer, and Yoram Bresler. "Learned Generative Misspecified Lower Bound." ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023.

