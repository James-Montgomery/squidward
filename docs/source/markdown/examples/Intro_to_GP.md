# An Introduction to Gaussian Processes Models

## Who Is This Notebook Intended For?

Many machine learning engineers have never even heard of a Gaussian process model. This is a real shame as they are quite interesting models that can be very useful in a variety of situations from Digital Marketing, Oil Drilling, Robotics, Aerospace Design, and Optimization. Gaussian processes are a relatively old methodology being used as far back as 1880 by astronomer T. N. Thiele and gaining populatrity in the 1970's in geostatistics under the name [kriging](https://en.wikipedia.org/wiki/Kriging) (pronaunced "kree-ging").

This notebook is meant to be a quick and dirty introduction to Gaussian process models for those unfamiliar with the topic. I assume some basic familiarity with machine learning and Bayesian statistics. We won't too deep into the math or the nitty gritty, but at the end of this crash course I provide links to much more in-depth and rigorous sources of information for those readers wishing to dive deeper.

This notebook is simply meant to whet the appetite of those interested in using Gaussian process models. I've found that there are very few good introductory materials on the subject and that many new learners get quickly discouraged by the steep learning curve of the subject.

## What Is A Gaussian Process?

Before we get into Gaussian process models, we should first address what a Gaussian process is.

A [process](https://www.itl.nist.gov/div898/handbook/pmd/section2/pmd211.htm) is simply a phenomena that occurs in the world that one might desire to model. For example, if I were to start flipping coins and recording the results I would consider the flipping of the coins to be a process. This is sometimes referred to as the "data generating process".

This process can be exchangeable or non-exchangeable, meaning that the ordering of samples taken from the process may or may not matter. For example, if I flip a coin many times the flips are almost certainly independent random variables and so the order of the observations does not matter. This means that the data is exchangeable. However, a time series problems such as modeling cards drawn from a deck without replacement requires that I keep track of the order of the observations because they are dependent on each other (once I have drawn the king of hearts I will never draw it again). This kind of data is non-exchangeable.

A process is a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process#Gaussian_process_prediction,_or_kriging) if any collection of finite random variables drawn from the process is Gaussian distributed. A single sampled random variable would be a univariate Gaussian while a collection of sampled random variables would be a multivariate Gaussian.

Our coin flipping example is obviously not a Gaussian process. Any individual coin flip is [Bernoulli distributed](https://en.wikipedia.org/wiki/Bernoulli_distribution) and any collection of coin flips is [Binomial distributed](https://en.wikipedia.org/wiki/Binomial_distribution). We would call the coin flips a [Bernoulli process](https://en.wikipedia.org/wiki/Bernoulli_process).

Modeling the relationship of height by age in a population that is homogeneous with respect to gender and ethnicity tends to be a Gaussian process. This is because [heights at any given age tend to be Gaussian distributed](http://faculty.virginia.edu/ASTR3130/lectures/error2/error2.html) and so modeling heights across ages tends to be a Gaussian process.

There are other common statistical processes of sue to a machine learning engineer other than the Bernoulli and Gaussian processes.

For example, if we were trying to model the distribution of heights by age in a heterogeneous population where the mixture of subpopulations was unobserved we might choose to use a Student-t process. The [Student-t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) can be thought of as a more general case of the Gaussian distribution (or even as a [mixture of Gaussian distributions](http://www.sumsar.net/blog/2013/12/t-as-a-mixture-of-normals/)).

Another popular process is the [Poisson process](https://towardsdatascience.com/the-poisson-distribution-and-poisson-process-explained-4e2cb17d459), commonly used to model events over time such as mechanical failures or injuries at job sites.

Finally, there is the [Dirichlet process](https://en.wikipedia.org/wiki/Dirichlet_process), commonly used in topic modeling to represent uncertainty over distributions.

Each of these processes could be a chapter (or even a book) in and of themselves. They are each unique in their own way and are used in very different ways. We focus here on Gaussian processes and provide links for learning more about the other processes mentioned above.

## What is a Gaussian Process Model?

First of all, I will mention that the term "Gaussian process" and "Gaussian process model" are often used interchangeably. A Gaussian process model is a model where the posterior belief about the data generating process is a Gaussian process. This means that every point prediction that the model makes (the marginal posterior at that point) is a one dimensional Gaussian. Additionally, any collection of predictions can be represented by a multivariate Gaussian distribution.

While the marginal posterior at any point in feature space is a one dimensional Gaussian distribution, the entire posterior is a Gaussian process. This means that the posterior describes a distribution over functions.

This may seem confusing to some, but is actually a very common concept in Bayesian statistics. When fitting a [Bayesian](https://en.wikipedia.org/wiki/Bayesian_statistics) [general linear model (GLM)](https://en.wikipedia.org/wiki/General_linear_model) the modeler gets a posterior distribution over model parameters after inference. Sampling values for model parameters from this posterior is equivalent to sampling a function (in the case of a GLM these functions are all linear).

For a Gaussian process we get a posterior over functions without ever having to worry about model parameters! Instead of worrying about model parameters we simply get functions. Wait...how do we get functions without parameters?

We do this using [kernel methods](https://en.wikipedia.org/wiki/Kernel_method) and [lazy learning](https://en.wikipedia.org/wiki/Lazy_learning).

This next section is where most students get lost, so I will try describing the concepts below in two different ways: GPs as smoothing models and GPs as linear regression.

*Attempt One: GPs as Smoothing Models*

Rather than trying to describe the data using some [parametric model](https://en.wikipedia.org/wiki/Parametric_model), we use kernels to find the relationship (covariance) between labeled points in a training set. When we want to make predictions we use kernels to find the covariance of the new points with our training points. The labels of the new points are arrived at by smoothing (getting a weighted average) over the training points closest to them. How close two points are depends on the kernel being used.

*Attempt Two: GPs as Linear Regression*

A different way to think about GPs is to imagine that you are trying to model non-linear data with a linear model. To model the data well you need to transform the data so that it becomes linear. This transformation can be done using kernels. We typically say that the untransformed data is in "feature space" and the transformed data is in "kernel space". If we pick the right kernel then we will model the data well. If we fit this linear model using Bayesian statistics we would get a posterior over the model weights (which is equivalent to getting a posterior over functions where each function is described by a specific value of the model parameters). This is the basic idea behind a GP model *except* that we never see the model weights. We directly sample outcomes from the posterior rather than sampling parameters and them predicting outcomes.

**Author's Note:** Readers familiar with neural networks might be interested to know that neural networks and Gaussian process models have an interesting relationship. Predictions from a Bayesian neural network can be though of as the sum of random variables where the random variables are the weights of the model. Bayesian statisticians treat these weights as random variables and use inference to find the posterior distribution over these weights. if we imagine that the number of weights go to infinity, then predictions from the model become Gaussian ([central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)). This means that infinitely large Bayesian neural networks are Gaussian processes! It implies that finite sized Bayesian neural networks can be thought of as approximations of true Gaussian process models. In fact, a number of breakthroughs in Bayesian deep learning have come from this line of thinking.

## What Is The Deal With Kernels?

In the last section many of you probably picked up on my comment "if we pick the right kernel". How do we pick the right kernel? What even is a kernel?

A kernel is simply a function that relates two points in space. For example:

$$
k(x_i, x_j) = |x_i - x_j|
$$

This is a very simple kernel. It tells us the absolute different between two points in space. We can think of a kernel as giving us the variance of one point with another. If we apply a kernel over a set of points we get the covariance of all of the points.

$$
K(x, x) =
\begin{bmatrix}
    k(x_i, x_j) & \dots  \\
    \dots & \dots  \\
    \dots & k(x_n, x_m)
\end{bmatrix}
= cov(x, x)
$$

This brings us to an important constraint on kernels. Kernels must produce [positive semi-definite](https://math.stackexchange.com/questions/1733726/positive-semi-definite-vs-positive-definite) matrices. This means that for the matrix produced all values along the diagonal have values >= 0. We put this constraint in place to ensure that the kernel will only produce valid covariance matrices.

Kernels must also be symmetric, meaning that the order of the points being compared should not matter:

$$
k(x_i, x_j) = k(x_j, x_i)
$$

**Author's Note:** In the last section I noted that neural networks and Gaussian processes share a philosophical relationship based on Bayesian neural network behavior in the limit of infinitely many parameters. neural networks also relate to GPs where kernels are concerned. Neural networks can be considered [universal approximators ](https://en.wikipedia.org/wiki/Universal_approximation_theorem). This means that (theoretically) they can approximate arbitrarily complex functions as their number of parameters increases. We can take advantage of this for GPs by using a neural network as a kernel function. Rather than hand picking a kernel function we let the neural network learn a kernel that works for us. We call such a kernel a "general kernel".

## Advantages Of Gaussian Process Models

So what is so great about these models? Firstly, they fit nicely into the Bayesian modeling paradigm. We set priors, collect data, and use inference to get a posterior. This provides us with a model of the data generating process we are interested with predictions and uncertainty baked in.

Additionally, GPs can model highly non-linear data, unlike many Bayesian models. If we have very complex data we can feel more confident about getting a good fit.

## Disadvantages Of Gaussian Process Models

GPs can be hard to fit. Choosing the right kernel can be difficult and might require a fair bit of domain specific knowledge.

Additionally, the lazy learning characteristics of a GP model mean that fitting and prediction can be computationally taxing.

There is a lot of research out there on how to efficiently fit and/or approximate a Gaussian process model to mitigate these problems.

## What Have We Learned?

We learned that a Gaussian process is simply a set of events that are produced in such a way that any (or every) collection of finite random variables from the process is Gaussian distributed.

We learned that a Gaussian process model is a Bayesian modeling method that uses kernels to produce non-linear predictive models with a Gaussian process posterior.

We learned that Gaussian process models are most useful for modeling highly non-linear data where it is important to get uncertainty estimates around out model predictions.

## Where To Learn More

We've barely scratched the surface on Gaussian process models. Below I list other useful educational resources that you might want to explore. I don't agree with 100% of the methods and advice for fitting Gaussian process models in the materials below, but they are all good introductions to GP modeling for the beginner.

Fun Visualizations:
* [GP Demo](https://chi-feng.github.io/gp-demo/)
* [Gaussian Process Regression Demo](http://www.tmpl.fi/gp/)

More About Kernels:
* [The Kernel Cookbook](http://www.cs.toronto.edu/~duvenaud/cookbook/index.html)
* [David Duvenaud's Thesis](http://www.cs.toronto.edu/~duvenaud/thesis.pdf)

Introductory Level Material:
* [An intuitive guide to Gaussian processes](https://towardsdatascience.com/an-intuitive-guide-to-gaussian-processes-ec2f0b45c71d)
* [Gaussian Processes for Dummies](http://katbailey.github.io/post/gaussian-processes-for-dummies/)
* [Statistical Techniques in Robotics: Gaussian Process Part One](http://www.cs.cmu.edu/~16831-f14/notes/F09/lec20/16831_lecture20.albertor.pdf)
* [Statistical Techniques in Robotics: Gaussian Process Part Two](http://www.cs.cmu.edu/~16831-f14/notes/F09/lec21/16831_lecture21.sross.pdf)

Practitioner Level Material:
* [Gaussian Processes for Regression:
A Quick Introduction](https://www.robots.ox.ac.uk/~mebden/reports/GPtutorial.pdf)
* [Tutorial: Gaussian process models
for machine learning](http://mlg.eng.cam.ac.uk/tutorials/06/es.pdf)
* [ Gaussian Process and Deep Kernel Learning](https://www.cs.cmu.edu/~epxing/Class/10708-17/notes-17/10708-scribe-lecture24.pdf)
* [Gaussian Processes](http://people.ee.duke.edu/~lcarin/David1.27.06.pdf)

The GP Bible:
* [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)

Notes and Guides:
* [Notes On Gaussian Processes](http://keeganhin.es/blog/gp.html)
* [A Practical Guide to Gaussian Processes](https://drafts.distill.pub/gp/)
* [A Visual Exploration of Gaussian Processes](https://www.jgoertler.com/visual-exploration-gaussian-processes/)

Packages For GP Modeling:
* [GPyTorch](https://gpytorch.ai/)
* [Pyro](http://docs.pyro.ai/en/0.3.1/contrib.gp.html)
* [GPy](https://gpy.readthedocs.io/en/deploy/)
* [GPFlow](https://gpflow.readthedocs.io/en/develop/)
* [Tensorflow Probability](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Gaussian_Process_Regression_In_TFP.ipynb)
* [Sklearn](https://scikit-learn.org/stable/modules/gaussian_process.html)
* [George](https://george.readthedocs.io/en/latest/tutorials/first/)
* [Pymc3](https://docs.pymc.io/api/gp.html)
* [Stan](https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html)
* [Squidward](https://github.com/James-Montgomery/squidward)

Information about Other Processes
* [Stochastic Processes](https://www.ee.ryerson.ca/~courses/ee8103/chap4.pdf)
* [Bernoulli and Poisson](https://www.unf.edu/~cwinton/html/cop4300/s09/class.notes/DiscreteDist.pdf)
* [Bernoulli and Poisson](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-436j-fundamentals-of-probability-fall-2008/lecture-notes/MIT6_436JF08_lec20.pdf)
* [Poisson Processes and Gaussian Processes](http://astrostatistics.psu.edu/su05/richards_poisson061005.pdf)
* [Poisson](https://fromosia.wordpress.com/2017/03/19/stochastic-poisson-process/)
* [Basic Concepts of the Poisson Process](https://www.probabilitycourse.com/chapter11/11_1_2_basic_concepts_of_the_poisson_process.php)
* [Dirichlet Processes: A Gentle Tutorial](https://www.cs.cmu.edu/~kbe/dp_tutorial.pdf)
* [Student-t Process Regression with Student-t Likelihood](https://www.ijcai.org/proceedings/2017/0393.pdf)
* [Student-t Processes as Alternatives to Gaussian Processes](https://www.cs.cmu.edu/~andrewgw/tprocess.pdf)
