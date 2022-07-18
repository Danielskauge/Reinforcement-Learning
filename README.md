# Reinforcement-Learning
Using deep RL algorithms to solve various OpenAI gym enviroments

My implementation of DDPG includes the following optional features:

- Action noise, epsilon greedy and not
- Parameter noise, epsilon greedy and not
- Batch normalization
- 2 features from TD3
    - Delayed Policy Updates
    - Target policy smoothing
- Automated hyperparameter tuning (currently in separate file)

These can be toggled in the [ddpg.py](http://ddpg.py) file to easily experiment with different combinations of features. Suitable hyperparameter sets for the environment are applied automatically when selecting from Pendulum-v0, MountainCarContinuous-v1, and LunarLanderContinuous-v1

Potential future features

- Clipped Double Q learning from TD3.
    - I have implemented it elsewhere, but it is currently excluded for being slow and messy.
- Prioritized Experience Replay
    - Could improve sample efficiency, particularly important for real life robotics.

## DDPG

Deep Deterministic Policy Gradient (DDPG) is the algorithm I was given to begin with. It is a combination of deep Q-learning and policy gradient methods, in that it concurrently learns a Q-function and a policy. It uses off-policy data stored in a replay buffer and the Bellman equation to learn the Q-function, and uses the Q values to learn the policy. 

If you know the optimal action-value function Q*(a,s), then the optimal action a*(s) can be found be found by solving 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dac920ec-fd1f-4fdf-8ee8-5d9c52694bc1/Untitled.png)

This proposes a problem unique to continuous action spaces. In a discrete action space we could simply compute the Q-values for each action and choose the one with the highest Q-value. When the action space is continuous we can not exhaustively search the space for the optimal action.  Instead of running an expensive optimization subroutine to compute a near optimal action for each time step, we can approximate with

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/89d6e21b-a040-4a5b-8413-f33403b7102f/Untitled.png)

Where mu represents a deterministic policy, in the form of a neural network, that outputs an action given a state.

I addition to learning an action policy, we must learn an approximator for the optimal action-value function Q*(a,s). The loss function used in training is a mean-squared Bellman error function given a set of transitions. This loss is minimized with stochastic gradient descent.

  

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d5815440-711c-4c28-867f-9a671937f580/Untitled.png)

### Target networks

DDPG uses target networks for both the actor and Q-value networks, that return the targets in the loss function. When minimizing the loss, we are trying to make the Q-function more like the target Q-function. The problem is that these networks use the same parameter set, making loss minimization unstable. The solution is to use a time delay and polyak averaging to iteratively update the target networks with the parameters from the first networks.

The policy learning side of DDPG cconsist of learning a policy that maximized the action-value function. Because the action space is continuous and the action-value function is differentiable with regards to policy parameters, we perform gradient ascent with respect to policy parameters to solve 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2e5dc123-f0e5-4922-8d2a-56594499df6f/Untitled.png)

### Replay buffer

DDPG uses a replay buffer to store previous experiences, which it samples when updating the Q-function and policy networks. Buffer size is an important hyperparamater to tune. Too big slows down training, too small and the networks overfits to recent data.

### Epsilon Greedy Action and Parameter space noise

In DDPG , exploration comes from applying gausssian noise either to the actions produced by the actor, or directly to the parameters of the policy network itself. The original paper uses action noise that with a standard deviation that decays over time, called epsilon-greedy action noise. This gives a gradual shift from exploration to exploitation.

[Open.ai](http://Open.ai) had demonstrated that adding noise to the parameter space of the actor neural network itself was superior to action space noise in most cases. Compared to action space noise, picking the right scale of parameter noise is difficult because of three reasons

- Different layers are not equally sensitive to noise
- Noise sensitivity may change over time
- It is not obvious how the scale of noise in the parameter space translates to the resulting change in the action space.

This is solved by adapting the scale of parameter noise to a desired scale of equivalent action noise. For each episode, a “distance” is calculated between the action taken by the perturbed (noisy) and unperturbed network. The parameters noise is then scaled such that the “distance” is closer to the desired action space noise.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9e707aba-cd98-402b-9c83-fb3e6f0e4a26/Untitled.png)

### TD3 (Twin Delayed DDPG)

I have also implemented the features of the TD3 algorithm, as it has better training stability in theory. DDPG is brittle with respect to hyperparameters, and TD3 addresses this with three improvements.

- Clipped Double Q Learning: TD3 learns two Q-functions in stead of one and uses the smaller Q-value to form targets for the loss fucntion. This helps with DDPG’s problem of overestiamtion Q-values.
- Delayed Policy Updates: TD3 updates the policy and policy target networks less often than the Q-function.
- Target Policy Smoothing: TD3 adds noise to the target actions, make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.

### Batch Normalization

Batch normalization will in theory stabilize learning, thus allowing higher learning rates,  reducing the number of episodes required to train neural networks. A batch normalization layer normalizes its inputs, and is typically used before an activation function layer. One reason to use batch normalizaiton, is that when different features come in different ranges, they should be normalized such that one feature does not have an outsized impact. In the mountain car case, the action space is bounded [-1,1] but the state space is unbounded. Normalizing the state space beforehand would be difficult because we do not know the range of realistic values. Thus batch normalization comes in handy. 

### Automated hyperparameter tuning

 The optimal set of hyperparameters are specific to environment and implementation, such that a good set of hyperparameters for task 1 might not be sufficient to solve task 2. I realized that iteratively tuning parameters and testing the outcomes would be way to time consuming, and found a way to automize this process. We can automate finding good hyperparameters for each environment by reeatedly training the algorithm from scratch, each time with a new set of hyperparameters sampled from probability distributions. In the way I used it, the best set was selected as the one which resultated in the smallest loss after a given episode count.  I chose to use uniform distributions centered at the hyperparamer values used in similar experiments. This way the values would be bounded 0-1. I used the Hyperopt library to do this. 

### My approach

The changes I made to DDPG were in effort to stabilize training and make it less brittle with respect to hyperparameters. As this was my first time doing reinforcement learning, I implemented perhaps more features than necessary, in order to maximize my own learning.

## TASK 1: Pendulum

I found that the hyperparameters given to me initilally differed from those in the official DDPG paper, meaning that neither of them were optimal. I therefore tried to find better parameters through use automated hyperparameter tuning. The resulting parameter set beat both other parameter sets.

higher learning rate

- [ ]  Improve pendulum ddpg

The highest score () came from a combination of ….

### RESULTS

## TASK 2: Mountain Car Continuous

In the mountain car environment the agent rarely recieves a large positive award, which is when it reaches the top of the hill. This could be charaterized as a sparse award environment. This led me to implement parameters noise, which in theory handled sparse awards far better than action noise. 

**Results**

The hyperparameter sets others have uses with DDPG for mountain car worked poorly out of the box for me. Training on this environment was also much slower than pendulum, such that automated hyperparameter tuning would take too long. This lead me to implement the td3 version of DDPG, which is supposed to be more stable and less sensitive to hyperparams.

**Results**

Training td3 on mountain car turned out to be too slow, I suspect because it trains more networks than. Also little improvement in rewards. 

In later experiments, the agent would eventually stabilize at a good score for at most a couple of hundred episodes, before failing completely. I found that the larger buffer size I used, the longer the agent would stay stable before failing. I suspect this is because once the agent had been stable for long enough, the buffer was filled up with only similar experiences, causing the agent to overfit and fail. 

**Results**

Combining parameter noise and batch normalization initially produced a bug in the actor network, where output actions became NaN. I suspect this was because I was adding noise to the batch normalizattion layers, such that they were no longer normalizing their output properly. I fixed the bug by not applying noise to these layers, yet this combination still yielded horrible performance. Thus I discarded parameter noise, as batch normalization had by far the biggest positive impact on performance.

In theory, the stability from batch normalization should allow for higher learning rates. Though in my experiments training remained unstable. Results from reducing the learning rates yielded …

The highest score () came from a combination of ….

### RESULTS

only batch gave positve reward indicating success

## TASK 4: Lunar Lander Continuous

### THOUGHS ON ENVIROMENT AND SOLUTION

The lunar lander reward function can be considered somewhat sparse for the same reasons as with mountain car, which means parameter noise has potential. 

It also has a more complex two dimensional action space, which calls for better exploration.
