# Building Blocks of Deep Learning Models
This repository contains modular Python scripts that demonstrate the core mathematical operations and structures used in building neural networks — implemented entirely from scratch using NumPy. The project is designed for educational purposes, especially to explore how neural networks learn using backpropagation, gradient descent, and activation functions.

📌 **Key Highlights**
Forward and backward pass through layers and activations

Detailed implementation of:

1. ReLU and Softmax activations

2. Categorical Cross-Entropy loss

3. Dense (fully connected) layer

4. Dropout regularization

5. Manual backpropagation through:Single layer, Multiple layers
6. Cross-entropy + Softmax combined
7. Optimizers:
      Gradient Descent (with Momentum)
      Adam
      RMSProp
      Adagrad

8. Concepts like learning rate decay, broadcasting, and L1/L2 regularization

📁 File Structure Overview
File Name	Description
**Dense_layer_class.py**   ---> Implements a fully connected dense layer

**ActivationFunction.py**	 ---> Contains ReLU and Softmax activation logic

**DropoutLayer.py**	       ---> Implements dropout for regularization

**BP_buildingblock.py**	   ---> Demonstrates forward + backward pass block

**BP_throughReLUactivation.py**	---> Backpropagation through ReLU

**BP_throughCrossEntropyLoss.py**	---> Backprop through categorical cross-entropy

**BackpropagationThrough1layer.py**	---> Shows full pass through 1 layer

**CombinedBP_SoftmaxAndCrossEntropy.py**	---> Softmax + cross-entropy combined loss backward pass

**ADAGRADOptimizer.py**	---> Implements Adagrad optimizer

**AdamOptimizer.py**	---> Implements Adam optimizer

**GDwithMomentum.py**	---> Implements Gradient Descent with momentum

**RMSPropOptimizer.py**	---> Implements RMSProp optimizer

**OptimizerGD.py**	---> Basic gradient descent implementation

**LearningRateDecay.py**	---> Shows how learning rate decays over time

**Regularizer.py**	---> L1 / L2 regularization loss logic

**Matrix_SumAndBroadcasting.py**	---> Demonstrates broadcasting and sum tricks in NumPy

**EntireNeuralNetwork.py**	---> Combines all components into a full network

**Overall_NN.py**	---> Alternative complete network with training loop

To train full model run:
**python DropoutLayer.py**
