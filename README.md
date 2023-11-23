# pyNACX
 Near-Axis Stellarator Code using JAX

TODO:

Take DESC's structure, readme and docs and create something similar as pyQSC+pyQIC up to order='r3' with the following changes.

- Give possibility of JAX, Tensorflow and Pytorch variants
- Create optimization scripts that take autodiff into account
- Create neural network training scripts for the forward and inverse solver approaches
- Perform hyperparameter optimization of the neural network parameters (batch size, learning rate, epochs, etc)
- Save neural network results in a given folder for different number of nfp's
- Give a neural network variant for the forward and inverse solver that loads the trained models and ouputs the results
- Create a physics-informed neural network that solves the first and second order solutions. This is a stepping stone for a neural network VMEC.
- Can the physics-informed neural network help create an inverse VMEC solver? or an inverse pyQSC solver?
- Test the performance of a reinforced learning approach to the inverse/forward pyQSC solver
- Create a database of configurations, similar to what the qsc code does
- Plot t-SNE and clusters of those configurations
- Take into account what is in https://github.com/rogeriojorge/qsc-ML including the branch quick-peek
- Add neural network to get near-axis configurations from VMEC (simplify workflow)
- Add loss fraction to database of x_samples using SIMPLE (useful for pyQIC)