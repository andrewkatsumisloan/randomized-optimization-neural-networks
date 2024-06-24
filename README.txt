Download the code from here: 
https://github.com/andrewkatsumisloan/randomized-optimization-neural-networks

Create a pip environment, ensure joblib is a version earlier than 1.2.0. 

pipenv shell
pipenv install

Part 1
Run problems/fourPeaks.py and problems/kColor.py to get results from the first two problems analysis.
The output for these problems is in output/

Part 2
Run neural_network/nn_a1.py to get the initial gradient descent performance / charts/ 

Run neural_network/nn.py to see the neural network with weights optimized by the randomized optimization algorithms. 

Run nn_rhc.py, nn_ga.py, and nn_sa.py to see charts showing performance of these RO algorithms on updating model weights, by hyperparameters.

The data that is referenced is stored in dataset/diabetes_balanced.csv.