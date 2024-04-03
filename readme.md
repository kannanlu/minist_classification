# Neural Network on MNIST classification
This repo contains the necessary components of running a neural network model on MNIST data set. The repo contains modules that downloads the data set to local directory, data loader set up, model definition, train and test functionalities. 

## Model
The models/simple_models/SimpleLayer defines the model.

## Training
To train the model, simply run main.py. It performs the training using function /models/utils/train_model_default, this function trains the model together with validation. It also saves the best model parameters given the validation result. In the end, it evaluates the test result on test data set. The results, model, and model config will be saved in directory /checkpoints. The training process can also utilize tensorboard logging if enabled, the logs are saved in /logs.