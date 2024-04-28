from mlp_model import MLP
from data_loader import get_fashion_mnist

def experiment_network_architecture(x_train, x_test, y_train, y_test):
    print("Runnning network experiment")
    
    architectures = [
        [784, 128, 10],
        [784, 256, 10],
        [784, 512, 10],
        [784, 256, 128, 10],
        [784, 256, 128, 10],
        [784, 256, 128, 64, 10],
        [784, 256, 32, 10],
        [784, 256, 16, 10]
    ]
    
    for architecture in architectures:
        model = MLP(architecture, 'sigmoid', 0.2, 'random', len(x_train), 30, 'sgd', 'MLP-Scratch/plots/architecture', str(architecture))
        model.fit(x_train, y_train, x_test, y_test)

def experiment_activation_functions(x_train, x_test, y_train, y_test):
    print("Runnning activation experiment")
    
    activations = ['tanh', 'relu']
    
    for activation in activations:
        model = MLP([784, 256, 32, 10], activation, 0.2, 'random', 128, 30, 'sgd', 'MLP-Scratch/plots/activation', str(activation))
        model.fit(x_train, y_train, x_test, y_test)

def experiment_optimizers(x_train, x_test, y_train, y_test):
    print("Runnning optimizer experiment")
    
    optimizers = ['nag', 'momentum', 'adagrad', 'rmsprop', 'adam']
    
    for optimizer in optimizers:
        model = MLP([784, 256, 32, 10], 'sigmoid', 0.2, 'random', 64, 30, optimizer, 'MLP-Scratch/plots/optimizer', str(optimizer))
        model.fit(x_train, y_train, x_test, y_test)
        
path = "MLP-Scratch/data"

# Call the experiments
x_train, x_test, y_train, y_test = get_fashion_mnist(path, 'standard')
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = get_fashion_mnist(path, 'minmax')

experiment_network_architecture(x_train, x_test, y_train, y_test)
experiment_network_architecture(x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled)

experiment_activation_functions(x_train, x_test, y_train, y_test)
experiment_activation_functions(x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled)

experiment_optimizers(x_train, x_test, y_train, y_test)
experiment_optimizers(x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled)
