import os
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def load_mnist(path, kind='train'):
    
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

def get_fashion_mnist(data_path, scaler_type = 'standard'):
    print("Scaling dataset")
    
    X_train, y_train = load_mnist(data_path, kind='train')
    X_test, y_test = load_mnist(data_path, kind='t10k')
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
        
    else:
        scaler = MinMaxScaler()
    
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)
    
    return x_train, x_test, y_train, y_test

