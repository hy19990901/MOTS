import math
import numpy as np
from keras.models import load_model
from keras.utils import np_utils

import time

# from read import model

if __name__ == "__main__":

    # # 载入模型
    # model = load_model('model/ConvNet_mnist.h5df')  # ConvNet模型 12层
    # # model = load_model('./model/densenet_cifar10.h5df')
    # # model = load_model('./model/model_fashion_mnist.h5')
    # # model.summary()

    f = open(r'./EAParameters/selectsize.txt', 'r')
    line = f.readlines()[0]
    f.close()
    selectsize = int(line)

    # f = np.load('./datasets/mnist_bim-b_compound8.npz')
    # f = np.load('./datasets/mnist_bim-b_compound8.npz')
    f = np.load('./datasets/cifar_bim-a_compound8.npz')
    # f = np.load('./datasets/mnist_translation_compound8.npz')
    # f = np.load('./datasets/cifar_scale_compound8.npz')
    # f = np.load('./datasets/cifar_bim-b_compound8.npz')
    # f = np.load('./datasets/fashion_mnist_fgsm_compound8.npz')
    # f = np.load('./datasets/fashion_mnist_bim-a_compound8.npz')
    # f = np.load('./datasets/fashion_mnist_bim-b_compound8.npz')
    # f = np.load('./datasets/fashion_mnist_scale_compound8.npz')

    y_test = f['y_test']
    x_test = f['x_test']
    # print(x_test[1].shape)
    # print(x_test.shape)  # (10000, 28, 28, 1)
    f_output = open(r'./EAParameters/softmaxoutput.txt', 'r')
    softmax_output = f_output.readlines()
    act_layers = []
    for act in softmax_output:
        act = act[:-1].split("\t")
        act = np.array([float(val) for val in act])
        act_layers.append(act)
    act_layers = np.array(act_layers)
    print(act_layers.shape)
    f = open(r'./EAParameters/A.txt', 'r')
    populations = f.readlines()
    f.close()

    f_score = open(r'./EAParameters/EAobjective.txt', 'w')
    for vector in populations:
        vector = vector[:-1].split("\t")
        # print(len(vector))
        # print(vector)
        vector = np.array([float(val) for val in vector])
        select_sample_index = list(np.argsort(-vector)[:selectsize])
        # x = np.zeros((selectsize, 28, 28, 1))
        x = np.zeros((selectsize, 32, 32, 3))
        y = np.zeros((selectsize,))
        for i in range(selectsize):
          x[i] = x_test[select_sample_index[i]]
          y[i] = y_test[select_sample_index[i]]
        ratio = 0
        for i in select_sample_index:
          sum = 0
          act = act_layers[i]
          # print(act)
          for m in act:
              if m != 0:
                  sum -= m * math.log(m)
          ratio += sum
        norm = 0
        for w in range(selectsize):
            for j in range(w + 1, selectsize):
                norm += np.linalg.norm(x[w] - x[j])
        f_score.write(str(norm) + "\t" + str(ratio) + "\n")
    f_score.close()
    f_output.close()