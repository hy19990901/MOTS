import time
import numpy as np
from keras.models import load_model
import EA

if __name__ == "__main__":
  # start_time = time.clock()


  # model = load_model('model/ConvNet_mnist.h5df')  # ConvNet模型 12层
  model = load_model('./model/densenet_cifar10.h5df')
  # model = load_model('./model/model_fashion_mnist.h5')

  # f = np.load('./datasets/mnist_bim-b_compound8.npz')
  f = np.load('./datasets/cifar_bim-a_compound8.npz')
  # f = np.load('./datasets/fashion_mnist_scale_compound8.npz')
  y_test = f['y_test']
  x_test = f['x_test']

  act_layers = model.predict(x_test)
  # print(act_layers.shape)

  # start_time = time.clock()
  f_output = open(r'./EAParameters/softmaxoutput.txt', 'w')
  for x in act_layers:
      for i in range(9):
          f_output.write(str(x[i]) + '\t')
      f_output.write(str(x[9]) + '\n')
  f_output.close()
  # end_time = time.clock()
  # print("消耗时间：" + str(end_time - start_time))

  # start_time = time.clock()
  for selectsize in [100,300,500,1000]:
    f_selectisize = open(r'./EAParameters/selectsize.txt', 'w')
    f_selectisize.write(str(selectsize))
    f_selectisize.close()
    EA.RunEA()
  # end_time = time.clock()
  # print("消耗时间：" + str(end_time - start_time))
  
