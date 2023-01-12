import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras.optimizers import RMSprop

def retrain():
    selectsize = 100
    # model = load_model('./model/ConvNet_mnist.h5df')  # ConvNet模型 12层
    # model = load_model('./model/densenet_cifar10.h5df')
    model = load_model('./model/model_fashion_mnist.h5')
    # model.summary()

    # f = np.load('./datasets/mnist_translation_compound8.npz')
    # f = np.load('./datasets/cifar_cw-l2_compound8.npz')
    f = np.load('./datasets/fashion_mnist_scale_compound8.npz')
    # f = np.load('./datasets/cifar_bim-b_compound8.npz')
    y_test = f['y_test']
    x_test = f['x_test']

    # f = open(r'./EA_Result/cw-l2_cifar/100/Phen.txt', 'r')
    # f = open(r'./EA_Result/jsma_fashion_mnist/1000/Phen.txt', 'r')
    f = open(r'./EA_Result/scale_fashion_mnist/100/Phen.txt', 'r')
    x = f.readlines()
    y = x[0].split("\t")
    z = np.array([float(val) for val in y])
    print(z)
    print(z.shape)
    select_sample_index = list(np.argsort(-z)[:selectsize])
    print(select_sample_index)
    f.close()

    x = np.zeros((selectsize, 28, 28, 1))
    # x = np.zeros((selectsize, 32, 32, 3))
    y = np.zeros((selectsize,))

    for i in range(selectsize):
        x[i] = x_test[select_sample_index[i]]
        y[i] = y_test[select_sample_index[i]]

    index = [i for i in range(x_test.shape[0])]
    for i in select_sample_index:
        index.remove(i)
    x_target = np.zeros((x_test.shape[0] - selectsize, 28, 28, 1))
    # x_target = np.zeros((x_test.shape[0] - selectsize, 32, 32, 3))
    y_target = np.zeros((x_test.shape[0] - selectsize,))
    # print(x_test.shape[0]-selectsize)
    for i in range(x_test.shape[0] - selectsize):
        x_target[i] = x_test[index[i]]
        y_target[i] = y_test[index[i]]
    y_target = np_utils.to_categorical(y_target, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    y = np_utils.to_categorical(y, 10)


    score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test Loss: %.4f' % score[0])
    origin_acc = score[1]
    # print('Before retrain, Test accuracy: %.4f'% origin_acc)


    # 重训练模型，得到重训练之后的准确率

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit(x, y, batch_size=100, epochs=5, shuffle=True, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    # score = model.evaluate(x_target, y_target, verbose=0)
    retrain_acc = score[1]
    return origin_acc,retrain_acc

if __name__ == "__main__":

    # resultfilename = './retrain_result/cifar_cw-l2_100.txt'
    resultfilename = './retrain_result/fashion_mnist_fgsm_100.txt'
    # resultfilename = './retrain_result/mnist_jsma_100.txt'
    result_to_write = ''
    result_to_write += 'retrain acc:' + '\n'
    result_to_write += '['
    retrain_acc_sum = 0

    for i in range(10):
        print("这是第" + str(i+1) + "次循环")
        origin_acc, retrain_acc = retrain()
        retrain_acc_sum += retrain_acc
        result_to_write += str(round(retrain_acc, 4)) + ('' if i == 9 else ',')
    retrain_acc = retrain_acc_sum / 10
    result_to_write += ']'
    result_to_write += '\n' + 'original acc: ' + str(round(origin_acc, 4))
    result_to_write += '\n' + 'average retrain acc: ' + str(round(retrain_acc, 4))
    result_to_write += '\n' + 'acc improvement: ' + str(round(retrain_acc-origin_acc, 4))
    with open(resultfilename, 'a') as file_object:
        file_object.write(result_to_write)

    print('Before retrain, Test accuracy: %.4f' % origin_acc)
    print('After retrain, Test accuracy: %.4f'% retrain_acc)
    print('accuracy improvement:%.4f'%(retrain_acc - origin_acc))



