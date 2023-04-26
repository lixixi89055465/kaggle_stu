from keras.layers import MaxPooling1D,Conv1D,UpSampling1D
from keras.layers import Dense, Input
from keras.models import Model
from keras import initializers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import time

my_seed = 999###
tf.random.set_seed(my_seed)##运行tf这段才能真正固定随机种子

plt.rcParams['font.sans-serif']=['SimHei']##中文乱码问题！
plt.rcParams['axes.unicode_minus']=False#横坐标负号显示问题！

originaldata = pd.read('建模数据.xlsx',index_col=0)

def data_format(data):

    data = (data - data.min()) / (data.max() - data.min())  ### minmax_normalized
  # data = (data - data.mean()) / data.std()  ### standarded_normalized
    data = np.array(data)##从矩阵形式转换数组形式
    return data
x_all_normalize = data_format(originaldata)

def bulid_BPNNmodel(train_data, test_data):

    ##压缩特征维度
    encoding_dim = 10
    input_msg = Input(shape=(train_data.shape[1],))

    # 编码层
    h1 = 128  ##隐藏层1
    h2 = 64  ##隐藏层2
    h3 = 32  ##隐藏层3
    encoded = Dense(h1, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01),bias_initializer='zeros')(input_msg)
    encoded = Dense(h2, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01),bias_initializer='zeros')(encoded)
    encoded = Dense(h3, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01),bias_initializer='zeros')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # 解码层
    decoded = Dense(h3, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01),bias_initializer='zeros')(encoder_output)
    decoded = Dense(h2, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01),bias_initializer='zeros')(decoded)
    decoded = Dense(h1, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01),bias_initializer='zeros')(decoded)
    decoded_output = Dense(train_data.shape[1], activation='relu',kernel_initializer=initializers.random_normal(stddev=0.01), bias_initializer='zeros')(decoded)

    autoencoder = Model(inputs=input_msg, outputs=decoded_output)  # 解码
    encoder_model = Model(inputs=input_msg, outputs=encoder_output)  # 编码
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    print(train_data.shape, input_msg.shape, decoded_output.shape, encoder_output.shape)
    bs = int(len(train_data) / 4)  #####数据集较少，全参与形式，epochs一般跟batch_size成正比
    epochs = max(int(bs / 2), 128 * 3)##最少循环128次
    a = autoencoder.fit(train_data, train_data, epochs=epochs, batch_size=bs, verbose=0, validation_split=0.2)##在训练集中划分0.2作为测试集
    print('训练集Loss列表（长度%s）%s：' % (len(a.history['loss']), a.history['loss']))
    print('测试集Loss列表（长度%s）%s：' % (len(a.history['val_loss']), a.history['val_loss']))
    print('\033[1;31m{0:*^80}\033[0m'.format('测试集损失函数值情况'))
    print(autoencoder.evaluate(test_data, test_data))  ##观察测试集损失情况
    # encoder_model.save('临时保存的BPNN模型.hdf5')
    return encoder_model,bs,a.history['loss'],a.history['val_loss']

###预测和作图
def outputresult(data,model):
    modelres = model(data,data)
    dim_msg = modelres[0].predict(data)##此时预测是拿纯测试集预测
    print('降维后数据维度：',dim_msg.shape)
    dim_msg = np.reshape(dim_msg,(dim_msg.shape[0],dim_msg.shape[1]))
    latent_feature = pd.DataFrame(dim_msg)
    latent_feature.index = originaldata.index##read_res[2]
    latent_feature.columns = [('feature'+str(i + 1)) for i in range(dim_msg.shape[1])]
    latent_feature = np.round(latent_feature,6)
    print(latent_feature)

    plt.figure(figsize=(15, 8))
    plt.plot(modelres[2], label='训练集损失',)
    plt.plot(modelres[3], label='测试集损失')
    plt.xlabel('循环次数',fontsize=18)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=15)
    plt.show()
    return x_all_normalize.shape
outputresult(x_all_normalize,bulid_BPNNmodel)