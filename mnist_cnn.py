
#TensorFlow 1.x 版本的原生 API
import tensorflow as tf# 导入 TensorFlow 库
import matplotlib.pyplot as plt# 导入 Matplotlib 库，用于绘制图像
import numpy as np#导入 Numpy 库，用于处理数组
import gzip #导入 gzip 库，用于处理 .gz 压缩文件
from struct import unpack# 从struct 库中导入 unpack 函数，用于解析二进制数据


train_num=60000  #训练集样本数
test_num=10000  #测试集样本数
img_dim=(1,28,28)  #图像的维度 (1 表示单通道，28x28 表示图像大小)
img_size=784  # 28*28的图像 28*28=784，表示一张图像展开后有 784 个像素点

# 定义训练集和测试集文件的路径（压缩格式）
x_train_path=r'D:\project\CNN_mnist_practice-master\CNN_mnist_practice-master\train_images\train-images-idx3-ubyte.gz'
y_train_path=r'D:\project\CNN_mnist_practice-master\CNN_mnist_practice-master\train_labels\train-labels-idx1-ubyte.gz'
x_test_path=r'D:\project\CNN_mnist_practice-master\CNN_mnist_practice-master\test_images\t10k-images-idx3-ubyte.gz'
y_test_path=r'D:\project\CNN_mnist_practice-master\CNN_mnist_practice-master\test_labels\t10k-labels-idx1-ubyte.gz'


def read_image(path):  # 读取图像数据
    with gzip.open(path,'rb') as f:# 打开.gz压缩文件
        magic,num,rows,cols=unpack('>4I',f.read(16)) #读取头部 16 个字节，包含魔数、样本数量、图像行列数
        img=np.frombuffer(f.read(),dtype=np.uint8).reshape(num,28*28) # 将剩余数据读取为 uint8 类型，并reshape成(num, 28*28)的数组
    return img


def read_label(path):  # 读取label数据
    with gzip.open(path,'rb') as f: #打开标签文件
        magic,num=unpack('>2I',f.read(8))#读取前 8 个字节，包含魔数和样本数量
        label=np.frombuffer(f.read(),dtype=np.uint8)
    return label
#这两个函数分别从压缩文件中读取图像和标签数据，解压缩后将图像数据格式化为28x28大小的图像，标签转为对应的类别。

def normalize_image(image):  # 将图像的像素值标准化到 [0, 1] 区间
    img=image.astype(np.float32)/255.0# 将像素值从 0-255 范围转换为 0-1 之间的浮点数
    return img


def one_hot_label(label):  # 图像标签的one-hot向量化
    lab=np.zeros((label.size,10)) # 初始化一个全 0 的数组，每行表示一个样本，有 10 个类别
    for i,row in enumerate(lab):# 遍历每一个样本
        row[label[i]]=1# 将标签对应的位置设为 1（one-hot 编码）
    return lab


def load_mnist(x_train_path,y_train_path,x_test_path,y_test_path,normalize=True,one_hot=True):  # 读取mnist数据集
    '''
    Parameter
    --------
    normalize：将图像的像素值标准化到0~1区间
    one_hot：返回的label是one_hot向量

    Return
    --------
    (训练图像，训练标签)，(测试图像，测试标签)
    '''
    #读取图像和标签
    image={
        'train':read_image(x_train_path),
        'test':read_image(x_test_path)
    }

    label={
        'train':read_label(y_train_path),
        'test':read_label(y_test_path)
    }
    #如果 normalize=True，则对图像进行标准化
    if normalize:
        for key in ('train','test'):
            image[key]=normalize_image(image[key])
    #如果one_hot=True，则对标签进行 one-hot编码
    if one_hot:
        for key in ('train','test'):
            label[key]=one_hot_label(label[key])

    return (image['train'],label['train']),(image['test'],label['test'])
#加载数据集函数

def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size #计算批次的起始索引
        end = start + batch_size #计算批次的结束索引
        batch_xs = X[start:end] #提取该批次的输入图像
        batch_ys = Y[start:end]#提取该批次的标签
        yield batch_xs, batch_ys #生成每一个batch
#生成批次数据

(x_train,y_train),(x_test,y_test)=load_mnist(x_train_path,y_train_path,x_test_path,y_test_path,normalize=True,one_hot=True)#加载数据集
'''
f,ax=plt.subplots(2,2)
ax[0,0].imshow(x_train[0].reshape(28,28),cmap='Greys')  #创建一个 2x2 的子图网格，生成 4 个子图，f 是整个图的容器，ax 是 2x2 的子图数组
ax[0,1].imshow(x_train[1].reshape(28,28),cmap='Greys')  #将训练集中的第 1 张图片绘制为灰度图，reshape 将一维数据变为 28x28 的二维图像
ax[1,0].imshow(x_train[2].reshape(28,28),cmap='Greys')
ax[1,1].imshow(x_train[3].reshape(28,28),cmap='Greys')
plt.show() #显示绘制的图像
'''
#TensorFlow 模型搭建
# 创建占位符
x=tf.placeholder("float",shape=[None,784])  #占位符x用于输入图像，大小为 [None, 784]，表示批次大小不固定，每个图像是 28*28=784 维
y=tf.placeholder("float",shape=[None,10])  #占位符y用于输入标签,大小为 [None, 10]，表示每个样本的 one-hot 标签

# 定义卷积层1的权重和偏置
w_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1)) #权重卷积核大小为 5x5,输入通道 1，输出通道 32
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))  #偏置为 32 维


# 搭建卷积层1
x_image=tf.reshape(x,[-1,28,28,1])  #将展平的图像数据重新排列为 [batch_size, height, width, channels]形状  将输入图像x reshape 为 28x28 大小，-1 表示批次大小自动调整，1是通道数（灰度图像只有一个通道）。
r_conv1=tf.nn.conv2d(x_image,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1  #指定卷积操作的步幅，四维数组中的 1 表示在每个维度上步幅为 1进行卷积操作，使用 stride=1 和 padding='SAME'
h_conv1=tf.nn.relu(r_conv1) # 使用 ReLU 激活函数

# 搭建池化层1
h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # 最大池化，池化窗口为 2x2，stride=2，padding='SAME'减少特征图的大小


# 定义卷积层2的权重和偏置
w_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1)) # 第二个卷积层的权重，卷积核 5x5，输入通道 32，输出通道 64
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))# 偏置为 64 维

# 搭建卷积层2
r_conv2=tf.nn.conv2d(h_pool1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2 # 卷积操作
h_conv2=tf.nn.relu(r_conv2)  # ReLU 激活函数


# 搭建池化层2
h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')# 最大池化，池化窗口为 2x2，stride=2

# 定义全连接层的权重和偏置
w_fc1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1)) # 全连接层的权重，输入为 7*7*64，输出为 1024
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))# 偏置为 1024 维

# 搭建全连接层
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

# 添加dropout
keep_prob=tf.placeholder(tf.float32)#Dropout概率的占位符，表示保留节点的比例，通常用于控制训练时的过拟合
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)# 对全连接层的输出进行 dropout，keep_prob 决定保留节点的概率

# 搭建输出层 使用 softmax 将模型的输出转换为概率分布，输出 10 个分类的概率值。
w_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1)) # 输出层的权重，输入 1024（上一层输出），输出 10（分类数量为 10）
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))# 输出层的偏置，大小为 10
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2) # 计算输出的 softmax 值，表示每个类别的概率

# 计算损失函数
cross_entropy=-tf.reduce_sum(y*tf.log(y_conv)) # 计算损失函数，用于评估预测值与真实值的差距

# 梯度下降法
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  #Adam 优化器，以 1e-4 的学习率最小化交叉熵损失

# 计算正确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1)) # 比较模型预测的标签和真实标签是否相同

accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float")) # 计算平均正确率，作为模型性能的评估指标


#训练和评估
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# 初始化所有变量
    for epoch in range(5):# 训练 5 个 epoch
        for batch_xs,batch_ys in generatebatch(x_train,y_train,60000,50):# 将训练集分为每批 50 个样本
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})# 每一批次进行一次梯度下降更新，使用 50% 的 dropout
        if epoch % 1 ==0: # 每个 epoch 打印一次准确率
            print("Epoch {0}, accuracy: {1}".format(epoch,sess.run(accuracy,feed_dict={x:x_test,y:y_test,keep_prob:1.0}))) # 在测试集上计算准确率，keep_prob=1 表示测试时不使用 dropout
            print(sess.run(y_conv[:10],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0}))# 打印前 10 个预测值
            print(sess.run(y[:10],feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0}))# 打印前 10 个真实标签