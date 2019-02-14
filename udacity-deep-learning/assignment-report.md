## 神经网络模型的搭建：

#### 1. 数据的预处理
将data 降到一维，然后将label进行 `one-hot encoding`：

     def reformat(dataset, labels):
       dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
       # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
       labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
       return dataset, labels

#### 2. tensorflow 的运作方式：
（1）graph: 创建需要的单元：输入，变量，计算，这些均作为计算图的节点，这些包含在如些的模块中：

    with graph.as_default():
	      …
        
常见的单元有：
- 使用`tf.constant`存储常量(dataset和label)， 使用 `tf.variable` 存储变量(W和b) 
- 使用`tf.matmul`实现矩阵相乘，
- 使用`tf.nn.softmax_cross_entropy_with_logits`计算W交叉熵
- 使用梯度下降最小化交叉熵：
    `optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)` 
    
(2) session: 
将计算单元graph传给session，调用通过 `session.run()` 运行这些操作单元。然数据在各个单元之间流动。
	
      with tf.Session(graph=graph) as session:
	       …
首先初始化所有参数，然后通过循环过程，不断的修正`W，b`。
  
	 with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		for step in range(num_steps):
			_, l, predictions = session.run([optimizer, loss, train_prediction])

Note：可以使用 test_prediction.eval() 来同步更新参数

#### 3. SGD： 
每次只取一小部分数据做训练，计算loss时，也只取一小部分数据计算loss，对应到程序中，即修改计算单元中的训练数据，每次输入的训练数据只有128个，随机取起点，取连续128个数据：

     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
     batch_data = train_dataset[offset:(offset + batch_size), :]
     batch_labels = train_labels[offset:(offset + batch_size), :]
   
由于这里的数据是会变化的，因此用`tf.placeholder`来存放这块空间：

     tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

####4. `relu`函数
使用`relu` 函数的效果和添加网络层数能够显著的提升神经网络的效果
总结： 我们可以使用`tf.nn.relu()` 对函数进行取`relu()`。



## 正则化:
#### 1. 使用`tf.nn.l2_loss(t)` 对神经网络中的各个W做正则化，并将其添加到`train_loss`上：

        l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + para_regul * loss
这里有个超参数β需要设置，设置的原则：将l2_loss加到train_loss上，每次的train_loss都特别大，几乎只取决于l2_loss，为了让原本的train_loss与l2_loss都能较好地对参数调整方向起作用，它们应当至少在同一个量级。比如观察不加l2_loss，step 0 时，train_loss在300左右；加l2_loss后， step 0 时，train_loss在300000左右，因此给l2_loss乘0.0001使之降到同一个量级。

#### 2. dropout： 
调用`nn.dropout()`,每次丢掉随机的数据，让神经网络每次都学习到更多，但也需要知道，这种方式只在我们有的训练数据比较少时很有效。这里有一个参数保留概率，即我们要保留的RELU的结果所占比例，tensorflow建议的语法是，让它作为一个`placeholder`，在run时传入。当然我们也可以不用placeholder，直接传一个0.5。

#### 3. dency learning rate
随着训练次数增加，自动调整步长的语法为：

    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)




#### 4. 多隐层神经网络事项：
（1）代码复用:

      def weight_var(layer_num1,layer_num2):
          return tf.Variable(tf.truncated_normal([layer_num1, layer_num2], stddev=np.sqrt(2.0 / layer_num1)))

      def bias_var(layer_nums):
          return tf.Variable(tf.zeros([layer_nums]))

      def compute_logits(input_data, weightss, biasess, dropout_vals=None):  
          temp = input_data  
          if dropout_vals:  
             for w,b,d in zip(weightss[:-1], biasess[:-1], dropout_vals[:-1]):  
                  temp = tf.nn.relu_layer(tf.nn.dropout(temp, d), w, b)  
             temp = tf.matmul(temp, weightss[-1]) + biasess[-1]  
          else:  
              for w,b in zip(weightss[:-1], biasess[:-1]): 
                  temp = tf.nn.relu_layer(temp, w, b)  
              temp = tf.matmul(temp, weightss[-1]) + biasess[-1]  
          return temp

（2）选用 stddev 调整标准差的问题
由于网络层数的增加，寻找step调整方向时具有更大的不确定性，很容易导致loss变得很大，需要用stddev调整其标准差到一个较小的范围：再次我们使用

     stddev = np.sqrt(2.0/n)

（3）Regularization，Dropout
启用regular时，也要适当调一下β，不要让它对原本的loss造成过大的影响。dropout时，因为后面的layer得到的信息越重要，需要动态调整丢弃的比例，到后面的网络层，丢弃的比例要减小



## 深度卷积神经网络
#### 1. conv2d

    tf.nn.conv2d(input,filter,strides.padding,use_cudnn_on_gpu = None, data_format=None,name = None)

- input 和 filter：类型相同，为下列类型之一： half, float32, float64.
- strides: 长度4， 在input上切片采样时，每个方向上的滑窗步长，必须和format指定的维度同阶。
- padding：填充类型： 可选类型为“SAME”,”VALID”
- data_format: 执行输入输出数据的格式。可选类型：”NHWC”,”NCHW”,默认为”NHWC“
	- 当为"NHWC"时, 数据存储顺序为： [batch, in_hight,in_width,in_channels]
  - 当为"NCHW"时, 数据存储顺序为：[batch, in_channels, int_height, in_width]

#### Max Pooling
在tf.nn.conv2d后面接tf.nn.max_pool，将卷积层输出减小，从而减少要调整的参数。

    tf.nn.max_pool(value,ksize,strides,padding,data_format = ‘NHWC’,name= None)
    
 - ksize: A list of ints that has length >= 4. 要执行取最值的切片在各个维度上的尺寸
 - strides: A list of ints that has length >= 4. 取切片的步长.
