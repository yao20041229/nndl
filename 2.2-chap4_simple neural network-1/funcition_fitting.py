import numpy as np  # 导入NumPy库用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库用于绘图

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络参数，包括权重和偏置。
        - input_size: 输入层神经元个数
        - hidden_size: 隐藏层神经元个数
        - output_size: 输出层神经元个数
        """
        # 使用 Xavier 初始化方法，防止梯度消失或梯度爆炸
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        
        # Adam 优化器参数
        self.vW1, self.vb1, self.vW2, self.vb2 = 0, 0, 0, 0  # 一阶矩估计（动量项）
        self.sW1, self.sb1, self.sW2, self.sb2 = 0, 0, 0, 0  # 二阶矩估计（自适应学习率）
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8  # Adam 超参数
    
    def relu(self, Z):
        """ReLU 激活函数"""
        return np.maximum(0, Z)

    def d_relu(self, Z):
        """ReLU 激活函数的导数"""
        return Z > 0
    
    def forward(self, x):
        """
        前向传播计算
        - x: 输入数据
        返回输出层结果
        """
        self.Z1 = x.dot(self.W1) + self.b1  # 计算隐藏层加权输入
        self.A1 = self.relu(self.Z1)  # 通过 ReLU 激活函数
        self.Z2 = self.A1.dot(self.W2) + self.b2  # 计算输出层加权输入
        return self.Z2  # 直接输出，适用于回归任务

    def compute_loss(self, y_true, y_pred):
        """计算均方误差（MSE）损失函数"""
        return ((y_true - y_pred) ** 2).mean()
    
    def backward(self, x, y, learning_rate, t):
        """
        反向传播计算梯度，并使用 Adam 进行参数更新。
        - x: 训练数据
        - y: 真实值
        - learning_rate: 学习率
        - t: 当前迭代次数（用于 Adam 偏差校正）
        """
        m = y.shape[0]  # 训练样本数
        
        # 计算输出层梯度
        dZ2 = self.Z2 - y
        dW2 = self.A1.T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        
        # 计算隐藏层梯度
        dZ1 = dZ2.dot(self.W2.T) * self.d_relu(self.Z1)
        dW1 = x.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m
        
        # Adam 优化器 - 一阶矩估计
        self.vW1 = self.beta1 * self.vW1 + (1 - self.beta1) * dW1
        self.vb1 = self.beta1 * self.vb1 + (1 - self.beta1) * db1
        self.vW2 = self.beta1 * self.vW2 + (1 - self.beta1) * dW2
        self.vb2 = self.beta1 * self.vb2 + (1 - self.beta1) * db2
        
        # Adam 优化器 - 二阶矩估计
        self.sW1 = self.beta2 * self.sW1 + (1 - self.beta2) * (dW1 ** 2)
        self.sb1 = self.beta2 * self.sb1 + (1 - self.beta2) * (db1 ** 2)
        self.sW2 = self.beta2 * self.sW2 + (1 - self.beta2) * (dW2 ** 2)
        self.sb2 = self.beta2 * self.sb2 + (1 - self.beta2) * (db2 ** 2)
        
        # 偏差校正
        vW1_corr = self.vW1 / (1 - self.beta1 ** t)
        vb1_corr = self.vb1 / (1 - self.beta1 ** t)
        vW2_corr = self.vW2 / (1 - self.beta1 ** t)
        vb2_corr = self.vb2 / (1 - self.beta1 ** t)
        
        sW1_corr = self.sW1 / (1 - self.beta2 ** t)
        sb1_corr = self.sb1 / (1 - self.beta2 ** t)
        sW2_corr = self.sW2 / (1 - self.beta2 ** t)
        sb2_corr = self.sb2 / (1 - self.beta2 ** t)
        
        # 更新参数
        self.W1 -= learning_rate * vW1_corr / (np.sqrt(sW1_corr) + self.epsilon)
        self.b1 -= learning_rate * vb1_corr / (np.sqrt(sb1_corr) + self.epsilon)
        self.W2 -= learning_rate * vW2_corr / (np.sqrt(sW2_corr) + self.epsilon)
        self.b2 -= learning_rate * vb2_corr / (np.sqrt(sb2_corr) + self.epsilon)
    
    def train(self, x_train, y_train, epochs, learning_rate, decay_rate=0.001, patience=10000):
        """训练神经网络，并使用早停机制"""
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(x_train)
            loss = self.compute_loss(y_train, y_pred)
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}, Loss: {best_loss}")
                break
            
            if epoch % 5000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Learning Rate: {learning_rate:.6f}")
            
            self.backward(x_train, y_train, learning_rate, epoch)
            learning_rate *= (1. / (1. + decay_rate * epoch))

# 目标函数（sin 函数）
def target_function(x):
    return np.sin(x)

# 生成训练和测试数据集
x_train = np.linspace(-np.pi, np.pi, 700).reshape(-1, 1)
y_train = target_function(x_train)
x_test = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
y_test = target_function(x_test)

# 初始化并训练神经网络
model = SimpleNeuralNetwork(input_size=1, hidden_size=64, output_size=1)
model.train(x_train, y_train, epochs=100000, learning_rate=0.01, decay_rate=1e-5, patience=5000)

# 预测并可视化
y_pred = model.forward(x_test)
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, label='True Function')
plt.plot(x_test, y_pred, label='Predicted Function', linestyle='--')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
