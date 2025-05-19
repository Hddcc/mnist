import os
import gzip
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def load_mnist(path, kind='train'):
    """加载原始MNIST格式的数据"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
    
    return images, labels

def train_and_save_model():
    """训练CNN模型并保存"""
    # 加载数据
    print("正在加载MNIST数据...")
    x_train, y_train = load_mnist('/public/home/hpc244712216/userfolder/20250509_mnist_dengdi/data/data/MNIST/MNIST/raw', kind='train')
    x_test, y_test = load_mnist('/public/home/hpc244712216/userfolder/20250509_mnist_dengdi/data/data/MNIST/MNIST/raw', kind='t10k')

    # 数据预处理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)  # 添加通道维度 (60000, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28, 1)

    # 构建CNN模型
    print("构建CNN模型...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    print("开始训练模型...")
    history = model.fit(x_train, y_train, 
                        epochs=10, 
                        batch_size=128, 
                        validation_data=(x_test, y_test),
                        verbose=1)

    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\n测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}')

    # 保存模型
    model.save('mnist_cnn_model.h5')
    print("模型已保存为 mnist_cnn_model.h5")
    return model

def preprocess_image(image_path):
    """
    预处理手写数字图片：
    1. 读取为灰度图
    2. 调整大小为28x28
    3. 反转颜色（MNIST是白底黑字）
    4. 归一化
    """
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 调整大小
    img = cv2.resize(img, (28, 28))
    
    # 反转颜色（如果你的图片是黑底白字）
    img = 255 - img
    
    # 归一化并reshape为模型需要的格式
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # 添加通道维度
    img = np.expand_dims(img, axis=0)   # 添加batch维度
    
    return img

def predict_custom_image(model, image_path):
    """使用训练好的模型预测自定义图片"""
    try:
        # 预处理图片
        processed_img = preprocess_image(image_path)
        
        # 进行预测
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        print(f"\n预测结果: 数字 {predicted_class}, 置信度: {confidence:.2%}")
        
        # 显示图片
        import matplotlib.pyplot as plt
        plt.imshow(processed_img[0, :, :, 0], cmap='gray')
        plt.title(f"预测: {predicted_class} (置信度: {confidence:.2%})")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"\n处理图片时出错: {e}")
        print("请确保:")
        print("1. 图片路径正确")
        print("2. 图片是手写数字的清晰图片")
        print("3. 如果图片是彩色或大小不符，代码会自动调整但效果可能受影响")

def main():
    # 检查模型是否已存在
    if os.path.exists('mnist_cnn_model.h5'):
        print("检测到已保存的模型，正在加载...")
        model = load_model('mnist_cnn_model.h5')
    else:
        print("未找到已保存的模型，开始训练新模型...")
        model = train_and_save_model()
    
    # 交互式预测
    while True:
        print("\n" + "="*50)
        print("手写数字识别系统")
        print("1. 识别自定义图片")
        print("2. 重新训练模型")
        print("3. 退出")
        choice = input("请选择操作 (1/2/3): ")
        
        if choice == '1':
            image_path = input("请输入图片路径: ").strip()
            if os.path.exists(image_path):
                predict_custom_image(model, image_path)
            else:
                print("错误: 文件不存在")
        elif choice == '2':
            model = train_and_save_model()
        elif choice == '3':
            print("退出程序")
            break
        else:
            print("无效输入，请重新选择")

if __name__ == "__main__":
    main()