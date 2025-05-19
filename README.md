# mnist
基于CNN的mnist手写数字识别


MNIST Handwritten Digit Recognition with CNN

Project Structure
```
data/
├── data/
│   └── MNIST/
│       └── MNIST/
│           └── RAW/  # Contains MNIST dataset files
│               ├── t10k-images-idx3-ubyte
│               ├── t10k-images-idx3-ubyte.gz
│               ├── t10k-labels-idx1-ubyte
│               ├── t10k-labels-idx1-ubyte.gz
│               ├── train-images-idx3-ubyte
│               ├── train-images-idx3-ubyte.gz
│               ├── train-labels-idx1-ubyte
│               └── train-labels-idx1-ubyte.gz
├── mnist_cnn_model.h5  # Trained CNN model
└── train.py            # Main Python script
```

Requirements
• Python 3.x

• Required packages:

  • TensorFlow (includes Keras)

  • NumPy

  • OpenCV (cv2)

  • Matplotlib (for image display)


Setup Instructions

1. Create Virtual Environment (Optional but recommended):
   ```bash
   python -m venv mnist_env
   source mnist_env/bin/activate  # Linux/Mac
   mnist_env\Scripts\activate     # Windows
   ```

2. Install Dependencies:
   ```bash
   pip install tensorflow numpy opencv-python matplotlib
   ```

Project Description

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The implementation includes:

• Loading and preprocessing MNIST data from raw binary files

• CNN model architecture with convolutional and dense layers

• Model training functionality

• Prediction interface for custom handwritten digit images

• Interactive command-line interface for user interaction


Usage

1. Run the main script:
   ```bash
   python train.py
   ```

The script provides an interactive menu with these options:
• Recognize a custom handwritten digit image

• Retrain the CNN model

• Exit the program


File Descriptions

• `train.py`: Contains all project code including data loading, model definition, training, and prediction functions

• `mnist_cnn_model.h5`: Saved model weights after training

• `data/`: Directory containing the raw MNIST dataset files


Note

The project expects MNIST data files in their original binary format (.ubyte files) in the specified directory structure. For custom predictions, provide clear images of handwritten digits (black digits on white background work best).
