# Handwritten Digit Classification with LeNet-5 Inspired CNN on MNIST
To built a compact LeNet-5 CNN using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The project includes data preprocessing, model architecture design, training, and performance evaluation—demonstrating a practical deep learning workflow for image classification.

### Basic Information

* **Person or organization developing model**: Patrick Hall, `jphall@gwu.edu` & N M Emran Hussain `nmemran.hussain@gwu.edu`
* **Model date**: June, 2025
* **Model version**: 1.0 
* **License**: [Apache License 2.0](https://github.com/nmemranhussain/6290_PAI_2/blob/main/LICENSE)
* **Model implementation code**: [Assignment_3](https://github.com/nmemranhussain/6290_PAI_2/blob/main/Assignment_3_final.ipynb), 

### Intended Use
* **Primary intended uses**: This project is designed as an educational and practical exercise in applying deep learning techniques to image classification tasks. Specifically, it aims to help learners understand how convolutional neural networks (CNNs)—inspired by the LeNet-5 architecture—can be built, trained, and evaluated using real-world datasets like MNIST. It demonstrates key steps such as data preprocessing, network design using TensorFlow/Keras, and performance evaluation, making it suitable for students, early practitioners, or anyone exploring computer vision fundamentals.
* **Out-of-scope use cases**: This project is not intended for production deployment or use in real-time digit recognition systems. It does not include advanced features like hyperparameter tuning, model optimization for edge devices, or comparison across multiple deep learning architectures. Additionally, it does not address scalability, adversarial robustness, or explainability in model predictions—topics that are essential for deploying CNNs in safety-critical or commercial applications.

### Training Data

* Data dictionary:

| Field Name | Description | Data Type | Example |  
|-----------|--------------|------------|----------|
| x_train	| Training images grayscale pixel values of handwritten digits (0–9) | ndarray (60000, 28, 28, 1) |	28×28 pixel matrix |  
| y_train	| Training labels; corresponding digit class for each image	| ndarray (60000, ) before encoding (60000, 10) after one-hot encoding	| 5 or [0,0,0,0,0,1,0,0,0,0] |  
| x_test	| Test images; grayscale pixel values of handwritten digits (0–9)	| ndarray (10000, 28, 28, 1) | 28×28 pixel matrix |  
| y_test	| Test labels; digit class for each test image	| ndarray (10000, ) before encoding (10000, 10) after one-hot encoding | 2 or [0,0,1,0,0,0,0,0,0,0]|  
| pixel values	| Intensity values ranging from 0 (black) to 255 (white), later normalized to [0,1]	| float32 after normalization	| 0.00 – 1.00 |  

