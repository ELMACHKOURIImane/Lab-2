# Lab-2
in this lab we will buil the neural architectures CNN and faster RCNN and VIT for computer vision. 

# the first part : Convolutional neural network 

in this part i had build a CNN architecture for the MNIST dataset, which contains grayscale images of handwritten digits from 0 to 9.


# Model Architecture:
Input Layer: The input images have a single channel (grayscale), so the input layer is defined with nn.Conv2d(1, 6, 5), which means it takes one input channel, generates six output channels, and uses a kernel size of 5x5.
Pooling Layers: Max-pooling layers (nn.MaxPool2d(2, 2)) are used to downsample the spatial dimensions of the feature maps.
Convolutional Layers: Two convolutional layers are defined (nn.Conv2d(6, 16, 5)), where the first layer has 6 input channels and 16 output channels, and the second layer has 16 input channels and 16 output channels.
Fully Connected Layers: There are three fully connected layers (nn.Linear) with ReLU activation functions.
The first fully connected layer (self.fc1) takes the flattened output of the second convolutional layer (16 * 4 * 4), implying a feature map size of 4x4 after pooling.
The second fully connected layer (self.fc2) has 120 output features.
The third fully connected layer (self.fc3) produces the final output, which is a vector of size 10, representing the probability distribution over the 10 classes (digits 0 to 9).
# Forward Pass:
The input image passes through the convolutional layers with ReLU activation functions, followed by max-pooling layers to reduce spatial dimensions.
The output feature maps are flattened and passed through fully connected layers with ReLU activation functions.
The final layer produces logits for each class, which are passed through a softmax function during inference to obtain class probabilities.
# Training:
Cross-entropy loss (nn.CrossEntropyLoss()) is used as the loss function, which is suitable for multi-class classification tasks.
The Adam optimizer (optim.Adam) is used for optimizing the model parameters with a learning rate of 0.001.
The model is trained for 10 epochs using mini-batch gradient descent (DataLoader) with a batch size of 64.
During training, the loss is printed every 100 mini-batches to monitor training progress.
# Evaluation:
After training, the model is tested on the test dataset to evaluate its accuracy.
The accuracy of the model on the test set is printed at the end of the script.
