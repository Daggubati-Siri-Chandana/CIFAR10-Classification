# CIFAR10-Classification

For the CIFAR 10 Classification, we compare the following techniques and their accuracies.
- SIFT/SURF BVOW Classification
- SIFT/SURF VLAD Classification
- CNN Classification

### SIFT/SURF BVOW Classification

The first classification technique is BOVW(Bag of visual words). In this technique we compute SIFT/SURF descriptors of image then compute n clusters using k means over the descriptor list and generate  histogram for n centers. The frequency list of these n centers is the final feature vector. The test and validation accuracies in BVOW technique using KNN Classifier is 13.7% and 39% respectively. 

Drawbacks of BVOW Classification:
- Under fitting i.e., the feature vector, is not rich enough to capture features of all ten classes. 
- This is one of the reasons for better performance binary classification over ten class classification.
- The small size of the image causes fewer descriptors/key points of the image; these descriptors of the image give rise to the feature vector, which gives the frequency of each cluster.

### SIFT/SURF VLAD Classification

The next classification technique is VLAD (vector of locally aggregated descriptors), which is an extension to BOVW where we focus on getting a better feature vector. The vectors obtained using this method are similar to the Fisher vectors.The test and validation accuracies in VLAD technique using KNN Classifier is 23% and 45% respectively. 

Drawbacks of BVOW Classification:
- The feature vectors of VLAD perform better over BVOW because of it the centralized representation of descriptors around the cluster center. Unlike frequency representation, the local aggregated descriptor representation gives more information regarding the image.
- VLAD can't classify of CIFAR 10 dataset well. As already mentioned, the images of CIFAR 10 produces fewer descriptors of the image than expected, maybe because of the low resolution/small size of the image. 

### CNN Classification

We finally use CNN classifier. The Convolution neural network contains 6 Convolution layers and two fully connected layers. Then each convolution layer is followed by a batch normalization layer. For each pair of convolutional layers, we have max-pooling layers to reduce the spatial size of output, thereby reducing the computational cost. A max-pooling layer is followed by a dropout layer of 20% to bring the regularization effect.
 

We used adaptive learning rate algorithm RMSProp with learning rate as 0.001 and rho value 0.9, batch size of 128, and 50 epochs. We then computed cross- entropy loss. We used the early stopping and checkpoints mechanism to save the best model.
A plot of accuracy and loss function for training and validation.We finally achieved 97% training accuracy and 84% validation accuracy on 50 epochs. I took 4.5 hours to train the network on CPU with 8 GB RAM and Intel i5 processor.
