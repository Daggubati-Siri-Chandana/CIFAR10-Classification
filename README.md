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

We finally use CNN classifier. 
