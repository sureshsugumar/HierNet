# HierNet
A Tensorflow implementation of Hierarchical Neural Network referencing Capsules Net in Hinton's paper Dynamic Routing Between Capsules
MNIST database is used as data set, and test set includes both standard test set from the dataset and also generated image sets that are specifically altered to test the network - such as rotated, superimposing 1 or more digits on top of digit, etc.

Steps:
1. cd code
2. Train the network
  - python train.py
3. Test on standard MNIST dataset
  - a mp4 video will be produced in the video folder
4. Test on custom images
  - images in the images folder will be processed and outputs displayed on console
