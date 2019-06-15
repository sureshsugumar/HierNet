# HierNet
A Tensorflow implementation of Hierarchical Neural Network referencing Capsules Net in Hinton's paper Dynamic Routing Between Capsules
MNIST database is used as data set, and test set includes both standard test set from the dataset and also generated image sets that are specifically altered to test the network - such as rotated, superimposing 1 or more digits on top of digit, etc.

Steps:
1. cd code
2. Train the network
  -- python train.py  (training stats will be displayed in console)
3. Test on standard MNIST dataset
  -- python test.py  (a mp4 video will be produced in the video folder)
4. Test on custom images
  -- python test_image.py  (images in the images folder will be processed and outputs displayed on console)

Acknowledgements: This project has referenced many other respositories for code, and reused some of the code with extensive modifications. My sincere thanks to all of those repository authors, without which this project would not have existed. This is a open community anyway!

Cheers
Suresh
