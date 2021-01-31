# Image-Colorization
Convert a greyscale image to a Colored image

# Description and Explaination:
In this model we have used the latest neural network which mainly involves using the Resnet for fundamental feature extraction and thereby combining it or rather merging it with a layer which introduces iself after the pooling layer of the input. We then decode it with the convolutional layers through upsampling in order to predict the new coloured image. We can also use various types of GAN such as Cycle gan and pix2pix GAN which are currently used for advanced image colourization.

In this model we have used an Auto-encoder which also becomes the generator in the case of the GAN network. 

The following image explains the architecture currently used in the model:
![Encoder_Image1](https://user-images.githubusercontent.com/77839791/106395130-9538e200-6426-11eb-82d4-06e88dfdd333.jpg)

There are majorly two techniques which could be used in order generate coloured images from grey scaled images:
1. In order to perform the above task , we first convert the RGB image into a LAB image in which we separate the L value and ab value from the image and then we train the model in order to predict the value of ab.
2. The second way we can proceed is to turn the RGB image into the a LUV image in which we can separate the L value and uv value from the image and then we train the model in order to predict the value of uv.

** L - lightness , 'a' and 'b' stand for two color spectra green-red and blue-yellow respectively.

# Points to Note:

1. The dataset which we have used consists of all coloured images and in order to use our dataset efficiently we need to use the black and white versions of the images present. So thereby we begin by starting to convert the coloured image into greyscaled image and then returning it back to it's RGB format to complete our entire process.
2. There are several codes present in the model which utilizes the common techniques used in image colourization.

# Conclusion:

1. The end result of this model is that it is able to identify the ideal or correct patterns or the base shades in which the colour must be filled.
