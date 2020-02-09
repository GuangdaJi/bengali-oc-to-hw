# bengali-oc-to-hw
 generating handwriting characters from optical characters

## Intro
This is a domain transfer projects that uses GAN to transfer OC characters to handwriting characters.

## Idea
Generator: $G = g\circ (f+e)$, $f$ as encoder, maps 128x128 images to its features, $e$ is embedding noise dimension to add diversity, and $g$ is the decoder.

- For $f$, I use the denseblock in DenseNet121, without the final linear classifier layer (with ReLU activation, about 1061 features). I plan to pretrain f using the original classification task, but the figure used are both handwritten and optical. I think in this way, $f$ can capture the features for both optical and handwritten images.
- For $e$, the embedding dimmension, I plan to train it with $\mathcal{N}(0,1)$ noise. This additional dimension may add diversity to the output image, which may be thought as a way of augmentation.
- For $g$, the decoder, I plan to use transplose convolution, with batchnorm and leaky relu, to generate 128x128 gray scale images.


Discriminator $D$, I plan to use batchnormed convolution with ReLU activation, the output channel is $1$, since I only want to tell real/fake from this discriminator.

## Training
I plan to first train the encoder as the feature maps of a classifier, the labels function as guides, for encoder to find best features.

Than I plan to adversally train the discriminator and the decoder to generate deep fake images.

For fine tuning stage, the generator as a whole join training.