# Feature aware normalisation network

Bug et al. suggested this model in 2017 as a good way to normalise histopathology images. More information is available on their [paper](https://export.arxiv.org/pdf/1708.04099).

The core ideia is quite clever - an initial mapping $T$ from the image to a feature space with $d$ channels is scaled and shifted $n$ times by features extracted from a network pretrained (in my case) on ImageNet whose layers I freeze during training (not doing this makes results a bit more unpredictable). After this, the network is remapped to the original colour space by $T^{-1}$. For clarity, $T$ and $T^{-1}$ are both parameterised as convolutional layers with a size 1 convolution.

In this repository, I show a network implements roughly the same as Bug et al. but using a MobileNetV2 rather than VGG19 (MobileNetV2 is faster and I was interested in maximising speed). I use, for this, four of the internal representations contained in MobileNetV2 (the output from blocks 3, 5, 7 and 9). However, the FAN model class is implemented in such a way that it takes a model producing $n$ outputs and constructs a feature-aware normalisation model that infers the correct number of FAN layers and their respective resizing operations.
