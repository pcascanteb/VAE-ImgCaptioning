# Variational Image Captioning Using Deterministic Attention Implementation.

Baseline model for the image captioning with the attention mechanism: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

Implementation code for the proposed model. The attempt was to generate diverse image descriptions for each input image using a Conditional Variational Autoencoder (CVAE) and deterministic attention. 
We use beam search for caption generation. Given the same image input and the same beam size input, the model is able to generate different image descriptions.

Some examples:
![](https://github.com/pcascanteb/VAE-ImgCaptioning/blob/master/Imgs/Samples.JPG?raw=true "Title")

The first caption is generated by the baseline model. The other captions are generated using our model. Note that our model also generates the baseline caption, but in addition is able to figure out more fine grained and general characteristics of the image for its corresponding description.
