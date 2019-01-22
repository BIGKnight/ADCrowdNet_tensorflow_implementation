# ADCrowdNet
by tensorflow
still working<br>
## PROGRESS
1.complete the custom deformable convolution op. I finished it by referencing these two repos:<br>
   <a>1).[msra official version by mxnet](https://github.com/msracver/Deformable-ConvNets)</a><br>
   <a>2).[a version of tf implements](https://github.com/Zardinality/TF-deformable-conv)</a>
   <br>in fact, the crucial kernels of the algorithm are almost same under different neural network frame.It's all cuda code, however, the c++ parts are different and these part had cost me a lot time to implements.<br>
   Besides, the new version of the deformable convolution has some new characteristics, and the 2) implement which is just mentioned above was coded 1 year ago, so its a v1 implements. And I implement the v2, although now only test the case: im2col_step = 1.
<br><br>2. the ADCrowdNet's author didn't give me any informations beyond the papers and they have not provided the negative data they mentioned in their paper which was essential in extracting the attention part. So the only thing I can do is to test the differences of effectiveness between the DME net and the DME_deformable net, and I will upload my result soon. May be 2 day later, I will finished soom.