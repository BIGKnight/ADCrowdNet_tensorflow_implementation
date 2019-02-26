# ADCrowdNet
by tensorflow
still working<br>
## PROGRESS
1.complete the custom deformable convolution op. I finished it by referencing these two repos:<br>
   <a>1).[msra official version by mxnet](https://github.com/msracver/Deformable-ConvNets)</a><br>
   <a>2).[a version of tf implements](https://github.com/Zardinality/TF-deformable-conv)</a>
   <br>in fact, the crucial kernels of the algorithm are almost same under different neural network frame.It's all cuda code, however, the c++ parts are different and these part had cost me a lot time to implements.<br>
   Besides, the new version of the deformable convolution has some new characteristics, and the 2) implement which is just mentioned above was coded 1 year ago, so its a v1 implements. And I implement the v2, although now only test the case: im2col_step = 1.
<br><br>2. the ADCrowdNet's author didn't give me any informations beyond the papers and they have not provided the negative data they mentioned in their paper which was essential in extracting the attention part. So the only thing I can do is to test the differences of effectiveness between the DME net and the DME_deformable net, and I will upload my results soon. May be 2 days later, in fact, I almost finished now, just waiting for the results of the experiments

## RESTART THE PROJECT, 02/26/19
I'm really disappointed about the last result, because the data showed the deformable layer seemed no benefit toward crowd counting problem. I'm not sure if it was due to the complex and deep net structure or the small learning rate in the deformablt layer according to the V2 paper publicized by Dai, which said the deformable layer only have the 1/10 rate comparing to other layer. However, the final cognitive fields of every pixels in the output map showed barely difference with the version which removed the deformable structure. That is, the deformable fabric seemed no contribute to the result. 

But there is stll hope existing because I haven't test it with a larger learning rate since the Spring Festival. And the vacation was gone now which means I'll start again. 

Besides, I prepare to transplant it into the pytorch frameworks which have a kinder mechanism in customizing structure.

