## Deep Neural Network
This is a deep neural network used to identify the cat picture. If the picture has a cat, the label is 1. If the picture has no cat, the label is 0.
### Build
If you use pipenv, just run `pipenv install`. Or install `numpy`, `h5py`, `scipy`, `pillow` by yourself
### Config
1. Put your image with name "data_xx"(xx is the number that show the order of the images) in the directory
2. Assignment the path of the directory to "pic_path" in *config.py*
3. change the setting of the neural network in *config.py*
> if you keep "pic_path" value None, the default path is `./pic`
#### Regularization
- change `lambd` or `keep_prob` to enable *L2* or *dropout*
- if `lambd` is 0, it means *L2* is disable
- if `keep_prob` is 1, it means *dropout* is disable
- you **can not** set both *L2* and *dropout* enabled
### Start
Just run `python main.py` and then it will output the array of result (the order as same as the images)
### More
- It sometimes report an error the **RuntimeWarning: overflow encountered in exp**. Just ignore it
- For further information, you can read [my blog](https://masterwangzx.com/2018/12/15/deep-neural-network/)