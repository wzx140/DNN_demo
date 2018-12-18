## Deep Neural Network
This is a deep neural network used to identify the cat picture. If the picture has a cat, the label is 1. If the picture has no cat, the label is 0.
### Build
If you use pipenv, just run `pipenv install`. Or install `numpy`, `h5py`, `scipy`, `pillow` by yourself
### Config
1. Put your image with name "data_xx"(xx is the number that show the order of the images) in the directory
2. Assignment the path of the directory to "pic_path" in *config.py*
3. change the setting of the neural network in *config.py*
> if you keep "pic_path" value None, the default path is `./pic`
### Start
Just run `python main.py` and then it will output the array of result (the order as same as the images)
### More
- It sometimes report an error the **RuntimeWarning: overflow encountered in exp**. Just ignore it
- For further information, you can read [my blog](https://wzx140.github.io/2018/12/15/deep-neural-network/)