# Stable Diffusion Addon - Dream Booth

## Using Colab

- [Colab Link](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb#scrollTo=O3KHGKqyeJp9)

Click on the first option to connect google drive
Click on the generate dependencies link and wait for the green tick to appear

![image](https://user-images.githubusercontent.com/12407183/206867762-0f10abf5-f7ff-4125-9f7a-aa98a249f44a.png)

A green tick mark will appear once these steps complete

Now paste in the HuggingFace token and press play:

![image](https://user-images.githubusercontent.com/12407183/206867925-5cf46ad1-9d39-4e06-b82e-6753e7258cbd.png)

Wait for the green tick to appear to confirm that the model has downloaded

![image](https://user-images.githubusercontent.com/12407183/206868045-e6d00065-1442-4eb5-b9e6-3ebb0d764238.png)

Give the session a name and press play again

![image](https://user-images.githubusercontent.com/12407183/206868102-122e62ce-a14b-4f56-8c07-91338fc9e170.png)

Now click the Session IMages play button and wait for the **choose files** button to appear.

![image](https://user-images.githubusercontent.com/12407183/206868180-c9664b63-c052-4a3e-bcd5-04f1fc4bcec6.png)

Once the images have uploaded we set the number of steps. As a guide, we use 100 steps for each image uploaded

![image](https://user-images.githubusercontent.com/12407183/206868298-f4e9b557-bdd0-4686-8e7c-67c90b5a68ab.png)

The new model will appear in your google drive once the training has completed.



## Installing Locally

Go to the Extensions tab and select Load From:

Once the extensions appear, select Dream Booth

![image](https://user-images.githubusercontent.com/12407183/206799012-67dba0d9-f2b4-487a-a75a-f8ceeeedd55d.png)

This will add the dreambooth tab

![image](https://user-images.githubusercontent.com/12407183/206800119-23390cba-32e9-4d0f-878e-156f1928164e.png)

We create a new model using the default model.ckpt as the source checkpoint.

![image](https://user-images.githubusercontent.com/12407183/206800377-61178728-c508-473b-8d1a-c47ddea32195.png)


After creating the model we can configure the training parameters.

![image](https://user-images.githubusercontent.com/12407183/206865827-d1100936-ba1e-495e-9658-e36d2fce2673.png)
![image](https://user-images.githubusercontent.com/12407183/206865854-3ef3f0c5-060d-44ab-9f61-2af707a4cd54.png)


- The Dataset Directory contains the data against which we wish to train the model.
- The instance prompt is a prompt used to generate this instance. It should not be a generic term like dog or man but something unique to the instance you are going to create.
- We could also supply a **Class Prompt**, this could be something generic like: man, dog, boat etc. If we decide to use this we can specifiy an additional path that allows us reference class images. These could be selected images of our choosen class, for example images of men or dogs etc.
- 
