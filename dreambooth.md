# Stable Diffusion Addon - Dream Booth

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
