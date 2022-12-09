
# Stable Diffusion

## References


- [Stable Diffusion - High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/CompVis/stable-diffusion)
- [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Installation and Tutorial](https://www.youtube.com/watch?v=DHaL56P6f5M&list=PL3C_1qgacm_qPG_t1r1Fad7kggUCroWxs&index=1)
- [The Stable Diffusion Search Engine](https://lexica.art/)



## Creating Images with Stable Diffusion

The prompt is your primary means of creating an image. Sample prompts can be found on [the Stable Diffusion Search Engine](https://lexica.art/). For example:

```txt
Goddess of all the goddesses from every place ever, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, unreal engine 5, 8k, art by ross tran and greg rutkowski and alphonse mucha
```
produces


![image](https://user-images.githubusercontent.com/12407183/206690770-29ab2020-cdf4-43be-8965-a29fe3368597.png)

Notice SD legendry difficulty with eyes and fingers. This can be addressed to some degree with:

![image](https://user-images.githubusercontent.com/12407183/206690910-e1ccc00b-1702-4a21-9432-0dd906e72ee8.png)


### Steps

The second control we have is the number of sampling steps:

![image](https://user-images.githubusercontent.com/12407183/206691283-7c02f43e-9ada-485a-8862-fe9bf4d5e44f.png)

Every image starts as a blur, increasing the steps inccreases the iterations over which the image will be developed. The type of algorithm also affects the impact increasing the steps will have on the final image.


| ![image](https://user-images.githubusercontent.com/12407183/206692164-3878229c-dde6-48b4-91c7-97ef1ba5e2d9.png) | ![image](https://user-images.githubusercontent.com/12407183/206692593-92378342-3f31-4c39-8a60-c8937eb6c5d3.png) |
|:--:|:--:| 
| *LMS with 50 Steps* | *LMS with 20 Steps* |


### Seeding

When an image is generated it produces a seed.
![image](https://user-images.githubusercontent.com/12407183/206693382-4c54ac93-9f0a-4961-b201-16233e88f9e0.png)



We can use the seed to regenerate an image by copying that number to the Seed box.

![image](https://user-images.githubusercontent.com/12407183/206693324-121e9969-9abe-4138-a0c4-631bfa4a27ec.png)


### Batch Size

Batch size controls the number of images that will be generated.

When we run batches each additional image gets a new seed so to re-iterate an image you must use the appropriate seed for the image.

### CFG Scale

![image](https://user-images.githubusercontent.com/12407183/206694211-10ed0d72-b09f-47df-888e-35ac932adcf6.png)

The scale determines the degree to which Stable Diffusion will use the information in your prompt to influence the creation of your image.
