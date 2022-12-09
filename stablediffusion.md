
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

The second control we have is the number of sampling steps:

![image](https://user-images.githubusercontent.com/12407183/206691283-7c02f43e-9ada-485a-8862-fe9bf4d5e44f.png)

Everyimage starts as a blur, increasing the steps inccreases the iterations over which the image will be developed. The type of algorithm also affects the impact increasing the steps will have on the final image.


| ![image](https://user-images.githubusercontent.com/12407183/206692164-3878229c-dde6-48b4-91c7-97ef1ba5e2d9.png) | 
|:--:| 
| *LMS with 50 Steps* |

