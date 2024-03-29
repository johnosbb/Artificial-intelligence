
# Stable Diffusion

## References


- [Stable Diffusion - High-Resolution Image Synthesis with Latent Diffusion Models](https://github.com/CompVis/stable-diffusion)
- [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [Installation and Tutorial](https://www.youtube.com/watch?v=DHaL56P6f5M&list=PL3C_1qgacm_qPG_t1r1Fad7kggUCroWxs&index=1)
- [The Stable Diffusion Search Engine](https://lexica.art/)


## Installation

Clone the repository at 
- https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

```sh
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

```
- Install you ckpt model from huggingface into the stable-diffusion-webui\models\Stable-diffusion directory
- from the stable diffusion directory run: webui-user.bat

## Index

- [Stable Diffusion Prompts](stable-diffusion-prompts.md)
- [Stable Diffusion Inpaintings](stable-diffusion-inpainting.md)

## Models

- [mdjrny-v4 style](https://huggingface.co/prompthero/openjourney)
- [PaperCut](https://huggingface.co/Fictiverse/Stable_Diffusion_PaperCut_Model)


## Creating Images with Stable Diffusion

### Prompts

The prompt is your primary means of creating an image. Sample prompts can be found on [the Stable Diffusion Search Engine](https://lexica.art/). For example:

```txt
Goddess of all the goddesses from every place ever, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, unreal engine 5, 8k, art by ross tran and greg rutkowski and alphonse mucha
```
produces


![image](https://user-images.githubusercontent.com/12407183/206690770-29ab2020-cdf4-43be-8965-a29fe3368597.png)

Notice SD legendry difficulty with eyes and fingers. This can be addressed to some degree with:

We can use parenthesis to emphasise certain word so that they get more focus. Each additional pair of parenthesis adds more empahsis to the selected word.

```txt
Goddess of all the goddesses from every place ever, highly detailed, digital painting, artstation, concept art, smooth, (sharp focus), illustration, unreal engine 5, 8k, art by ross tran and greg rutkowski and ((alphonse mucha))
```

![image](https://user-images.githubusercontent.com/12407183/206697311-46645825-a879-414d-aa30-245335d0cd13.png)

We can see that the emphasis on ((alphonse mucha)) has changed the style to more closely follow that style.

- [Stable Diffusion prompting cheatsheet](https://moritz.pm/posts/parameters)

![image](https://user-images.githubusercontent.com/12407183/206690910-e1ccc00b-1702-4a21-9432-0dd906e72ee8.png)

The items at the start of the prompt generally have a more significant impact than those at the end.

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

### Width and Height

The Widthh and Height parameters change the resolution, but remember the computational effort required also increases. The Stable Diffusion image library has been trained on images with a size of 512x512 so best results are obtained by staying at this resolution and using upscaling later to increase the image size.

### Batch Size

Batch size controls the number of images that will be generated.

When we run batches each additional image gets a new seed so to re-iterate an image you must use the appropriate seed for the image you have selected.

![image](https://user-images.githubusercontent.com/12407183/206695280-d6707a6f-3f39-4392-bd97-39a6ce926f88.png)



### CFG Scale

![image](https://user-images.githubusercontent.com/12407183/206694211-10ed0d72-b09f-47df-888e-35ac932adcf6.png)

The scale determines the degree to which Stable Diffusion will use the information in your prompt to influence the creation of your image. The maximum value (30) will track very closely while the minimum value will largely ignore the prompt.

We can control this loyalty to the prompt by using parenthesis around certain elements of the prompt.


| ![image](https://user-images.githubusercontent.com/12407183/206698168-6f3b5d5b-6ae9-4f56-9640-5079336a24c9.png)) | ![image](https://user-images.githubusercontent.com/12407183/206699515-758bbbf3-bbec-4042-a234-c90867b74fb8.png) | ![image](https://user-images.githubusercontent.com/12407183/206698785-7615ec33-c387-49c7-ac60-f93b70827c2b.png) | 
|:--:|:--:|:--:| 
| *CFG Scale of 1 using LMS at 70 Steps* |  *CFG Scale of 15 using LMS at 70 Steps* |*CFG Scale of 30 using LMS at 70 Steps* |

Notice that with a very low scale value or image does not resemble our prompt. If we push the scale value to the maximum we risk introducing noise into the image. The amount of this noise varies greatly depending on the algorithm used.

## Image to Image

We can further refine images with Stable Diffusion by sending the image to the "Image to Image" processing engine.

Let us say we create the image below:

| ![image](https://user-images.githubusercontent.com/12407183/206700189-25f6e7be-b927-41d2-beb9-5622a000b5a8.png) |
|:--:|
| Goddess of all the goddesses from every place ever, highly detailed, digital painting, artstation, concept art, smooth, (sharp focus), illustration, unreal engine 5, 8k, art by ross tran and greg rutkowski and ((alphonse mucha))
Steps: 70, Sampler: LMS, CFG scale: 8, Seed: 3376993924, Size: 512x512, Model hash: 7460a6fa  |

### Denoising
In the Image to Image window we have a new setting called denoising:

![image](https://user-images.githubusercontent.com/12407183/206700717-e1e459e4-6780-4c5e-98cf-bda17b6e9cc1.png)

A denoising value of 0 would mean that we generate the same input image. Increasing the denoising strenght moves us further away from our starting image.

| ![image](https://user-images.githubusercontent.com/12407183/206701411-2717f6d6-d740-423a-806a-c6f1a0587652.png) |
|:--:|
| *Goddess alphonse mucha ocean in background Steps: 70, Sampler: Euler a, CFG scale: 7, Seed: 3376993924, Size: 512x512, Model hash: 7460a6fa, Denoising strength: 0.75, Mask blur: 4*  |

We have moved quite far from our original image. We can control this by reducing the denoising factor.

| ![image](https://user-images.githubusercontent.com/12407183/206702110-10155767-be77-4969-acdc-2b7095676096.png) | ![image](https://user-images.githubusercontent.com/12407183/206702506-1c9b7948-7754-4b97-917f-877f8637c8a3.png) |
|:--:|:--:|
| *Goddess alphonse mucha ocean in background Steps: 70, Sampler: Euler a, CFG scale: 7, Seed: 3376993924, Size: 512x512, Model hash: 7460a6fa, Denoising strength: 0.42, Mask blur: 4* | *Goddess alphonse mucha ocean in background Steps: 70, Sampler: Euler a, CFG scale: 7, Seed: 3376993924, Size: 512x512, Model hash: 7460a6fa, Denoising strength: 0.15, Mask blur: 4* |

We are not really getting the desired "ocean in background) so I increase the denoise value and the CFG Scale so that I follow the prompt more closely and I allow the algorithm the permission to deconstruct and reconstruct the image a little more.

! ![image](https://user-images.githubusercontent.com/12407183/206703589-e5f9c074-f02b-4ddb-9ce1-1e92f967af20.png) |
|:--:|
| *Goddess alphonse mucha (ocean in background)
Steps: 70, Sampler: Euler a, CFG scale: 12, Seed: 3376993924, Size: 512x512, Model hash: 7460a6fa, Denoising strength: 0.37, Mask blur: 4* |

## Inpainting

One can further control the creative process by using inpainting to mask out aspects of my image I may wish to keep.

![image](https://user-images.githubusercontent.com/12407183/206738673-b1588ca2-0e6d-4812-a722-14c5d909121f.png)

The inpaint area can be either masked or not masked. If you wish to preserve what you have masked then select inpaint not masked. We can also increase the denoise value and reduce the scale as they masked area will not be affected.

![image](https://user-images.githubusercontent.com/12407183/206738601-080e53d7-507c-41be-9ed8-01c714fa1e48.png)


# Merging Models in Stable Diffusion

For merging models we use the Checkpoint Merger tab.

![image](https://user-images.githubusercontent.com/12407183/208090501-dd39bab4-16bb-4da6-a9b5-4a4b3508429d.png)

We can merge upto three models at once. 

![image](https://user-images.githubusercontent.com/12407183/208091510-c5dcf7c2-30ff-4a1a-96d7-dd07dc40e897.png)

The primary model is normally the model you have trained in Dreambooth, the secondary model is the model you wish to mix your model into. This secondary model could be a main stream model or a model that complements your existing model.

Give your model a custom name that reflects the models you are mixing. This will help when you later try to identify the model's origins.

The Multiplier setting determines how much of model B will be merged into model A. A value of .3 means 30% of B will be merged into A.

![image](https://user-images.githubusercontent.com/12407183/208097364-95ba25ca-d7e2-4fd6-ba05-467cd84a23ef.png)

When you press the **run** button the models will be merged and the resultant model will be placed in your stable diffusion models directory.



# Image Upscaling

- [tiny wow upscaler](https://tinywow.com/image/upscale)
