# Replacing a face

Select the Inpaint tab and drop a 512x152 image into the Inpaint Window:

![image](https://user-images.githubusercontent.com/12407183/209564092-af71483e-7f19-4d74-9b64-b65063df76c0.png)


Using the inpainting brush, paint the section you wish to replace and select the "inpaint masked" option.

![image](https://user-images.githubusercontent.com/12407183/209564220-62d482fb-194a-40d4-a4c9-25da2f8798d2.png)


Set the rest of the settings as follows:

- Masked Content: Original
- Inpaint at full resolution: Enabled
- Inpaint at full resolution padding, pixels: 90
- Sampling Method: Euler a at 30 sampling steps or DDIM at 70 sampling steps
- Width: 640
- Height: 640
- Restore Faces: Enabled
![image](https://user-images.githubusercontent.com/12407183/209564620-922ca3a3-7924-42a8-8f9f-bccbd9a5b85a.png)
- Batch size: 1
- CFG Scale: 8
- Denoising Strength: 0.5 - 0.75
- Seed: -1

Prompt: Only right what you wish to change.
