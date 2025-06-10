// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "rknn_api.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include "postprocess.h"

#define PERF_WITH_POST 1

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    char dims[128] = {0};
    for (int i = 0; i < attr->n_dims; ++i)
    {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, get_format_string(attr->fmt),
           get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static void *load_file(const char *file_path, size_t *file_size)
{
    FILE *fp = fopen(file_path, "r");
    if (fp == NULL)
    {
        printf("failed to open file: %s\n", file_path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    size_t size = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);

    void *file_data = malloc(size);
    if (file_data == NULL)
    {
        fclose(fp);
        printf("failed allocate file size: %zu\n", size);
        return NULL;
    }

    if (fread(file_data, 1, size, fp) != size)
    {
        fclose(fp);
        free(file_data);
        printf("failed to read file data!\n");
        return NULL;
    }

    fclose(fp);

    *file_size = size;

    return file_data;
}

/*
Load a single image from the given image_path and return the image data as a pointer to an array of unsigned char.
The image is loaded and potentially resized to meet the specified input requirements for an RKNN (Rockchip Neural Network) model.
The image is loaded with the specified number of channels (req_channel),
which can force the image to be loaded as RGB (3 channels) or RGBA (4 channels), depending on the network's requirements.
This function uses The stb library which is a collection of single-file,
open-source C libraries designed to be easily integrated into C/C++ projects.
 These libraries are created by Sean Barrett and are widely used for various purposes,
 including image loading, font rendering, and 2D graphics.
 The libraries are known for being simple to use, lightweight, and efficient.
 The image is returned from stbi_load can be in three formats:
 RGB (3 channels): The image data is stored in a contiguous block of memory in the format [R1, G1, B1, R2, G2, B2, ..., Rn, Gn, Bn].
Each pixel is represented by 3 consecutive values, one for each color channel (Red, Green, and Blue).
RGBA (4 channels): The image data is stored in the format [R1, G1, B1, A1, R2, G2, B2, A2, ..., Rn, Gn, Bn, An].
Each pixel has an additional alpha channel (A) for transparency.
Grayscale (1 channel): The image data is stored as [Gray1, Gray2, ..., Grayn].
Each pixel is represented by a single value for brightness.
*/
static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr, int *img_height, int *img_width)
{
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    switch (input_attr->fmt)
    {
    case RKNN_TENSOR_NHWC:
        req_height = input_attr->dims[1];
        req_width = input_attr->dims[2];
        req_channel = input_attr->dims[3];
        printf("Our model requires NHWC format with height=%d,width=%d,channels=%d\n", req_height, req_width, req_channel);
        break;
    case RKNN_TENSOR_NCHW:
        req_height = input_attr->dims[2];
        req_width = input_attr->dims[3];
        req_channel = input_attr->dims[1];
        printf("Our model requires NCHW format with height=%d,width=%d,req_channels=%d\n", req_height, req_width, req_channel);
        break;
    default:
        printf("We found an unsupported layout. Please ensure that the image conforms to NHWC or NCHW\n");
        printf("N: Batch size (number of images processed at once\n");
        printf("C: Number of channels (e.g., 3 for RGB images\n");
        printf("H: Height of the image\n");
        printf("W: Width of the image\n");
        return NULL;
    }

    int channel = 0;
    /*
    If req_channel is 0, stbi_load() will load the image with its original number of channels (as indicated by &channel).
    If req_channel is set to 1, 3, or 4, stbi_load() will force the image to be loaded with that many channels:
    1: Grayscale (single channel)
    3: RGB (three channels)
    4: RGBA (four channels, with an alpha channel for transparency)
    */
    unsigned char *image_data = stbi_load(image_path, img_width, img_height, &channel, req_channel);

    if (image_data == NULL)
    {
        printf("load image failed!\n");
        return NULL;
    }
    else
    {
        printf("Our image loaded with height=%d,width=%d,channels=%d\n", img_height, img_width, channel);
    }

    if (*img_width != req_width || *img_height != req_height) // does the image dimensions match the model's requirement?
    {
        printf("Our image loaded with height=%d,width=%d,channels=%d but we need height=%d,width=%d,channels=%d\n, so we are resizing the image.\n",
               img_height, img_width, channel, req_height, req_width, req_channel);
        unsigned char *image_resized = (unsigned char *)STBI_MALLOC(req_width * req_height * req_channel);
        if (!image_resized)
        {
            printf("malloc image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        if (stbir_resize_uint8(image_data, *img_width, *img_height, 0, image_resized, req_width, req_height, 0, channel) != 1)
        {
            printf("resize image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        STBI_FREE(image_data);
        image_data = image_resized;
    }

    return image_data;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage:%s model_path input_path [loop_count]\n Where input_path is the path to the target image we wish to classify.\n", argv[0]);
        return -1;
    }

    char *model_path = argv[1];
    char *input_path = argv[2];

    int loop_count = 1;
    if (argc > 3)
    {
        loop_count = atoi(argv[3]);
    }

    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;

    int img_width = 0;
    int img_height = 0;

    rknn_context ctx = 0;

    // Load RKNN Model
#if 1
    // Init rknn from model path
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
#else
    // Init rknn from model data
    size_t model_size;
    void *model_data = load_file(model_path, &model_size);
    if (model_data == NULL)
    {
        return -1;
    }
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
#endif
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    else
    {
        printf("Initialised Model Successfully\n");
    }

    // Get sdk and driver version
    rknn_sdk_version sdk_ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    /*
    io_num.n_input holds the number of input tensors the model expects.
    io_num.n_output holds the number of output tensors the model produces.
    */
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    // Get custom string
    rknn_custom_string custom_string;
    ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("custom string: %s\n", custom_string.string);

    unsigned char *input_data = NULL;
    rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
    rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;

    // Load image
    input_data = load_image(input_path, &input_attrs[0], &img_height, &img_width);

    if (!input_data)
    {
        return -1;
    }
    else
    {
        printf("Loaded input data from path : %s\n", input_path);
    }

    printf("Creating tensor input memory\n");
    // Create input tensor memory
    rknn_tensor_mem *input_mems[1];
    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    input_attrs[0].type = input_type;
    // default fmt is NHWC, npu only support NHWC in zero copy mode
    input_attrs[0].fmt = input_layout;

    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    // Copy input data to input tensor memory
    int width = input_attrs[0].dims[2];
    int stride = input_attrs[0].w_stride;

    if (width == stride)
    {
        printf("Copying input data too input tensor memory: width =%d, stride=%d\n", width, stride);
        memcpy(input_mems[0]->virt_addr, input_data, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
    }
    else
    {
        printf("Copying input data too input tensor memory after reformat\n");
        int height = input_attrs[0].dims[1];
        int channel = input_attrs[0].dims[3];
        // copy from src to dst with stride
        uint8_t *src_ptr = input_data;
        uint8_t *dst_ptr = (uint8_t *)input_mems[0]->virt_addr;
        // width-channel elements
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h)
        {
            memcpy(dst_ptr, src_ptr, src_wc_elems);
            src_ptr += src_wc_elems;
            dst_ptr += dst_wc_elems;
        }
    }

    // Create output tensor memory
    printf("Creating tensor output memory\n");
    rknn_tensor_mem *output_mems[io_num.n_output];
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
    }

    // Set input tensor memory
    printf("Setting tensor input memory\n");
    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret < 0)
    {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }

    // Set output tensor memory
    printf("Setting tensor output memory\n");
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        // set output memory and attribute
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0)
        {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    // Run
    printf("Begin perf ...\n");
    for (int i = 0; i < loop_count; ++i)
    {
        int64_t start_us = getCurrentTimeUs();
        /*
        rknn_run triggers the actual execution of the model on the hardware (such as the RKV1106 or other RKNPU-based devices).
        It processes the input data through the layers of the pre-trained neural network to compute the final output,
        such as object detection results, classifications, etc.
        The RKNN runtime handles all the tensor manipulations (such as matrix multiplications,
        convolutions, and activations) and produces the result in the output tensors that are accessible after the rknn_run() call.
        After run completes, The outputs of the model (such as detection results) will be stored in the output tensors, which have
        also been pre-allocated using rknn_set_output() or obtained by querying the output information after the model is initialized.
        */
        ret = rknn_run(ctx, NULL);
        int64_t elapse_us = getCurrentTimeUs() - start_us;
        if (ret < 0)
        {
            printf("rknn run error %d\n", ret);
            return -1;
        }
        printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
    }

    int model_width = 0;
    int model_height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        model_width = input_attrs[0].dims[2];
        model_height = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input format\n");
        model_width = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
    }
    // post process
    float scale_w = (float)model_width / img_width;
    float scale_h = (float)model_height / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    post_process((int8_t *)output_mems[0]->virt_addr, (int8_t *)output_mems[1]->virt_addr, (int8_t *)output_mems[2]->virt_addr, 640, 640,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    char text[256];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        printf("%s @ (%d %d %d %d) %f\n",
               det_result->name,
               det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
               det_result->prop);
    }

    // Destroy rknn memory
    printf("Releasing rknn memory\n");
    rknn_destroy_mem(ctx, input_mems[0]);
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        rknn_destroy_mem(ctx, output_mems[i]);
    }

    // destroy
    rknn_destroy(ctx);

    free(input_data);

    return 0;
}
