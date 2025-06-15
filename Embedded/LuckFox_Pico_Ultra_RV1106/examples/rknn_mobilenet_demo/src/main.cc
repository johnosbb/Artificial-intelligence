// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
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

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#define DEBUG_DEMO

#define MAX_CLASSES 1008
#define MAX_LABEL_LENGTH 128
char *labels[MAX_CLASSES];

// Function to load labels from file
int load_labels(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "Error: Could not open label file %s\n", filename);
        return -1;
    }

    char line[MAX_LABEL_LENGTH];
    int i = 0;
    while (fgets(line, sizeof(line), fp) && i < MAX_CLASSES)
    {
        // Remove the WordNet ID and just keep the label
        char *label_start = strchr(line, ' '); // Find the first space
        if (label_start)
        {
            label_start++; // Move past the space
            // Remove trailing newline character if present
            line[strcspn(line, "\n")] = 0;
            labels[i] = strdup(label_start); // Duplicate the string
            if (!labels[i])
            {
                fprintf(stderr, "Error: Memory allocation failed for label %d\n", i);
                fclose(fp);
                return -1;
            }
            i++;
        }
    }
    fclose(fp);
    return i; // Return number of loaded labels
}

/*-------------------------------------------
                Functions
-------------------------------------------*/
static inline int64_t getCurrentTimeUs()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

static int rknn_GetTopN(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum)
{
    uint32_t i, j;
    uint32_t top_count = outputCount > topNum ? topNum : outputCount;

    for (i = 0; i < topNum; ++i)
    {
        pfMaxProb[i] = -FLT_MAX;
        pMaxClass[i] = -1;
    }

    for (j = 0; j < top_count; j++)
    {
        for (i = 0; i < outputCount; i++)
        {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
                (i == *(pMaxClass + 4)))
            {
                continue;
            }

            float prob = pfProb[i];
            if (prob > *(pfMaxProb + j))
            {
                *(pfMaxProb + j) = prob;
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

static int rknn_GetTopN_int8(int8_t *pProb, float scale, int zp, float *pfMaxProb, uint32_t *pMaxClass,
                             uint32_t outputCount, uint32_t topNum)
{
    uint32_t i, j;
    uint32_t top_count = outputCount > topNum ? topNum : outputCount;

    for (i = 0; i < topNum; ++i)
    {
        pfMaxProb[i] = -FLT_MAX;
        pMaxClass[i] = -1;
    }

    for (j = 0; j < top_count; j++)
    {
        for (i = 0; i < outputCount; i++)
        {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
                (i == *(pMaxClass + 4)))
            {
                continue;
            }

            float prob = (pProb[i] - zp) * scale;
            if (prob > *(pfMaxProb + j))
            {
                *(pfMaxProb + j) = prob;
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
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

static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr)
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
        break;
    case RKNN_TENSOR_NCHW:
        req_height = input_attr->dims[2];
        req_width = input_attr->dims[3];
        req_channel = input_attr->dims[1];
        break;
    default:
        printf("meet unsupported layout\n");
        return NULL;
    }

    int height = 0;
    int width = 0;
    int channel = 0;

    unsigned char *image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image failed!\n");
        return NULL;
    }

#ifdef DEBUG_DEMO
    printf("[DEBUG] Loaded image properties:\n");
    printf("[DEBUG] Original image dimensions: %dx%dx%d\n", width, height, channel);
    printf("[DEBUG] Required image dimensions (from model input attr): %dx%dx%d\n", req_width, req_height, req_channel);
#endif

    if (width != req_width || height != req_height)
    {
#ifdef DEBUG_DEMO
        printf("[DEBUG] Resizing image from %dx%d to %dx%d...\n", width, height, req_width, req_height);
#endif
        unsigned char *image_resized = (unsigned char *)STBI_MALLOC(req_width * req_height * req_channel);
        if (!image_resized)
        {
            printf("malloc image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        if (stbir_resize_uint8(image_data, width, height, 0, image_resized, req_width, req_height, 0, channel) != 1)
        {
            printf("resize image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        STBI_FREE(image_data);
        image_data = image_resized;
    }

#ifdef DEBUG_DEMO
    printf("[DEBUG] Image loaded and processed successfully.\n");
#endif

    return image_data;
}

/*-------------------------------------------
                Main Functions
-------------------------------------------*/
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage:%s model_path input_path [loop_count]\n", argv[0]);
        return -1;
    }
    int num_loaded_labels = load_labels("./model/RV1106/synset.txt");
    char *model_path = argv[1];
    char *input_path = argv[2];

    int loop_count = 1;
    if (argc > 3)
    {
        loop_count = atoi(argv[3]);
    }

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

#ifdef DEBUG_DEMO
    printf("[DEBUG] Model initialization status: %s\n", (ret == RKNN_SUCC) ? "SUCCESS" : "FAILED");
    if (ret != RKNN_SUCC)
    {
        printf("[DEBUG] rknn_init returned error code: %d\n", ret);
    }
#endif

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
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

#ifdef DEBUG_DEMO
    if (io_num.n_input == 0 || io_num.n_output == 0)
    {
        printf("[DEBUG] Model appears to have invalid input/output numbers. n_input: %d, n_output: %d\n", io_num.n_input, io_num.n_output);
    }
    else
    {
        printf("[DEBUG] Model input/output numbers seem valid.\n");
    }
#endif

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
#ifdef DEBUG_DEMO
        printf("[DEBUG] Input tensor %d attributes dumped.\n", i);
        // Basic validation for input tensor attributes
        if (input_attrs[i].n_dims == 0 || input_attrs[i].n_elems == 0 || input_attrs[i].size == 0)
        {
            printf("[DEBUG] WARNING: Input tensor %d appears to have invalid dimensions/size.\n", i);
        }
#endif
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
#ifdef DEBUG_DEMO
        printf("[DEBUG] Output tensor %d attributes dumped.\n", i);
        // Basic validation for output tensor attributes
        if (output_attrs[i].n_dims == 0 || output_attrs[i].n_elems == 0 || output_attrs[i].size == 0)
        {
            printf("[DEBUG] WARNING: Output tensor %d appears to have invalid dimensions/size.\n", i);
        }
#endif
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
    input_data = load_image(input_path, &input_attrs[0]);

    if (!input_data)
    {
        printf("Failed to load or process input image: %s\n", input_path);
        return -1;
    }

    // Create input tensor memory
    rknn_tensor_mem *input_mems[1];
    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    input_attrs[0].type = input_type;
    // default fmt is NHWC, npu only support NHWC in zero copy mode
    input_attrs[0].fmt = input_layout;

    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (!input_mems[0])
    {
        printf("Failed to create input tensor memory!\n");
        free(input_data);
        return -1;
    }

    // Copy input data to input tensor memory
    int width = input_attrs[0].dims[2];
    int stride = input_attrs[0].w_stride;

    if (width == stride)
    {
        memcpy(input_mems[0]->virt_addr, input_data, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
#ifdef DEBUG_DEMO
        printf("[DEBUG] Input data copied directly (width == stride).\n");
#endif
    }
    else
    {
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
#ifdef DEBUG_DEMO
        printf("[DEBUG] Input data copied with stride adjustment (width != stride).\n");
#endif
    }

    // Create output tensor memory
    rknn_tensor_mem *output_mems[io_num.n_output];
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].n_elems * sizeof(float));
        if (!output_mems[i])
        {
            printf("Failed to create output tensor memory for output %d!\n", i);
            // Clean up previously allocated output_mems and input_mems
            for (uint32_t j = 0; j < i; ++j)
            {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            rknn_destroy_mem(ctx, input_mems[0]);
            free(input_data);
            return -1;
        }
    }

    // Set input tensor memory
    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret < 0)
    {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        // Clean up
        rknn_destroy_mem(ctx, input_mems[0]);
        for (uint32_t i = 0; i < io_num.n_output; ++i)
        {
            rknn_destroy_mem(ctx, output_mems[i]);
        }
        free(input_data);
        return -1;
    }
#ifdef DEBUG_DEMO
    printf("[DEBUG] Input tensor memory set successfully.\n");
#endif

    // Set output tensor memory
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        // set output memory and attribute
        output_attrs[i].type = RKNN_TENSOR_FLOAT32; // Ensure output is float32 for GetTopN
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0)
        {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            // Clean up
            rknn_destroy_mem(ctx, input_mems[0]);
            for (uint32_t j = 0; j <= i; ++j)
            { // Destroy up to current i
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            free(input_data);
            return -1;
        }
#ifdef DEBUG_DEMO
        printf("[DEBUG] Output tensor %d memory set successfully.\n", i);
#endif
    }

    // Run
    printf("Begin perf ...\n");
    for (int i = 0; i < loop_count; ++i)
    {
        int64_t start_us = getCurrentTimeUs();
        ret = rknn_run(ctx, NULL);
        int64_t elapse_us = getCurrentTimeUs() - start_us;
        if (ret < 0)
        {
            printf("rknn run error %d\n", ret);
            // Clean up
            rknn_destroy_mem(ctx, input_mems[0]);
            for (uint32_t j = 0; j < io_num.n_output; ++j)
            {
                rknn_destroy_mem(ctx, output_mems[j]);
            }
            free(input_data);
            return -1;
        }
        printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
#ifdef DEBUG_DEMO
        printf("[DEBUG] RKNN run for loop %d completed.\n", i);
#endif
    }

    // Get top 5
    uint32_t topNum = 5;
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        uint32_t MaxClass[topNum];
        float fMaxProb[topNum];

        uint32_t sz = output_attrs[i].n_elems;
        int top_count = sz > topNum ? topNum : sz;

        float *buffer = (float *)output_mems[i]->virt_addr;

#ifdef DEBUG_DEMO
        if (buffer == NULL)
        {
            printf("[DEBUG] ERROR: Output buffer for tensor %d is NULL!\n", i);
        }
        else
        {
            // Check a few values in the output buffer for sanity (not exhaustive)
            if (sz > 0)
            {
                printf("[DEBUG] First few values of output tensor %d: %f, %f, %f...\n", i, buffer[0], (sz > 1 ? buffer[1] : 0.0f), (sz > 2 ? buffer[2] : 0.0f));
            }
        }
#endif

        rknn_GetTopN(buffer, fMaxProb, MaxClass, sz, topNum);

        for (int j = 0; j < top_count; j++)
        {
            printf("%8.6f - %d", fMaxProb[j], MaxClass[j]);
            if (MaxClass[j] >= 0 && MaxClass[j] < num_loaded_labels && labels[MaxClass[j]] != NULL)
            {
                printf(" - %s\n", labels[MaxClass[j]]);
            }
            else
            {
                printf("\n");
            }
        }

        // Free the allocated label strings
        for (int i = 0; i < num_loaded_labels; ++i)
        {
            if (labels[i])
            {
                free(labels[i]);
            }
        }
    }

    // Destroy rknn memory
    rknn_destroy_mem(ctx, input_mems[0]);
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        rknn_destroy_mem(ctx, output_mems[i]);
    }
#ifdef DEBUG_DEMO
    printf("[DEBUG] RKNN input and output memories destroyed.\n");
#endif

    // destroy
    rknn_destroy(ctx);
#ifdef DEBUG_DEMO
    printf("[DEBUG] RKNN context destroyed.\n");
#endif

    free(input_data);
#ifdef DEBUG_DEMO
    printf("[DEBUG] Input image data freed.\n");
#endif

    return 0;
}