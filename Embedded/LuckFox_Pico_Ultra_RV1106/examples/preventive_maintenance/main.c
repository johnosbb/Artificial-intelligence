#include <stdio.h>
#include <stdlib.h>
#include "rknn_api.h"

#define MODEL_PATH "models/preventive_forecast.rknn"

// Replace these with your actual quantization parameters
#define INPUT_SCALE 0.021073076874017715
#define INPUT_ZERO_POINT 3

void handle_error(int ret);
void *read_model(const char *model_path, uint32_t *model_size);

int main()
{
    rknn_context ctx;
    uint32_t model_size = 0;
    void *model_data = read_model(MODEL_PATH, &model_size);
    if (!model_data)
    {
        fprintf(stderr, "Failed to read model file.\n");
        return -1;
    }

    printf("Initializing RKNN...\n");
    rknn_init_extend extend = {0};
    int ret = rknn_init(&ctx, model_data, model_size, 0, &extend);
    free(model_data);
    if (ret != 0)
    {
        handle_error(ret);
        return -1;
    }

    // Prepare quantized input data
    // Example: original input data = {1700, 25.5, 0.135, 3.75}
    float original_input_data[4] = {1700, 25.5, 0.135, 3.75};

    // Normalize and quantize the input
    int8_t quantized_input_data[4];
    for (int i = 0; i < 4; ++i)
    {
        quantized_input_data[i] = (int8_t)round((original_input_data[i] / INPUT_SCALE) + INPUT_ZERO_POINT);
    }

    printf("Setting input data...\n");
    rknn_input input = {0};
    input.index = 0;
    input.buf = quantized_input_data; // Use quantized data
    input.size = sizeof(quantized_input_data);
    input.type = RKNN_TENSOR_UINT8; // Change to UINT8 for quantized input
    input.fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(ctx, 1, &input);
    if (ret != 0)
    {
        handle_error(ret);
        rknn_destroy(ctx);
        return -1;
    }

    printf("Running inference...\n");
    ret = rknn_run(ctx, NULL);
    if (ret != 0)
    {
        handle_error(ret);
        rknn_destroy(ctx);
        return -1;
    }

    printf("Getting output...\n");
    rknn_output output = {0};
    output.want_float = 1; // If you want the output as float
    ret = rknn_outputs_get(ctx, 1, &output, NULL);
    if (ret != 0)
    {
        handle_error(ret);
        rknn_destroy(ctx);
        return -1;
    }

    printf("Model output: %f\n", ((float *)output.buf)[0]);

    rknn_outputs_release(ctx, 1, &output);
    rknn_destroy(ctx);
    return 0;
}

void handle_error(int ret)
{
    fprintf(stderr, "RKNN Error: %d\n", ret);
}

void *read_model(const char *model_path, uint32_t *model_size)
{
    FILE *file = fopen(model_path, "rb");
    if (!file)
    {
        perror("fopen");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    *model_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    void *model_data = malloc(*model_size);
    if (!model_data)
    {
        perror("malloc");
        fclose(file);
        return NULL;
    }

    fread(model_data, 1, *model_size, file);
    fclose(file);
    return model_data;
}
