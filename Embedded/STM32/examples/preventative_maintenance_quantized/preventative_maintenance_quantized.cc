#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"

// Macro for error checking
#define TFLITE_PM_CHECK(x)                                   \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: preventative_maintenance <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_PM_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_PM_CHECK(interpreter != nullptr);

  // Allocate tensors
  TFLITE_PM_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Retrieve input tensor information
  int input_index = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input_index);

  // Check input tensor type (either float32 or quantized int8)
  if (input_tensor->type == kTfLiteFloat32) {
    printf("Model expects float32 input\n");
  } else if (input_tensor->type == kTfLiteInt8) {
    printf("Model expects quantized int8 input\n");
  } else {
    fprintf(stderr, "Unsupported input tensor type: %d\n", input_tensor->type);
    return 1;
  }

  // Ensure input tensor is of the expected type (int8)
  if (input_tensor->type != kTfLiteInt8) {
    fprintf(stderr,
            "Error: Model does not use quantized int8 input. Exiting...\n");
    return 1;
  }

  // Print input tensor information
  printf("Input tensor info:\n");
  printf("  - Type: %d\n", input_tensor->type);
  printf("  - Dimensions: %d\n", input_tensor->dims->size);
  for (int i = 0; i < input_tensor->dims->size; i++) {
    printf("    - Dim[%d]: %d\n", i, input_tensor->dims->data[i]);
  }

  // Prepare raw input data (before normalization)
  float rpm = 1699.34f;
  float temperature = 30.99f;
  float vibration = 0.106f;
  float current = 2.92f;

  // Standardize the input data (using predefined means and standard deviations)
  constexpr float rpmMean = 1603.866f;
  constexpr float tMean = 24.354f;
  constexpr float vMean = 0.120f;
  constexpr float cMean = 3.494f;
  constexpr float rpmStd = 195.843f;
  constexpr float tStd = 4.987f;
  constexpr float vStd = 0.020f;
  constexpr float cStd = 0.308f;

  // Normalize input data
  printf("Normalize input data\n");
  float standardized_input[4];
  standardized_input[0] = (rpm - rpmMean) / rpmStd;
  standardized_input[1] = (temperature - tMean) / tStd;
  standardized_input[2] = (vibration - vMean) / vStd;
  standardized_input[3] = (current - cMean) / cStd;

  // Retrieve quantization parameters
  printf("Retrieve quantization parameters\n");
  float scale = input_tensor->params.scale;
  int zero_point = input_tensor->params.zero_point;

  // Quantize the normalized input data
  printf("Quantize the normalized input data\n");
  int8_t quantized_input[4];
  for (int i = 0; i < 4; i++) {
    quantized_input[i] = static_cast<int8_t>(
        std::round(standardized_input[i] / scale) + zero_point);
  }

  // Fill the input tensor with quantized data
  printf("Fill the input tensor with quantized data\n");
  std::memcpy(interpreter->typed_tensor<int8_t>(input_index), quantized_input,
              sizeof(quantized_input));

#ifdef SHOW_INTERPRETER_STATE
  // After allocation of tensors, before running inference
  TFLITE_PM_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-inference Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());
#endif
  // Run inference
  printf("Run inference\n");
  TFLITE_PM_CHECK(interpreter->Invoke() == kTfLiteOk);

#ifdef SHOW_INTERPRETER_STATE
  // After running inference
  TFLITE_PM_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("=== Post-inference Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());
#endif

  // Retrieve and check output tensor
  printf("Retrieve and check output tensor\n");
  int output_index = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output_index);

  // Print output tensor information for debugging
  printf("Output tensor info:\n");
  printf("  - Type: %d\n", output_tensor->type);
  printf("  - Dimensions: %d\n", output_tensor->dims->size);
  for (int i = 0; i < output_tensor->dims->size; i++) {
    printf("    - Dim[%d]: %d\n", i, output_tensor->dims->data[i]);
  }

  // Handle quantized output
  if (output_tensor->type == kTfLiteInt8) {
    // Output is quantized
    printf(
        "Retrieving scale and output zero point for quantized (int8) data\n");
    float output_scale = output_tensor->params.scale;
    int output_zero_point = output_tensor->params.zero_point;

    printf(
        "Handling quantized output for quantized (int8)... scale = %f "
        "output_zero_point = %d\n",
        output_scale, output_zero_point);

    // Try accessing the raw output tensor data using output_tensor function
    int8_t* quantized_output = output_tensor->data.int8;

    // Ensure quantized_output is not NULL before proceeding
    if (quantized_output == nullptr) {
      fprintf(
          stderr,
          "Error: Failed to retrieve the quantized output tensor pointer.\n");
      return 1;
    }

    printf("Converting quantized output from quantized (int8) to float\n");
    float output_value =
        (quantized_output[0] - output_zero_point) * output_scale;
    printf("Motor failure prediction (quantized): %.2f\n", output_value);
  } else {
    fprintf(stderr, "Unsupported output tensor type: %d\n",
            output_tensor->type);
    return 1;
  }

  return 0;
}
