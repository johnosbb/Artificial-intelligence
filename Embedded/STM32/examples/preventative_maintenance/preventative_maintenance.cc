#include <tensorflow/lite/core/interpreter_builder.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model_builder.h>
#include <tensorflow/lite/optional_debug_tools.h>

#include <cstdio>
#include <iostream>
#include <vector>

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

  // Allocate tensor buffers
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "Error allocating tensors\n");
    return 1;
  }

  // Print input tensor information
  int input_index = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input_index);
  printf("Input tensor info:\n");
  printf("  - Type: %d\n", input_tensor->type);  // Should be 1 (float32)
  printf("  - Dimensions: %d\n", input_tensor->dims->size);
  for (int i = 0; i < input_tensor->dims->size; i++) {
    printf("  - Dim[%d]: %d\n", i, input_tensor->dims->data[i]);
  }

  // Ensure the input tensor is of type float32
  if (input_tensor->type != kTfLiteFloat32) {
    fprintf(stderr, "Input tensor type mismatch. Expected float32.\n");
    return 1;
  }

  // Ensure the input tensor shape is [1, 4]
  if (input_tensor->dims->data[0] != 1 || input_tensor->dims->data[1] != 4) {
    fprintf(stderr, "Input tensor shape mismatch. Expected [1, 4].\n");
    return 1;
  }

  // Input data (raw values)
  float rpm = 1699.34f;
  float temperature = 30.99f;
  float vibration = 0.106f;
  float current = 2.92f;

  // Normalization constants (mean and std values provided)
  constexpr float tMean = 24.354f;
  constexpr float rpmMean = 1603.866f;
  constexpr float vMean = 0.120f;
  constexpr float cMean = 3.494f;
  constexpr float tStd = 4.987f;
  constexpr float rpmStd = 195.843f;
  constexpr float vStd = 0.020f;
  constexpr float cStd = 0.308f;

  // Standardize the input data
  float input_data[] = {
      (rpm - rpmMean) / rpmStd,      // Standardize RPM
      (temperature - tMean) / tStd,  // Standardize Temperature
      (vibration - vMean) / vStd,    // Standardize Vibration
      (current - cMean) / cStd       // Standardize Current
  };

  // Get pointer to input tensor data
  float* input = interpreter->typed_tensor<float>(input_index);
  if (input == nullptr) {
    fprintf(stderr, "Failed to get input tensor pointer.\n");
    return 1;
  }

  // Fill the input tensor with data
  for (int i = 0; i < 4; ++i) {
    input[i] = input_data[i];
  }

  // Check if input data was written properly
  printf("Input data written: ");
  for (int i = 0; i < 4; ++i) {
    printf("%f ", input[i]);
  }
  printf("\n");

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    fprintf(stderr, "Error invoking the model\n");
    return 1;
  }

  // Get output tensor info
  int output_index = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output_index);
  if (output_tensor == nullptr) {
    fprintf(stderr,
            "Failed to retrieve output tensor. Invalid index or allocation "
            "error.\n");
    return 1;
  }

  // Print output tensor info
  printf("Output tensor info:\n");
  printf("  - Type: %d\n", output_tensor->type);  // Should be float32
  printf("  - Dimensions: %d\n", output_tensor->dims->size);
  for (int i = 0; i < output_tensor->dims->size; i++) {
    printf("  - Dim[%d]: %d\n", i, output_tensor->dims->data[i]);
  }

  // Ensure output tensor is of type float32
  if (output_tensor->type != kTfLiteFloat32) {
    fprintf(stderr, "Output tensor type mismatch. Expected float32.\n");
    return 1;
  }

  // Get pointer to output tensor data
  float* output = interpreter->typed_tensor<float>(output_index);
  if (output == nullptr) {
    fprintf(stderr, "Failed to get output tensor pointer.\n");
    return 1;
  }

  // Print output result
  printf("Motor failure prediction: %.2f\n", output[0]);

  return 0;
}
