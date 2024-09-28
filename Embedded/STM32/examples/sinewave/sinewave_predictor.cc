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
#define TFLITE_SP_CHECK(x)                                   \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  // Load the TensorFlow Lite model
  const char* filename =
      "sinewave_predictor_float.tflite";  // We use the sinewave predictor float
                                          // tflite model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_SP_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_SP_CHECK(interpreter != nullptr);

  // Allocate memory for tensors
  TFLITE_SP_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Retrieve input and output tensor information
  int input_index = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input_index);
  int output_index = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output_index);

  // Display input tensor info (dimensions, type)
  printf("Input tensor info:\n");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    printf("  - Dim[%d]: %d\n", i, input_tensor->dims->data[i]);
  }

  printf("Please input a float number between 0 and 6.28:\n");
  float x;

  // Continuously read user input, perform inference, and display results
  while (true) {
    // Scan for user input from terminal
    scanf("%f", &x);  // Capture user input (expects a float)

    // Ensure input value stays within range [0, 6.28]
    if (x < 0) x = 0;
    if (x > 6.28) x = 6.28;

    printf("Your input value: %.2f\n", x);

    // Set the value in the model's input tensor (input tensor is typically
    // float)
    input_tensor->data.f[0] = x;

    // Run inference
    TFLITE_SP_CHECK(interpreter->Invoke() == kTfLiteOk);

    // Retrieve output from the model's output tensor
    float y = output_tensor->data.f[0];

    // Print inferred and actual sine value
    printf("Inferred Sin(%.2f) = %.2f\n", x, y);
    printf("Actual Sin(%.2f) = %.2f\n", x, sin(x));
  }

  return 0;
}
