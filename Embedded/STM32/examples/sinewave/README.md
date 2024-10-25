# Sinewave Inference Example

This program loads a TensorFlow Lite model, sets up an interpreter, takes a float input from the user, and uses it as the input for a model that predicts sin(x). It runs inference in a loop, printing both the model’s prediction and the actual sine value for comparison. This code demonstrates the basics of using TensorFlow Lite in C++ for real-time, model-based inference.

## Code Explanation

### Header Inclusions and Macro Definition

```
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

```

- Includes Standard Libraries: cmath for mathematical functions, cstdio for standard I/O functions, cstring for string operations, and memory for smart pointers.
- TensorFlow Lite Includes: The TensorFlow Lite libraries are included to access the core functionalities required for loading and interpreting models.
- Macro for Error Checking (TFLITE_SP_CHECK): This macro verifies a condition, and if it fails, it prints an error message with the file and line number, then exits the program. This is used throughout the code to handle critical failures safely.


### Function Initialization and Model Loading

```
int main(int argc, char* argv[]) {
  // Load the TensorFlow Lite model
  const char* filename =
      "sinewave_predictor_float.tflite";  // We use the sinewave predictor float
                                          // tflite model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_SP_CHECK(model != nullptr);

```

- Model Filename: Sets the filename for the .tflite model file (sinewave_predictor_float.tflite), assumed to be a TensorFlow Lite model for predicting the sine wave.
- Load Model into Memory: The model is loaded into memory using tflite::FlatBufferModel::BuildFromFile(). If loading fails, the macro TFLITE_SP_CHECK exits with an error.


### Interpreter Creation

```cpp
  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_SP_CHECK(interpreter != nullptr);

```

- Interpreter and Op Resolver: Initializes a BuiltinOpResolver, which links TensorFlow Lite’s built-in operations with the interpreter.
- Interpreter Builder: Uses tflite::InterpreterBuilder to create an interpreter from the loaded model. The interpreter will handle the model’s operations and manage input/output tensors.
- Check Interpreter Creation: Uses TFLITE_SP_CHECK to ensure the interpreter was created successfully.

###  Tensor Memory Allocation and Tensor Info Retrieval

```cpp
  // Allocate memory for tensors
  TFLITE_SP_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

  // Retrieve input and output tensor information
  int input_index = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input_index);
  int output_index = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output_index);

```

- Allocate Tensor Memory: Allocates memory for the model’s tensors, necessary before inference.
- Retrieve Tensor Indices: Retrieves indices for the input and output tensors using interpreter->inputs() and interpreter->outputs().
- Access Tensors: Gets pointers to the input and output tensors, allowing direct access to their data during inference.

### Display Input Tensor Information

```cpp
  // Display input tensor info (dimensions, type)
  printf("Input tensor info:\n");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    printf("  - Dim[%d]: %d\n", i, input_tensor->dims->data[i]);
  }

```

- Tensor Dimension Information: Outputs the dimensions of the input tensor, which helps verify that the model’s input shape matches the expected data input format.

### User Input for Inference Loop

```cpp
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

```

- Prompt and Input: Prompts the user to input a float value, ideally between 0 and 6.28 (approximate bounds for one cycle of a sine wave).
- Value Clamping: Ensures that x is within the range [0,6.28] to match the model’s expected input range.

### Setting Input Tensor, Running Inference, and Displaying Output

```cpp
    // Set the value in the model's input tensor (input tensor is typically float)
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

```

- Set Input Tensor Value: Sets the value of the input tensor to x, the user-provided float value.
- Run Model Inference: Calls interpreter->Invoke() to run the model and produce an output in the output tensor.
- Retrieve and Display Model Output: Retrieves the result from the output tensor, assumed to be the model’s prediction for sin(x).
- Compare with Actual Sine Value: Calculates the actual sine value of x using sin(x) from <cmath> and prints it alongside the model’s inferred result for easy comparison.

  
