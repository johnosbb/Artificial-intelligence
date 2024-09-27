# Preventative Maintenance Inference Example

This repository contains a C++ program that uses TensorFlow Lite to perform inference on a model that predicts motor failure based on several input parameters: RPM, Temperature (°C), Vibration (g), and Current (A). The model is expected to be pre-trained and saved in the TensorFlow Lite format (.tflite).

![image](https://github.com/user-attachments/assets/5b403e23-b8b5-4081-bd02-a604f8e10fc3)


This program serves as a basic example of how to use TensorFlow Lite for inference in C++. By understanding the concepts involved, such as tensor allocation, data normalization, and model invocation, you can extend this code for more complex applications and integrations.

## Overview

![image](https://github.com/user-attachments/assets/85c85036-a7e3-4d74-b174-401653316786)

The code demonstrates the following key concepts:

- Loading a TensorFlow Lite model.
- Allocating tensors for input and output data.
- Standardizing input data to match the training conditions of the model.
- Running inference and retrieving predictions.

## Code Explanation

### Includes and Definitions

```cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include <tensorflow/lite/core/interpreter_builder.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model_builder.h>
#include <tensorflow/lite/optional_debug_tools.h>

#define TFLITE_PM_CHECK(x)                                   \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }
```

- Includes: The necessary TensorFlow Lite headers are included to manage the model and interpreter.
- Macro Definition: A macro TFLITE_PM_CHECK is defined for error checking. It simplifies error handling by exiting the program if a condition fails.

## Main Function

```cpp
int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: preventative_maintenance <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];


```

- Argument Check: The program expects a single command-line argument: the path to the TensorFlow Lite model. If not provided, it will display usage instructions.

## Model Loading

```cpp
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_PM_CHECK(model != nullptr);

```

- Model Loading: The model is loaded from the specified file using FlatBufferModel::BuildFromFile. Error checking ensures that the model is loaded successfully.

## Building the Interpreter

```cpp
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_PM_CHECK(interpreter != nullptr);
```

- Interpreter Creation: An interpreter is created using InterpreterBuilder. This handles all the necessary setup for the model to be used for inference.

## Allocating Tensors

```cpp
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "Error allocating tensors\n");
    return 1;
  }

```

- Tensor Allocation: This step allocates memory for the input and output tensors that will be used during inference. Proper allocation is crucial for efficient model execution.

## Input Tensor Information

```cpp

  int input_index = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input_index);
  // Print input tensor information...

```

- Input Tensor: The program retrieves information about the input tensor, including its type and dimensions, ensuring it is as expected.
- Input Data: The program currently uses hard-coded input values for the motor parameters. If you want to modify the inputs, you can change the values in the source code before recompiling. The relevant section is where the raw input values are defined:

## Standardizing Input Data

```cpp

  // Input data (raw values)
  float rpm = 1699.34f;
  float temperature = 30.99f;
  float vibration = 0.106f;
  float current = 2.92f;

  // Normalization constants
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
      (rpm - rpmMean) / rpmStd,           // Standardize RPM
      (temperature - tMean) / tStd,       // Standardize Temperature
      (vibration - vMean) / vStd,         // Standardize Vibration
      (current - cMean) / cStd            // Standardize Current
  };

```

- Input Preparation: The raw input values are defined, followed by normalization constants (means and standard deviations) that were used during model training.
- Standardization: Each input feature is standardized by subtracting its mean and dividing by its standard deviation, making the data consistent with the model's training conditions.

## Running Inference

```cpp
 float* input = interpreter->typed_tensor<float>(input_index);
 // Fill the input tensor with data...

 if (interpreter->Invoke() != kTfLiteOk) {
   fprintf(stderr, "Error invoking the model\n");
   return 1;
 }
```

- Filling the Input Tensor: The standardized input data is written into the model's input tensor.
- Model Invocation: The model is invoked to perform inference. If the invocation fails, an error message is displayed.


![image](https://github.com/user-attachments/assets/fe17a110-b734-4384-9b08-08da2f75e762)

## Output Tensor Information

```cpp
  int output_index = interpreter->outputs()[0];
  TfLiteTensor* output_tensor = interpreter->tensor(output_index);
  // Print output tensor information...

  float* output = interpreter->typed_tensor<float>(output_index);
  // Print prediction...

```

- Output Handling: After invoking the model, the program retrieves information about the output tensor, including its type and dimensions.
- Prediction Result: The result of the inference (motor failure prediction) is printed.

## Building the Project

To build the project, ensure you have CMake and TensorFlow Lite installed. Run the following

```cmake
cmake -DCMAKE_TOOLCHAIN_FILE=stm32_toolchain.cmake -DTFLITE_HOST_TOOLS_DIR=/usr/local/bin ../tensorflow/tensorflow/lite/examples/preventative_maintenance
```

Then compile the project using: make

## Running the Program

After building the project, you can run the program with the TensorFlow Lite model as an argument:

```bash
./preventative_maintenance <path_to_your_model.tflite>

```

For example

```bash
./preventative_maintenance ./models/preventive_forecast_float32.tflite

```
