/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
int inference_count=0;
int input_length;

constexpr int kTensorArenaSize = 128*1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
//  static tflite::AllOpsResolver resolver;
  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddFullyConnected();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

    // Obtain pointer to the model's input tensor.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || 
      (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != 6) ||
      (model_input->dims->data[2] != 6) ||
      (model_input->dims->data[3] != 1) ||
      (model_input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter,"Bad input tensor parameters in model");
    return;
  }  
  input_length = model_input->bytes/sizeof(float);
  model_output = interpreter->output(0);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  float x_input[50];

  for (int i=inference_count;i<inference_count+input_length;i++){
    float pos = static_cast<float>(i*kXrange)/static_cast<float>(kInferencesPerCycle);
    float wave = sin(pos+cos(pos))+1;
    x_input[i-inference_count] = wave;
  }

  for (int j=0; j<input_length; j++){
    model_input->data.f[j] = x_input[j];
//    TF_LITE_REPORT_ERROR(error_reporter, "%f", model_input->data.f[j]);
  }

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

//  TF_LITE_REPORT_ERROR(error_reporter, "%f", model_output->data.f[0]);
  int t = model_output->data.f[0] * 100;
  TF_LITE_REPORT_ERROR(error_reporter, "%d", t);
  
  if (inference_count == (kInferencesPerCycle - input_length))
    inference_count = 0;
  else
    inference_count += 1;
}
