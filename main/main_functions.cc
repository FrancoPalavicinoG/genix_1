/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 39 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<9> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }

  printf("Imagen capturada:\n");
  for (int i = 0; i < kNumCols * kNumRows * kNumChannels; i++) {
    input->data.int8[i] = ((uint8_t *)input->data.uint8)[i];  
    printf("%d, ", input->data.int8[i]);
  }
  printf("\n");

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t zero_score = output->data.uint8[zero_Index];
  float zero_score_f = (zero_score - output->params.zero_point) * output->params.scale;
  printf("Score zero: %f \n",zero_score_f);
  int8_t one_score = output->data.uint8[one_Index];
  float one_score_f = (one_score - output->params.zero_point) * output->params.scale;
  printf("Score one: %f \n",one_score_f);
  int8_t two_score = output->data.uint8[two_Index];
  float two_score_f = (two_score - output->params.zero_point) * output->params.scale;
  printf("Score two: %f \n",two_score_f);
  int8_t three_score = output->data.uint8[three_Index];
  float three_score_f = (three_score - output->params.zero_point) * output->params.scale;
  printf("Score three: %f \n",three_score_f);
  int8_t four_score = output->data.uint8[four_Index];
  float four_score_f = (four_score - output->params.zero_point) * output->params.scale;
  printf("Score four: %f \n",four_score_f);
  int8_t five_score = output->data.uint8[five_Index];
  float five_score_f = (five_score - output->params.zero_point) * output->params.scale;
  printf("Score five: %f \n",five_score_f);
  int8_t six_score = output->data.uint8[six_Index];
  float six_score_f = (six_score - output->params.zero_point) * output->params.scale;
  printf("Score six: %f \n",six_score_f);
  int8_t seven_score = output->data.uint8[seven_Index];
  float seven_score_f = (seven_score - output->params.zero_point) * output->params.scale;
  printf("Score seven:%f \n",seven_score_f);
  int8_t eight_score = output->data.uint8[eight_Index];
  float eight_score_f = (eight_score - output->params.zero_point) * output->params.scale;
  printf("Score eight: %f \n",eight_score_f);
  int8_t nine_score = output->data.uint8[nine_Index];
  float nine_score_f = (nine_score - output->params.zero_point) * output->params.scale;
  printf("Score nine: %f \n",nine_score_f);
  int8_t ten_score = output->data.uint8[ten_Index];
  float ten_score_f = (ten_score - output->params.zero_point) * output->params.scale;
  printf("Score ten: %f \n", ten_score_f);
  int8_t eleven_score = output->data.uint8[eleven_Index];
  float eleven_score_f = (eleven_score - output->params.zero_point) * output->params.scale;
  printf("Score eleven: %f \n", eleven_score_f);
  int8_t twelve_score = output->data.uint8[twelve_Index];
  float twelve_score_f = (twelve_score - output->params.zero_point) * output->params.scale;
  printf("Score twelve: %f \n", twelve_score_f);
  int8_t thirteen_score = output->data.uint8[thirteen_Index];
  float thirteen_score_f = (thirteen_score - output->params.zero_point) * output->params.scale;
  printf("Score thirteen: %f \n", thirteen_score_f);
  int8_t fourteen_score = output->data.uint8[fourteen_Index];
  float fourteen_score_f = (fourteen_score - output->params.zero_point) * output->params.scale;
  printf("Score fourteen: %f \n", fourteen_score_f);
  int8_t fifteen_score = output->data.uint8[fifteen_Index];
  float fifteen_score_f = (fifteen_score - output->params.zero_point) * output->params.scale;
  printf("Score fifteen: %f \n", fifteen_score_f);
  int8_t sixteen_score = output->data.uint8[sixteen_Index];
  float sixteen_score_f = (sixteen_score - output->params.zero_point) * output->params.scale;
  printf("Score sixteen: %f \n", sixteen_score_f);
  int8_t seventeen_score = output->data.uint8[seventeen_Index];
  float seventeen_score_f = (seventeen_score - output->params.zero_point) * output->params.scale;
  printf("Score seventeen: %f \n", seventeen_score_f);
  int8_t eighteen_score = output->data.uint8[eighteen_Index];
  float eighteen_score_f = (eighteen_score - output->params.zero_point) * output->params.scale;
  printf("Score eighteen: %f \n", eighteen_score_f);
  int8_t nineteen_score = output->data.uint8[nineteen_Index];
  float nineteen_score_f = (nineteen_score - output->params.zero_point) * output->params.scale;
  printf("Score nineteen: %f \n", nineteen_score_f);
  int8_t twenty_score = output->data.uint8[twenty_Index];
  float twenty_score_f = (twenty_score - output->params.zero_point) * output->params.scale;
  printf("Score twenty: %f \n", twenty_score_f);
  int8_t twentyOne_score = output->data.uint8[twentyOne_Index];
  float twentyOne_score_f = (twentyOne_score - output->params.zero_point) * output->params.scale;
  printf("Score twenty-one: %f \n", twentyOne_score_f);
  int8_t twentyTwo_score = output->data.uint8[twentyTwo_Index];
  float twentyTwo_score_f = (twentyTwo_score - output->params.zero_point) * output->params.scale;
  printf("Score twenty-two: %f \n", twentyTwo_score_f);
  int8_t twentyThree_score = output->data.uint8[twentyThree_Index];
  float twentyThree_score_f = (twentyThree_score - output->params.zero_point) * output->params.scale;
  printf("Score twenty-three: %f \n", twentyThree_score_f);
  // int8_t twentyFour_score = output->data.uint8[twentyFour_Index];
  // float twentyFour_score_f = (twentyFour_score - output->params.zero_point) * output->params.scale;
  // printf("Score twenty-four: %f \n", twentyFour_score_f);

  // Respond to detection
  // RespondToDetection(person_score_f, no_person_score_f);
  vTaskDelay(10); // to avoid watchdog trigger
}
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

void run_inference(void *ptr) {
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    
    input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;
    printf("%d, ", input->data.int8[i]);
    //printf("%d, ", input->data.int8[i]); // Revisamos que hay dentro del buffer	
  }

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  //printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  /* Reset times */
  total_time = 0;
  //softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);
  
  // Process the inference results.
  int8_t zero_score = output->data.uint8[zero_Index];
  float zero_score_f = (zero_score - output->params.zero_point) * output->params.scale;
  printf("Score zero: %f \n",zero_score_f);
  int8_t one_score = output->data.uint8[one_Index];
  float one_score_f = (one_score - output->params.zero_point) * output->params.scale;
  printf("Score one: %f \n",one_score_f);
  int8_t two_score = output->data.uint8[two_Index];
  float two_score_f = (two_score - output->params.zero_point) * output->params.scale;
  printf("Score two: %f \n",two_score_f);
  int8_t three_score = output->data.uint8[three_Index];
  float three_score_f = (three_score - output->params.zero_point) * output->params.scale;
  printf("Score three: %f \n",three_score_f);
  int8_t four_score = output->data.uint8[four_Index];
  float four_score_f = (four_score - output->params.zero_point) * output->params.scale;
  printf("Score four: %f \n",four_score_f);
  int8_t five_score = output->data.uint8[five_Index];
  float five_score_f = (five_score - output->params.zero_point) * output->params.scale;
  printf("Score five: %f \n",five_score_f);
  int8_t six_score = output->data.uint8[six_Index];
  float six_score_f = (six_score - output->params.zero_point) * output->params.scale;
  printf("Score six: %f \n",six_score_f);
  int8_t seven_score = output->data.uint8[seven_Index];
  float seven_score_f = (seven_score - output->params.zero_point) * output->params.scale;
  printf("Score seven:%f \n",seven_score_f);
  int8_t eight_score = output->data.uint8[eight_Index];
  float eight_score_f = (eight_score - output->params.zero_point) * output->params.scale;
  printf("Score eight: %f \n",eight_score_f);
  int8_t nine_score = output->data.uint8[nine_Index];
  float nine_score_f = (nine_score - output->params.zero_point) * output->params.scale;
  printf("Score nine: %f \n",nine_score_f);
  int8_t ten_score = output->data.uint8[ten_Index];
  float ten_score_f = (ten_score - output->params.zero_point) * output->params.scale;
  printf("Score ten: %f \n", ten_score_f);
  int8_t eleven_score = output->data.uint8[eleven_Index];
  float eleven_score_f = (eleven_score - output->params.zero_point) * output->params.scale;
  printf("Score eleven: %f \n", eleven_score_f);
  int8_t twelve_score = output->data.uint8[twelve_Index];
  float twelve_score_f = (twelve_score - output->params.zero_point) * output->params.scale;
  printf("Score twelve: %f \n", twelve_score_f);
  int8_t thirteen_score = output->data.uint8[thirteen_Index];
  float thirteen_score_f = (thirteen_score - output->params.zero_point) * output->params.scale;
  printf("Score thirteen: %f \n", thirteen_score_f);
  int8_t fourteen_score = output->data.uint8[fourteen_Index];
  float fourteen_score_f = (fourteen_score - output->params.zero_point) * output->params.scale;
  printf("Score fourteen: %f \n", fourteen_score_f);
  int8_t fifteen_score = output->data.uint8[fifteen_Index];
  float fifteen_score_f = (fifteen_score - output->params.zero_point) * output->params.scale;
  printf("Score fifteen: %f \n", fifteen_score_f);
  int8_t sixteen_score = output->data.uint8[sixteen_Index];
  float sixteen_score_f = (sixteen_score - output->params.zero_point) * output->params.scale;
  printf("Score sixteen: %f \n", sixteen_score_f);
  int8_t seventeen_score = output->data.uint8[seventeen_Index];
  float seventeen_score_f = (seventeen_score - output->params.zero_point) * output->params.scale;
  printf("Score seventeen: %f \n", seventeen_score_f);
  int8_t eighteen_score = output->data.uint8[eighteen_Index];
  float eighteen_score_f = (eighteen_score - output->params.zero_point) * output->params.scale;
  printf("Score eighteen: %f \n", eighteen_score_f);
  int8_t nineteen_score = output->data.uint8[nineteen_Index];
  float nineteen_score_f = (nineteen_score - output->params.zero_point) * output->params.scale;
  printf("Score nineteen: %f \n", nineteen_score_f);
  int8_t twenty_score = output->data.uint8[twenty_Index];
  float twenty_score_f = (twenty_score - output->params.zero_point) * output->params.scale;
  printf("Score twenty: %f \n", twenty_score_f);
  int8_t twentyOne_score = output->data.uint8[twentyOne_Index];
  float twentyOne_score_f = (twentyOne_score - output->params.zero_point) * output->params.scale;
  printf("Score twenty-one: %f \n", twentyOne_score_f);
  int8_t twentyTwo_score = output->data.uint8[twentyTwo_Index];
  float twentyTwo_score_f = (twentyTwo_score - output->params.zero_point) * output->params.scale;
  printf("Score twenty-two: %f \n", twentyTwo_score_f);
  int8_t twentyThree_score = output->data.uint8[twentyThree_Index];
  float twentyThree_score_f = (twentyThree_score - output->params.zero_point) * output->params.scale;
  printf("Score twenty-three: %f \n", twentyThree_score_f);
  // int8_t twentyFour_score = output->data.uint8[twentyFour_Index];
  // float twentyFour_score_f = (twentyFour_score - output->params.zero_point) * output->params.scale;
  // printf("Score twenty-four: %f \n", twentyFour_score_f);

  //float person_score_f =
  //    (person_score - output->params.zero_point) * output->params.scale;
  //float no_person_score_f =
  //    (no_person_score - output->params.zero_point) * output->params.scale;
  //RespondToDetection(person_score_f, no_person_score_f);
}
