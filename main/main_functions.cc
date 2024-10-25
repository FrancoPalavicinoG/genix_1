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

#include <string>
#include "driver/gpio.h"
#define LED_PIN GPIO_NUM_4 


namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 39 * 1024;
#else
constexpr int scratchBufSize = 300000;
#endif

constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;
}  

void setup() {

  gpio_reset_pin(LED_PIN); 
  gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

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

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE

void loop() {
  
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }

  gpio_set_level(LED_PIN, 1);

  /*   printf("Imagen capturada:\n");
  for (int i = 0; i < kNumCols * kNumRows * kNumChannels; i++) {
    input->data.int8[i] = ((uint8_t *)input->data.uint8)[i];  
    // printf("%d, ", input->data.int8[i]);
  }
  printf("\n"); */

  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int idx = 0;
  int idx2 = 0;
  int8_t max_confidence = output->data.uint8[idx];
  int8_t cur_confidence;
  float max_tmp = -10000.0;

  for(int i = 0; i < kCategoryCount; i++) {
      float tmp = output->data.f[i];
      cur_confidence = output->data.uint8[i];

      if(max_confidence < cur_confidence) {
          idx2 = idx;
          idx = i;
          max_confidence = cur_confidence;
      }
      if (tmp > max_tmp) {
          max_tmp = tmp;
      }
  }

  std::string detected = kCategoryLabels[idx];
  std::string detected2 = kCategoryLabels[idx2];

  printf("%s;%s. \n", detected.c_str(), detected2.c_str());
  // printf("detected2: %s\n", detected2.c_str());

  gpio_set_level(LED_PIN, 0);
  vTaskDelay(100); // to avoid watchdog trigger
}
#endif

