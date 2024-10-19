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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_MODEL_SETTINGS_H_

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 24;
constexpr int zero_Index = 0;
constexpr int one_Index = 1;
constexpr int two_Index = 2;
constexpr int three_Index = 3;
constexpr int four_Index = 4;
constexpr int five_Index = 5;
constexpr int six_Index = 6;
constexpr int seven_Index = 7;
constexpr int eight_Index = 8;
constexpr int nine_Index = 9;
constexpr int ten_Index = 10;
constexpr int eleven_Index = 11;
constexpr int twelve_Index = 12;
constexpr int thirteen_Index = 13;
constexpr int fourteen_Index = 14;
constexpr int fifteen_Index = 15;
constexpr int sixteen_Index = 16;
constexpr int seventeen_Index = 17;
constexpr int eighteen_Index = 18;
constexpr int nineteen_Index = 19;
constexpr int twenty_Index = 20;
constexpr int twentyOne_Index = 21;
constexpr int twentyTwo_Index = 22;
constexpr int twentyThree_Index = 23;
// constexpr int twentyFour_Index = 24;

extern const char* kCategoryLabels[kCategoryCount];

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_MODEL_SETTINGS_H_
