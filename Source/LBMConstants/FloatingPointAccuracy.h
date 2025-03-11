#pragma once

#include <stdint.h>
#include <string>

enum FloatingPointAccuracy {
	fp16 = 0,
	fp32 = 1,
};

int32_t get_FLoatingPointAccuracy_size_in_bytes(FloatingPointAccuracy floating_accuracy);
std::string get_FLoatingPointAccuracy_to_macro(FloatingPointAccuracy floating_accuracy);