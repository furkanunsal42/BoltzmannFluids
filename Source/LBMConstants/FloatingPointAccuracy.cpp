#include "FloatingPointAccuracy.h"

#include <iostream>

int32_t get_FloatingPointAccuracy_size_in_bytes(FloatingPointAccuracy floating_accuracy)
{
	switch (floating_accuracy) {
	case FloatingPointAccuracy::fp16:
		return 2;
	case FloatingPointAccuracy::fp32:
		return 4;
	}

	std::cout << "[LBM Error] get_FloatingPointAccuracy_size_in_bytes() but given floating_accuracy is not supported" << std::endl;
}

std::string get_FloatingPointAccuracy_to_macro(FloatingPointAccuracy floating_accuracy)
{
	switch (floating_accuracy) {
	case FloatingPointAccuracy::fp16:
		return "fp16";
	case FloatingPointAccuracy::fp32:
		return "fp32";
	}

	std::cout << "[LBM Error] get_FloatingPointAccuracy_to_macro() but given floating_accuracy is not supported" << std::endl;

}
