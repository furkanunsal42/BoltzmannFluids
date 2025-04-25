#pragma once

#include <stdio.h>

#include <cuda_runtime.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define HANDLE_ERROR(message) (HandleError(message, __FILE__, __LINE__))

#define HANDLE_NULL(a) {if (a == nullptr) {\
	fprintf(stderr, "Host memory failed in %s at line %d\n", __FILE__, __LINE__); \
	exit(EXIT_FAILURE); } }

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s (in %s at line %d)\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

static void HandleError(const char* message, const char* file, int line) {
	fprintf(stderr, "%s (in %s at line %d)\n", message, file, line);
	exit(EXIT_FAILURE);
}