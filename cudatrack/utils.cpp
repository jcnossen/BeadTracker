#include <cuda_runtime.h>
#include <cstdarg>
#include "utils.h"
#include <Windows.h>
#include <string>

void throwCudaError(cudaError_t err)
{
	std::string msg = SPrintf("CUDA error: %s", cudaGetErrorString(err));
	dbgout(msg);
	throw std::runtime_error(msg);
}


std::string SPrintf(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);

	va_end(ap);
	return buf;
}

void dbgout(std::string s) {
	OutputDebugString(s.c_str());
	printf(s.c_str());
}
