#pragma once

#include "fftw3.h"
#include <complex>
#include <vector>

#ifdef TRK_USE_DOUBLE
	typedef double xcor_t;
#else
	typedef float xcor_t;
#endif

#ifdef TRK_USE_DOUBLE
	typedef fftw_plan fftw_plan_t;
#else
	typedef fftwf_plan fftw_plan_t;
#endif

typedef std::complex<xcor_t> complexc;

