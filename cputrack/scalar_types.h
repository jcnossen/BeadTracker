#pragma once

#include <complex>
#include <vector>

#ifdef TRK_USE_DOUBLE
	typedef double xcor_t;
#else
	typedef float xcor_t;
#endif

typedef std::complex<xcor_t> complexc;

