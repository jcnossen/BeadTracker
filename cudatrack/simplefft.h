// Power-of-two nonrecursive FFT
#pragma once

#include <complex>

namespace sfft {
#ifndef SFFT_BOTH
#define SFFT_BOTH __device__ __host__
#endif
	
	
template<typename T>
struct complex
{
	T x, y;
	CUBOTH T& imag() { return y; }
	CUBOTH T& real() { return x; }
	CUBOTH const T& imag() const { return y; }
	CUBOTH const T& real() const { return x; }
	CUBOTH complex() : x(0.0f), y(0.0f) {}
	CUBOTH complex(T a, T b=0.0) : x(a), y(b) {}
	CUBOTH complex conjugate() { return complex(x, -y); }

	CUBOTH complex operator*(const T& b) const { return complex(x*b,y*b); }
	CUBOTH complex& operator*=(const T& b) { x*=b; y*=b; return *this; }
	CUBOTH complex operator*(const complex& b) const { return complex(x*b.x - y*b.y, x*b.y + y*b.x); }
	CUBOTH complex operator-(const complex& b) const { return complex(x-b.x, y-b.y); }
	CUBOTH complex operator+(const complex& b) const { return complex(x+b.x, y+b.y); }
	CUBOTH complex& operator+=(const complex& b) { x+=b.x; y+=b.y; return *this; }

	complex(const std::complex<T>& a) : x(a.real()), y(a.imag()) {}
};

template<typename T>
CUBOTH void swap(T& a, T& b) {
	T tmp(a); 
	a = b;
	b = tmp;
}


template<typename T, int sign>
SFFT_BOTH void fft(size_t N, std::complex<T> *zs, complex<T>* twiddles) {
    unsigned int j=0;
	if (sign < 0) 
		twiddles += N; // forward FFT twiddles are located after inverse twiddles
    // Warning about signed vs unsigned comparison
    for(unsigned int i=0; i<N-1; ++i) {
        if (i < j) 
			swap(zs[i], zs[j]);
        int m=N/2;
        j^=m;
        while ((j & m) == 0) { m/=2; j^=m; }
    }
    for(unsigned int j=1; j<N; j*=2)
        for(unsigned int m=0; m<j; ++m) {
            //T t = pi * sign * m / j;
			//dbgprintf("fac: m=%d. j=%d. k=%d\n", m, j, N*m/(j*2));
            //complex<T> wcmp = complex<T>(cos(t),sin(t));
			complex<T> w = twiddles[N*m/(j*2)];
            for(unsigned int i = m; i<N; i+=2*j) {
                complex<T> zi = zs[i], t = w * zs[i + j];
				zs[i] = zi+t;
				zs[i+j] = zi-t;
            }
        }
}


template<typename T>
std::vector< complex<T> > fill_twiddles(int N) {
	const T pi = 3.14159265359;
	std::vector< complex<T> > twiddles(N*2);
	for(int i=0;i<N;i++) {
		T t = pi * 1 * i * 2 / N;
		twiddles[i] = complex<T>(cos(t),sin(t));
	}
	for(int i=0;i<N;i++) {
		T t = pi * -1 * i * 2 / N;
		twiddles[i+N] = complex<T>(cos(t),sin(t));
	}
	return twiddles;
}


template<typename T>
SFFT_BOTH void fft_forward(size_t N, complex<T> *zs, complex<T> *twiddles) {
	fft<T, -1>(N,zs,twiddles);
}
template<typename T>
SFFT_BOTH void fft_inverse(size_t N, complex<T> *zs, complex<T> *twiddles) {
	fft<T, 1>(N,zs,twiddles);
}

#undef SFFT_BOTH

};
