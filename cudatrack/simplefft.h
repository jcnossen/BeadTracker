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
	SFFT_BOTH T& imag() { return y; }
	SFFT_BOTH T& real() { return x; }
	SFFT_BOTH const T& imag() const { return y; }
	SFFT_BOTH const T& real() const { return x; }
	SFFT_BOTH complex() : x(0.0f), y(0.0f) {}
	SFFT_BOTH complex(T a, T b=0.0) : x(a), y(b) {}
	SFFT_BOTH complex conjugate() { return complex(x, -y); }

	SFFT_BOTH complex operator*(const T& b) const { return complex(x*b,y*b); }
	SFFT_BOTH complex& operator*=(const T& b) { x*=b; y*=b; return *this; }
	SFFT_BOTH complex operator*(const complex& b) const { return complex(x*b.x - y*b.y, x*b.y + y*b.x); }
	SFFT_BOTH complex operator-(const complex& b) const { return complex(x-b.x, y-b.y); }
	SFFT_BOTH complex operator+(const complex& b) const { return complex(x+b.x, y+b.y); }
	SFFT_BOTH complex& operator+=(const complex& b) { x+=b.x; y+=b.y; return *this; }

	complex(const std::complex<T>& a) : x(a.real()), y(a.imag()) {}
};

template<typename T>
SFFT_BOTH void swap(T& a, T& b) {
	T tmp(a); 
	a = b;
	b = tmp;
}


template<typename T, int sign>
SFFT_BOTH void fft(size_t N, complex<T> *zs) {
	const T pi = 3.14159265359;
    unsigned int j=0;
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
            T t = pi * sign * m / j;
            complex<T> w = complex<T>(cos(t),sin(t));
            for(unsigned int i = m; i<N; i+=2*j) {
                complex<T> zi = zs[i], t = w * zs[i + j];
				zs[i] = zi+t;
				zs[i+j] = zi-t;
            }
        }
}

template<typename T>
SFFT_BOTH void fft_forward(size_t N, complex<T> *zs) {
	fft<T, -1>(N,zs);
}
template<typename T>
SFFT_BOTH void fft_inverse(size_t N, complex<T> *zs) {
	fft<T, 1>(N,zs);
}

#undef SFFT_BOTH

};
