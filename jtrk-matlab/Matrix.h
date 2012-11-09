
#pragma once

#ifndef NO_COMPLEX
#include <complex>
using std::complex;
#endif

template<typename T> class Matrix;
template<typename T> class MatrixFFTHelper;


/*
Simple 2D matrix class
*/
template<typename T>
class Matrix
{
public:
	Matrix() { d=0;w=h=0; }
	Matrix(int w,int h) { d=0;this->w=this->h=0; init(w,h); }
	template<typename Tsrc>
	Matrix(int w,int h, Tsrc* data){
		this->w=w; this->h=h;
		d=new T[w*h];
		for(int i=0;i<w*h;i++)
			d[i]=data[i];
	}
	Matrix(const Matrix& o) {
		d=0; w=h=0;
		init(o);
	}
	
	Matrix& operator=(const Matrix& m) {
		if (d) delete[] d;
		init(m);
		return *this;
	}
	template<typename A>
	void init(const Matrix<A>& m) {
		d=new T[m.w*m.h];
		w=m.w; h=m.h;
		for(int i=0;i<w*h;i++)
			d[i]=m.d[i];
	}
	~Matrix() {	
		delete[] d; 
		d=0;
	}
	void init(int w,int h) {
		if (this->w*this->h != w*h) {
			if (d) delete[] d;
			this->w=w;
			this->h=h;
			d=new T[w*h];
		}
		clear();
	}
	T& elem(int x,int y) { return d[w*y+x]; }
	const T& elem(int x,int y) const { return d[w*y+x]; }
	void clear(T v=T()) {
		if (d) {
			for(int i=0;i<w*h;i++) 
				d[i]=v; 
		}
	}
	int memsize() { return sizeof(T)*w*h; }

	Matrix operator*(const Matrix& m) {
		Matrix r(m.w,  h);
		assert(w==m.h);

		for (int x=0;x<m.w;x++) {
			for (int y=0;y<h;y++) {
				T accum=T();
				for (int j=0;j<m.h;j++)
					accum += m.elem(x, j) * elem(j, y);
				r.elem(x,y) = accum;
			}
		}
		return r;
	}
	Matrix operator*(T v) const {
		Matrix r(w,h);
		for(int i=0;i<w*h;i++)
			r.d[i]=d[i]*v;
		return r;
	}
	Matrix& operator*=(T v) {
		*this = *this*v;
		return *this;
	}
	template<typename T>
	Matrix operator+(const Matrix<T>& m) const {
		assert(m.w==w&&m.h==h);
		Matrix r(w,h);
		for (int i=0;i<w*h;i++)
			r.d[i]=d[i]+m.d[i];
		return r;
	}
	Matrix operator-(const Matrix& m) const {
		assert(m.w==w&&m.h==h);
		Matrix r(w,h);
		for (int i=0;i<w*h;i++)
			r.d[i]=d[i]-m.d[i];
		return r;
	}
	Matrix scalarmul(const Matrix& m) const {
		Matrix r(w,h);
		for (int i=0;i<w*h;i++)
			r.d[i]=d[i]*m.d[i];
		return r;
	}
	void normalize() {
		T min, max;
		min=max=d[0];
		for(int i=0;i<w*h;i++) {
			T v = d[i];
			if (v<min) min=v;
			if (v>max) max=v;
		}
		if (fabs(min-max)<0.00000001f) max=min+0.00000001f;
		T m=1.0f/(max-min);
		for(int i=0;i<w*h;i++)
			d[i]=(d[i]-min)*m;
	}
	void normalizeSum(float wantedSum) {
		T sum=0.0;
		for (int a=0;a<w*h;a++)
			sum+=d[a];
		if (sum != 0) {
			T f = wantedSum / sum;
			for (int a=0;a<w*h;a++)
				d[a]*=f;
		}
	}
	Matrix transpose() const {
		Matrix r(h,w);
		for (int y=0;y<h;y++)
			for(int x=0;x<w;x++)
				r.elem(y,x) = elem(x,y);
		return r;
	}
	template<typename RT>
	Matrix<RT> real() const {
		Matrix<RT> r(w, h);
		for(int y=0;y<h;y++)  {
			for (int x=0;x<w;x++)
				r.elem(x,y) = elem(x,y).real();
		}
		return r;
	}
	template<typename RT>
	Matrix<RT> imag() const {
		Matrix<RT> r(w, h);
		for(int y=0;y<h;y++)  {
			for (int x=0;x<w;x++)
				r.elem(x,y) = elem(x,y).imag();
		}
		return r;
	}
	template<typename T>
	Matrix& operator+=(const Matrix<T>& b) {
		for(int y=0;y<h;y++)  {
			for (int x=0;x<w;x++)
				elem(x,y)+=b.elem(x,y);
		}
		return *this;
	}
	template<typename T>
	Matrix& operator-=(const Matrix<T>& b) {
		for(int y=0;y<h;y++)  {
			for (int x=0;x<w;x++)
				elem(x,y)-=b.elem(x,y);
		}
		return *this;
	}
	Matrix sq() const {
		Matrix r(w,h);
		for(int y=0;y<h;y++)  {
			for (int x=0;x<w;x++)
				r.elem(x,y)=elem(x,y)*elem(x,y);
		}
		return r;
	}
	Matrix normalized() const {
		Matrix r(*this);
		r.normalize();
		return r;
	}

	T* d;
	int w,h;
};


typedef Matrix<float> Matrixf;
typedef Matrix<double> Matrixd;
#ifndef NO_COMPLEX
typedef complex<float> complexf;
typedef Matrix<complexf> Matrixcf;
typedef complex<double> complexd;
typedef Matrix<complexd> Matrixcd;


template<typename T>
Matrix<T> abs(const Matrix<complex<T> > &m) {
	Matrix<T> r(m.w, m.h);
	for(int y=0;y<m.h;y++)  {
		for (int x=0;x<m.w;x++) {
			complex<T> c = m.elem(x,y);
			r.elem(x,y) = sqrtf(c.real()*c.real() + c.imag()*c.imag());
		}
	}
	return r;
}





template<typename T>
Matrix<T> angle(const Matrix<complex<T> > &m) {
	Matrix<T> r(m.w, m.h);
	for(int y=0;y<m.h;y++)  {
		for (int x=0;x<m.w;x++) {
			complex<T> c = m.elem(x,y);
			r.elem(x,y) = atan2(c.imag(), c.real());
		}
	}
	return r;
}

#endif

