
#pragma once

#ifndef LSQFIT_FUNC 
	#define LSQFIT_FUNC
#endif

template<typename T>
class LsqSqQuadFit
{
public:
	T a,b,c;
	float xoffset;

	struct Coeff {
		T s40, s30, s20, s10, s21, s11, s01, s00;

        LSQFIT_FUNC void abc(float& a, float& b, float& c) {			
			a = (s21*(s20 * s00 - s10 * s10) - s11*(s30 * s00 - s10 * s20) + s01*(s30 * s10 - s20 * s20))
                /
				(s40*(s20 * s00 - s10 * s10) - s30*(s30 * s00 - s10 * s20) + s20*(s30 * s10 - s20 * s20));

			//b = Db/D
			b = (s40*(s11 * s00 - s01 * s10) - s30*(s21 * s00 - s01 * s20) + s20*(s21 * s10 - s11 * s20))
					/
					(s40 * (s20 * s00 - s10 * s10) - s30 * (s30 * s00 - s10 * s20) + s20 * (s30 * s10 - s20 * s20));

			c = (s40*(s20 * s01 - s10 * s11) - s30*(s30 * s01 - s10 * s21) + s20*(s30 * s11 - s20 * s21))
					/
					(s40 * (s20 * s00 - s10 * s10) - s30 * (s30 * s00 - s10 * s20) + s20 * (s30 * s10 - s20 * s20));
		}
	};

	LSQFIT_FUNC LsqSqQuadFit(uint numPts, const T* xval, const T* yval)
	{
		calculate(numPts, xval, yval);
		xoffset =0;
	}

	LSQFIT_FUNC LsqSqQuadFit()
	{
		a=b=c=0;
		xoffset =0;
	}

	LSQFIT_FUNC void calculate(uint numPts, const T* X, const T* Y)
	{
		Coeff co = computeSums(X, Y, numPts);
		co.abc(a,b,c);
	}
    
	LSQFIT_FUNC T compute(T pos)
	{
		pos -= xoffset;
		return a*pos*pos + b*pos + c;
	}

	LSQFIT_FUNC T computeDeriv(T pos)
	{
		pos -= xoffset;
		return 2*a*pos + b;
	}

	LSQFIT_FUNC T maxPos()
	{
		return -b/(2*a);
	}
   
	template<int numPts>
	LSQFIT_FUNC void fromArray(T* data, int len, float pos)
	{
		int iPos = (int)pos;
		T xs[numPts]; 
		int startPos = max_(iPos-numPts/2, 0);
		int endPos = min_(iPos+(numPts-numPts/2), len);
		int numpoints = endPos - startPos;
		
		for(int i=startPos;i<endPos;i++)
			xs[i-startPos] = i-iPos;

		Calculate(numpoints, xs, &data[startPos]);
		xoffset = iPos;
	}

private:

    LSQFIT_FUNC Coeff computeSums(const T* X, const T* Y, uint numPts) // get sum of x
    {
        //notation sjk to mean the sum of x_i^j*y_i^k. 
    /*    s40 = getSx4(); //sum of x^4
        s30 = getSx3(); //sum of x^3
        s20 = getSx2(); //sum of x^2
        s10 = getSx();  //sum of x
        

        s21 = getSx2y(); //sum of x^2*y
        s11 = getSxy();  //sum of x*y
        s01 = getSy();   //sum of y
		*/

		T Sx = 0, Sy = 0;
		T Sx2 = 0, Sx3 = 0;
		T Sxy = 0, Sx4=0, Sx2y=0;
        for (uint i=0;i<numPts;i++)
        {
			T x = X[i];
			T y = Y[i];
			Sx += x;
            Sy += y;
			T sq = x*x;
			Sx2 += x*x;
			Sx3 += sq*x;
			Sx4 += sq*sq;
			Sxy += x*y;
			Sx2y += sq*y;
        }

		Coeff co;
		co.s10 = Sx; co.s20 = Sx2; co.s30 = Sx3; co.s40 = Sx4;
		co.s01 = Sy; co.s11 = Sxy; co.s21 = Sx2y;
		co.s00 = numPts;
        return co;
    }

};

// Computes the interpolated maximum position
template<typename T, int numPts=3>
class ComputeMaxInterp {
public:
	static LSQFIT_FUNC T max_(T a, T b) { return a>b ? a : b; }
	static LSQFIT_FUNC T min_(T a, T b) { return a<b ? a : b; }

	static LSQFIT_FUNC T Compute(T* data, int len)
	{
		int iMax=0;
		T vMax=data[0];
		for (int k=1;k<len;k++) {
			if (data[k]>vMax) {
				vMax = data[k];
				iMax = k;
			}
		}
		T xs[numPts];
		int startPos = max_(iMax-numPts/2, 0);
		int endPos = min_(iMax+(numPts-numPts/2), len);
		int numpoints = endPos - startPos;

		if (numpoints<3)
			return iMax;
		else {
			for(int i=startPos;i<endPos;i++)
				xs[i-startPos] = i-iMax;

			LsqSqQuadFit<T> qfit(numpoints, xs, &data[startPos]);
			//printf("iMax: %d. qfit: data[%d]=%f\n", iMax, startPos, data[startPos]);
			//for (int k=0;k<numpoints;k++) {
		//		printf("data[%d]=%f\n", startPos+k, data[startPos]);
			//}

			if (fabs(qfit.a)<1e-9f)
				return (T)iMax;
			else {
				T interpMax = qfit.maxPos();
				return (T)iMax + interpMax;
			}
		}
	}


};
