
#pragma once

#ifndef LSQFIT_FUNC 
	#define LSQFIT_FUNC
#endif

template<typename T>
class LsqSqQuadFit
{
public:

	uint numPts;
	const T *X;
	const T *Y;

	T a,b,c;
	T s40, s30, s20, s10, s21, s11, s01;

	LSQFIT_FUNC LsqSqQuadFit(uint numPts, const T* xval, const T* yval) : numPts(numPts), X(xval), Y(yval)
	{
		computeSums();
        //notation sjk to mean the sum of x_i^j*y_i^k. 
    /*    s40 = getSx4(); //sum of x^4
        s30 = getSx3(); //sum of x^3
        s20 = getSx2(); //sum of x^2
        s10 = getSx();  //sum of x
        

        s21 = getSx2y(); //sum of x^2*y
        s11 = getSxy();  //sum of x*y
        s01 = getSy();   //sum of y
		*/

		T s00 = numPts;  //sum of x^0 * y^0  ie 1 * number of entries

		        //a = Da/D
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
    
	LSQFIT_FUNC T compute(T pos)
	{
		return a*pos*pos + b*pos + c;
	}

	LSQFIT_FUNC T computeDeriv(T pos)
	{
		return 2*a*pos + b;
	}

	LSQFIT_FUNC T maxPos()
	{
		return -b/(2*a);
	}
   
private:

    LSQFIT_FUNC T computeSums() // get sum of x
    {
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

		s10 = Sx; s20 = Sx2; s30 = Sx3; s40 = Sx4;
		s01 = Sy; s11 = Sxy; s21 = Sx2y;
        return Sx;
    }

public:
	static LSQFIT_FUNC T max_(T a, T b) { return a>b ? a : b; }
	static LSQFIT_FUNC T min_(T a, T b) { return a<b ? a : b; }

	static LSQFIT_FUNC void ComputeCoefficients(

	static LSQFIT_FUNC T ComputeMax(T* data, int len)
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
			T interpMax = qfit.maxPos();

			if (fabs(qfit.a)<1e-9f)
				return (T)iMax;
			else
				return (T)iMax + interpMax;
		}
	}
};


// Computes the interpolated maximum position
template<typename T, int numPts=7>
class ComputeMaxInterp {
public:



};
