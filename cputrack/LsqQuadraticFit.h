
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
};
