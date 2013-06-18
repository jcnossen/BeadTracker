#pragma once

#include "LsqQuadraticFit.h"

// Builds a fisher matrix for a localization ZLUT
class LUTFisherMatrix
{
public:
	LUTFisherMatrix(float* lut, int radialsteps, int planes)  {
		this->lut = lut;
		this->radialsteps = radialsteps;
		this->planes = planes;
		profile = 0;
		dzProfile = 0;
	}
	~LUTFisherMatrix() {
		delete[] profile;
		delete[] dzProfile;
	}

	void Compute(int w, int h, vector3f pos, float zlutMinRadius, float zlutMaxRadius) 
	{
		InterpolateProfile(pos.z);

		// Subpixel grid size
		const int SW=4;
		const int SH=4;
		const float xstep = 1.0f/SW;
		const float ystep = 1.0f/SH;

		double Izz, Ixx , Iyy, Ixy, Ixz, Iyz;

		for (int py=0;py<h;py++) {
			for (int px=0;px<w;px++) {
				float xOffset = px + 0.5f/SW;
				float yOffset = py + 0.5f/SH;

				// 4x4 grid over each pixel, to approximate the integral
				for (int sy=0;sy<SW;sy++) {
					for (int sx=0;sx<SW;sx++) {
						float x = xOffset + sx*xstep;
						float y = yOffset + sy*ystep;
					}
				}


			}
		}

		
	}


	// Compute profile
	void InterpolateProfile(float z)
	{
		int iz = (int)z;
		iz = std::max(0, std::min(planes-2, iz));
		float *prof0 = &lut[iz*radialsteps];
		float *prof1 = &lut[(iz+1)*radialsteps];

		for (int r=0;r<radialsteps;r++) {
			profile[r] = prof0[r] + (prof1[r] - prof0[r]) * (z-iz);
		}

		// Compute derivative
		const int NumZPlanes = 5;
		float zplanes[NumZPlanes];
		float prof[NumZPlanes];
		int minZ = std::max(iz-NumZPlanes/2, 0);
		int maxZ = std::min(iz+(NumZPlanes-NumZPlanes/2), planes);

		for(int r=0;r<radialsteps;r++) {
			for (int p=minZ;p<maxZ;p++) {
				zplanes[p]=p-minZ;
				prof[p]=LUT(p,r);
			}

			LsqSqQuadFit<float> qfit(NumZPlanes, zplanes, prof);
//			dzProfile[r] = prof1[r] - prof0[r]; // TODO: interpolate this stuff
			dzProfile[r] = qfit.computeDeriv(z-minZ);
		}
	}

	float LUT(int pl, int r) {
		return lut[pl*radialsteps+r];
	}

	int radialsteps, planes;
	float* lut;

	float* profile;
	float* dzProfile; // derivative of profile wrt Z plane

	vector3f matrix[3];
};
