
#include "std_incl.h"
#include "utils.h"

#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>

#include "../cudatrack/Array2D.h"

#include <thrust/functional.h>

#include "../cudatrack/tracker.h"

using gpuArray::Array2D;
using gpuArray::reducer_buffer;

static DWORD startTime = 0;
void BeginMeasure() { startTime = GetTickCount(); }
DWORD EndMeasure(const std::string& msg) {
	DWORD endTime = GetTickCount();
	DWORD dt = endTime-startTime;
	dbgprintf("%s: %d ms\n", msg.c_str(), dt);
	return dt;
}

vector2f ComputeCOM(pixel_t* data, uint w,uint h)
{
	float sum=0.0f;
	float momentX=0.0f;
	float momentY=0.0f;

	for (uint y=0;y<h;y++)
		for(uint x=0;x<w;x++)
		{
			pixel_t v = data[y*w+x];
			sum += v;
			momentX += x*v;
			momentY += y*v;
		}
	vector2f com;
	com.x = momentX / sum;
	com.y = momentY / sum;
	return com;
}




std::string getPath(const char *file)
{
	std::string s = file;
	int pos = s.length()-1;
	while (pos>0 && s[pos]!='\\' && s[pos]!= '/' )
		pos--;
	
	return s.substr(0, pos);
}

int main(int argc, char *argv[])
{
//	testLinearArray();

	std::string path = getPath(argv[0]);

	Tracker tracker(150,150);
	tracker.loadTestImage(5, 5, 1);
	
	Array2D<float> tmp(10,2);
	float *tmpdata=new float[tmp.w*tmp.h];
	float sumCPU=0.0f;
	for (int y=0;y<tmp.h;y++){
		for (int x=0;x<tmp.w;x++) {
			tmpdata[y*tmp.w+x]=x;
			sumCPU += x;
		}
	}
	tmp.set(tmpdata, sizeof(float)*tmp.w);

	float* checkmem=new float[tmp.w*tmp.h];
	tmp.copyToHost(checkmem);
	assert (memcmp(checkmem, tmpdata, sizeof(float)*tmp.w*tmp.h)==0);
	delete[] checkmem;

	delete[] tmpdata;
	reducer_buffer<float> rbuf(tmp.w,tmp.h);
	float sumGPU = tmp.sum(rbuf);
	dbgout(SPrintf("SumCPU: %f, SUMGPU: %f\n", sumCPU, sumGPU));
	
	return 0;
}
