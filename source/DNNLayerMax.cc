#include "DNNLayerMax.h"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

DNNLayerMax::DNNLayerMax(GPU *_gpu, int _numPics, int _x1, int _x2, int _d1, int _d2, int _batchSize)
	: DNNLayer(_gpu, _batchSize, _numPics * _x1 * _x2, _numPics * (_x1 / _d1) * (_x2 / _d2), 0, 0, 0)
{
	numPics = _numPics;
	x1 = _x1;
	x2 = _x2;
	d1 = _d1;
	d2 = _d2;
}

DNNLayerMax::~DNNLayerMax()
{

}

void DNNLayerMax::Forward(CPUGPUMemory* input)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)&numPics, (void*)&x1, (void*)&x2, (void*)&d1, (void*)&d2, (void*)&t};
	gpu->Execute((char*)"max_forward", (char*)"ppiiiiii\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerMax::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int t = (input->GetSize() / inputWidth);
		void* pp[] = {(void*)deltaInput->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
			(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)&numPics, (void*)&x1, (void*)&x2, (void*)&d1, (void*)&d2, (void*)&t};
		gpu->Execute((char*)"max_backward", (char*)"ppppiiiiii\0", pp, input->GetSize() / inputWidth);
	}
}
