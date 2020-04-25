#include "DNNLayerConvolution.h"

#include <cstdlib>
#include <cstdio>

DNNLayerConvolution::DNNLayerConvolution(GPU *_gpu, int _numPics, int _x1, int _x2, int _numConvolutions, int _y1, int _y2, int _batchSize, float _initVal, float _stepSize)
	: DNNLayer(_gpu, _batchSize, _numPics * _x1 * _x2, _numPics * (_x1 - _y1 + 1) * (_x2 - _y2 + 1) * _numConvolutions, _numConvolutions * _y1 * _y2, _initVal, _stepSize)
{
	x1 = _x1;
	x2 = _x2;
	y1 = _y1;
	y2 = _y2;
	numPics = _numPics;
	numConvolutions = _numConvolutions;
}

DNNLayerConvolution::~DNNLayerConvolution()
{

}

void DNNLayerConvolution::Forward(CPUGPUMemory* input)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&numPics, (void*)&inputWidth, 
		(void*)&outputWidth, (void*)&numConvolutions, (void*)&x1, (void*)&x2, (void*)&y1, (void*)&y2, (void*)&t};
	gpu->Execute((char*)"convolution_forward", (char*)"pppiiiiiiiii\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerConvolution::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {deltaInput == NULL ? (void*)NULL : (void*)deltaInput->GetGPUMemory(), (void*)dparams->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
		(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&numPics, (void*)&inputWidth, 
		(void*)&outputWidth, (void*)&numConvolutions, (void*)&x1, (void*)&x2, (void*)&y1, (void*)&y2, (void*)&t};
	gpu->Execute((char*)"convolution_backward", (char*)"ppppppiiiiiiiii\0", pp, input->GetSize() / inputWidth);
}
