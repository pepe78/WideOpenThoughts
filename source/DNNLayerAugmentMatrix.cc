#include "DNNLayerAugmentMatrix.h"

#include <cstdlib>
#include <cstdio>


DNNLayerAugmentMatrix::DNNLayerAugmentMatrix(GPU *_gpu, int _numPics, int _x1, int _x2, int _numConvolutions, int _batchSize, float _initVal, float _stepSize)
	: DNNLayer(_gpu, _batchSize, _numPics * _x1 * _x2, _numPics * _x1 * _x2 * _numConvolutions, _numConvolutions * _x1 * _x2, _initVal, _stepSize)
{
	x1 = _x1;
	x2 = _x2;
	if(x1 != x2)
	{
		fprintf(stderr, "supporting only squares for augment matrix layer now\n");
		exit(-1);
	}
	numPics = _numPics;
	numConvolutions = _numConvolutions;

	//set diagonal to 1, with rest random numbers so it starts with close to identy projection
	float* pars = (float*)params->GetCPUMemory();
	for(int i=0;i<numConvolutions;i++)
	{
		for(int j=0;j<x1;j++)
		{
			pars[i * x1 * x2 + j * x2 +j] = 1.0f;
		}
	}
	params->CopyCPUtoGPU();
}

DNNLayerAugmentMatrix::~DNNLayerAugmentMatrix()
{

}

void DNNLayerAugmentMatrix::Forward(CPUGPUMemory* input)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&numPics, 
		(void*)&inputWidth, (void*)&outputWidth, (void*)&numConvolutions, (void*)&x1, (void*)&x2, (void*)&t};
	gpu->Execute((char*)"augmentmatrix_forward", (char*)"pppiiiiiii\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerAugmentMatrix::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {deltaInput == NULL ? (void*)NULL : (void*)deltaInput->GetGPUMemory(), (void*)dparams->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
		(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&numPics, 
		(void*)&inputWidth, (void*)&outputWidth, (void*)&numConvolutions, (void*)&x1, (void*)&x2, (void*)&t};
	gpu->Execute((char*)"augmentmatrix_backward", (char*)"ppppppiiiiiii\0", pp, input->GetSize() / inputWidth);
}
