#include "DNNLayerPreprocess.h"
#include "RandUtils.h"

#define NOKERNELS
#include "kernels.cl"

#include <cstdlib>
#include <cstdio>

DNNLayerPreprocess::DNNLayerPreprocess(GPU *_gpu, int _x1, int _x2, int _batchSize, float _minAngle, float _maxAngle, float _minStretch, float _maxStretch, float _minNoise, float _maxNoise, int _flipHor, int _flipVer, int _x1SamplePoints, int _x2SamplePoints)
	: DNNLayer(_gpu, _batchSize, _x1 * _x2, _x1 * _x2, 0, 0, 0)
{
	x1 = _x1;
	x2 = _x2;
	dom = new CPUGPUMemory(gpu, true, _batchSize * _x1 * _x2, 0);
	
	minAngle = _minAngle;
	maxAngle = _maxAngle;
	minStretch = _minStretch;
	maxStretch = _maxStretch;
	minNoise = _minNoise;
	maxNoise = _maxNoise;
	flipHor = _flipHor;
	flipVer = _flipVer;
	x1SamplePoints = _x1SamplePoints;
	x2SamplePoints = _x2SamplePoints;
	
	if (x1 * x2 > PMAXX1X2)
	{
		fprintf(stderr, "Project needs to be recompiled with larger field for preprocess layer\n");
		exit(-1);
	}
}

DNNLayerPreprocess::~DNNLayerPreprocess()
{
	delete dom;
}

void DNNLayerPreprocess::Forward(CPUGPUMemory* input)
{
	if (trainRun)
	{
		float *p = (float*)dom->GetCPUMemory();
		for (int i = 0; i < dom->GetSize(); i++)
		{
			p[i] = minNoise + getRand() * (maxNoise - minNoise);
		}
		dom->CopyCPUtoGPU();
	}
	int threadsPerBlock = 256;
	int numBlocks = ((input->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;

	float angle = minAngle + getRand() * (maxAngle - minAngle);
	float sx = (minStretch + getRand() * (maxStretch - minStretch)) * (flipHor == 0 ? 1 : (getRand() < 0.5f ? -1 : 1));
	float sy = (minStretch + getRand() * (maxStretch - minStretch)) * (flipVer == 0 ? 1 : (getRand() < 0.5f ? -1 : 1));
	int bs = (input->GetSize() / inputWidth);
	int tr = trainRun ? 1 : 0;

	void* pp[] = {(void*)output->GetGPUMemory(), (void*)dom->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)&inputWidth, (void*)&outputWidth, (void*)&x1, (void*)&x2, (void*)&bs,
		(void*)&angle, (void*)&sx, (void*)&sy, (void*)&x1SamplePoints, (void*)&x2SamplePoints, (void*)&tr};
	gpu->Execute((char*)"preprocess_forward", (char*)"pppiiiiifffiii\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerPreprocess::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
}
