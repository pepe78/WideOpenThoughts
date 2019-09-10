#include "DNNLayerDropout.h"
#include "RandUtils.h"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

DNNLayerDropout::DNNLayerDropout(GPU *_gpu, int _inputWidth, int _batchSize, float _perc)
	: DNNLayer(_gpu, _batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{
	dom = new CPUGPUMemory(gpu, true, _batchSize * _inputWidth, 0);
	perc = _perc;
}

DNNLayerDropout::~DNNLayerDropout()
{
	delete dom;
}

void DNNLayerDropout::Forward(CPUGPUMemory* input)
{
	if (trainRun)
	{
		float *p = (float*)dom->GetCPUMemory();
		for (int i = 0; i < dom->GetSize(); i++)
		{
			p[i] = getRand();
		}
		dom->CopyCPUtoGPU();
	}
	else
	{
		dom->Reset();
	}

	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)dom->GetGPUMemory(), (void*)&inputWidth, (void*)&perc, (void*)&t};
	gpu->Execute((char*)"dropout_forward", (char*)"pppifi\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerDropout::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int t = (input->GetSize() / inputWidth);
		void* pp[] = {(void*)deltaInput->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
			(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)dom->GetGPUMemory(), (void*)&inputWidth, (void*)&perc, (void*)&t};
		gpu->Execute((char*)"dropout_backward", (char*)"pppppifi\0", pp, input->GetSize() / inputWidth);
	}
}
