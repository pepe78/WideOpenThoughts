#include "DNNLayerSigmoid.h"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

DNNLayerSigmoid::DNNLayerSigmoid(GPU *_gpu, int _inputWidth, float _min, float _max, int _batchSize)
	: DNNLayer(_gpu, _batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{
	o_min = _min;
	o_max = _max;
}

DNNLayerSigmoid::~DNNLayerSigmoid()
{

}

void DNNLayerSigmoid::Forward(CPUGPUMemory* input)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)&inputWidth, (void*)&t, (void*)&o_min, (void*)&o_max};
	gpu->Execute((char*)"sigmoid_forward", (char*)"ppiiff\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerSigmoid::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int t = (input->GetSize() / inputWidth);
		void* pp[] = {(void*)deltaInput->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
			(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)&inputWidth, (void*)&t, (void*)&o_min, (void*)&o_max};
		gpu->Execute((char*)"sigmoid_backward", (char*)"ppppiiff\0", pp, input->GetSize() / inputWidth);
	}
}
