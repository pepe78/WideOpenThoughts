#include "DNNLayerRelu.h"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

DNNLayerRelu::DNNLayerRelu(GPU *_gpu, int _inputWidth, int _batchSize)
	: DNNLayer(_gpu, _batchSize, _inputWidth, _inputWidth, 0, 0, 0)
{

}

DNNLayerRelu::~DNNLayerRelu()
{

}

void DNNLayerRelu::Forward(CPUGPUMemory* input)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)&inputWidth, (void*)&t};
	gpu->Execute((char*)"relu_forward", (char*)"ppii\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerRelu::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	if (deltaInput != NULL)
	{
		int t = (input->GetSize() / inputWidth);
		void* pp[] = {(void*)deltaInput->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
			(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)&inputWidth, (void*)&t};
		gpu->Execute((char*)"relu_backward", (char*)"ppppii\0", pp, input->GetSize() / inputWidth);
	}
}
