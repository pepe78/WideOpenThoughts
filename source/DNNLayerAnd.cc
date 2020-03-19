#include "DNNLayerAnd.h"

#include <cstdlib>
#include <cstdio>

DNNLayerAnd::DNNLayerAnd(GPU *_gpu, int _inputWidth, int _outputWidth, int _batchSize, float _initVal, float _stepSize)
	: DNNLayer(_gpu, _batchSize, _inputWidth, _outputWidth, _outputWidth * (_inputWidth + 1), _initVal, _stepSize)
{
}

DNNLayerAnd::~DNNLayerAnd()
{

}

void DNNLayerAnd::Forward(CPUGPUMemory* input)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&inputWidth, (void*)&outputWidth, (void*)&t};
	gpu->Execute((char*)"and_forward", (char*)"pppiii\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerAnd::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {deltaInput == NULL ? (void*)NULL : (void*)deltaInput->GetGPUMemory(), (void*)dparams->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
		(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&inputWidth, (void*)&outputWidth, (void*)&t};
	gpu->Execute((char*)"and_backward", (char*)"ppppppiii\0", pp, input->GetSize() / inputWidth);
}
