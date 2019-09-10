#include "DNNLayerMatrix.h"

#include <cstdlib>
#include <cstdio>

DNNLayerMatrix::DNNLayerMatrix(GPU *_gpu, int _inputWidth, int _outputWidth, int _batchSize, float _initVal, float _stepSize)
	: DNNLayer(_gpu, _batchSize, _inputWidth, _outputWidth, _outputWidth * (_inputWidth + 1), _initVal, _stepSize)
{
}

DNNLayerMatrix::~DNNLayerMatrix()
{

}

void DNNLayerMatrix::Forward(CPUGPUMemory* input)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&inputWidth, (void*)&outputWidth, (void*)&t};
	gpu->Execute((char*)"matrix_forward", (char*)"pppiii\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerMatrix::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {deltaInput == NULL ? (void*)NULL : (void*)deltaInput->GetGPUMemory(), (void*)dparams->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
		(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&inputWidth, (void*)&outputWidth, (void*)&t};
	gpu->Execute((char*)"matrix_backward", (char*)"ppppppiii\0", pp, input->GetSize() / inputWidth);
}
