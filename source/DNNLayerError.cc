#include "DNNLayerError.h"
#include "GPU.h"

#include <cstdlib>

DNNLayerError::DNNLayerError(GPU *_gpu, int _inputWidth, int _batchSize, bool _square)
{
	gpu = _gpu;
	square = _square;
	inputWidth = _inputWidth;
	batchSize = _batchSize;
	deltaInput = new CPUGPUMemory(gpu, true, inputWidth * batchSize, 0);
	error = new CPUGPUMemory(gpu, true, inputWidth * batchSize, 0);
}

DNNLayerError::~DNNLayerError()
{
	delete deltaInput;
	delete error;
}

double DNNLayerError::ComputeError(CPUGPUMemory* output, CPUGPUMemory *expectedOutput)
{
	int threadsPerBlock = 256;
	int numBlocks = ((output->GetSize() / inputWidth) + threadsPerBlock - 1) / threadsPerBlock;
	if (square)
	{
		int t = (output->GetSize() / inputWidth);
		void* pp[] = {(void*)error->GetGPUMemory(), (void*)deltaInput->GetGPUMemory(), (void*)expectedOutput->GetGPUMemory(), (void*)output->GetGPUMemory(), (void*)&inputWidth, (void*)&t};
		gpu->Execute((char*)"error_square_kernel", (char*)"ppppii\0", pp, (output->GetSize() / inputWidth));
		gpu->WaitForGPUToFinish();
	}
	else
	{
		int t = (output->GetSize() / inputWidth);
		void* pp[] = {(void*)error->GetGPUMemory(), (void*)deltaInput->GetGPUMemory(), (void*)expectedOutput->GetGPUMemory(), (void*)output->GetGPUMemory(), (void*)&inputWidth, (void*)&t};
		gpu->Execute((char*)"error_log_kernel", (char*)"ppppii\0", pp, (output->GetSize() / inputWidth));
		gpu->WaitForGPUToFinish();
	}

	error->CopyGPUtoCPU();
	double ret = 0.0;
	float *m = (float*)error->GetCPUMemory();
	for (int i = 0; i < expectedOutput->GetSize(); i++)
	{
		ret += m[i];
	}

	return ret;
}

CPUGPUMemory* DNNLayerError::GetDeltaInput()
{
	return deltaInput;
}
