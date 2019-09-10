#include "DNNLayer.h"
#include "GPU.h"

#include <cstdlib>
#include <cstdio>
#include <vector>

DNNLayer::DNNLayer(GPU *_gpu, int _batchSize, int _inputWidth, int _outputWidth, int _numParams, float _initVal, float _stepSize)
{
	gpu = _gpu;
	inputWidth = _inputWidth;
	outputWidth = _outputWidth;
	numParams = _numParams;

	stepSize = _stepSize;

	deltaInput = new CPUGPUMemory(gpu, true, inputWidth * _batchSize, 0);
	output = new CPUGPUMemory(gpu, true, outputWidth * _batchSize, 0);

	if (numParams == 0)
	{
		params = NULL;
		dparams = NULL;
	}
	else
	{
		params = new CPUGPUMemory(gpu, true, numParams, _initVal);
		dparams = new CPUGPUMemory(gpu, true, numParams, 0);
	}
}

DNNLayer::~DNNLayer()
{
	if (deltaInput != NULL)
	{
		delete deltaInput;
	}
	if (params != NULL)
	{
		delete params;
	}
	if (dparams != NULL)
	{
		delete dparams;
	}
	if (output != NULL)
	{
		delete output;
	}
}

void DNNLayer::Forward(CPUGPUMemory* input)
{
	fprintf(stderr, "forward not implemented!\n");
	exit(-1);
}

void DNNLayer::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	fprintf(stderr, "backward not implemented!\n");
	exit(-1);
}

void DNNLayer::ResetDeltaInput()
{
	if (deltaInput != NULL)
	{
		deltaInput->Reset();
	}
}

int DNNLayer::GetInputWidth()
{
	return inputWidth;
}

int DNNLayer::GetOutputWidth()
{
	return outputWidth;
}

CPUGPUMemory* DNNLayer::GetOutput()
{
	return output;
}

CPUGPUMemory* DNNLayer::GetDeltaInput()
{
	return deltaInput;
}

CPUGPUMemory* DNNLayer::GetParams()
{
	return params;
}

void DNNLayer::MakeStep()
{
	if (params != NULL && dparams != NULL)
	{
		float sugStepSize = -1;
		if (stepSize < 0)
		{
			params->CopyGPUtoCPU();
			dparams->CopyGPUtoCPU();

			double ps = 0;
			double dps = 0;
			float *p = (float*)params->GetCPUMemory();
			float *dp = (float*)dparams->GetCPUMemory();
			for (int i = 0; i < params->GetSize(); i++)
			{
				ps += abs(p[i]);
				dps += abs(dp[i]);
			}

			sugStepSize = (float)((-stepSize) * ((float)(ps + 0.01f) / (dps + 0.01f)));
		}

		float ss = stepSize < 0 ? sugStepSize : stepSize;
		int ps = params->GetSize();
		void* pp[] = {(void*)params->GetGPUMemory(), (void*)dparams->GetGPUMemory(), (void*)&ss, (void*)&ps};
		gpu->Execute((char*)"make_step_kernel", (char*)"ppfi\0", pp, params->GetSize());
		gpu->WaitForGPUToFinish();
	}
}

void DNNLayer::RemoveDeltaInput()
{
	if (deltaInput != NULL)
	{
		delete deltaInput;
		deltaInput = NULL;
	}
}

void DNNLayer::SetTrainRun()
{
	trainRun = true;
}

void DNNLayer::SetTestRun()
{
	trainRun = false;
}
