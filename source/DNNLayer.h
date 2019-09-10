#ifndef DNNLAYERCUH
#define DNNLAYERCUH

#include "CPUGPUMemory.h"
#include "GPU.h"

class DNNLayer
{
protected:
	int inputWidth;
	int outputWidth;
	int numParams;

	float stepSize;

	CPUGPUMemory* deltaInput;
	CPUGPUMemory* output;
	CPUGPUMemory* params;
	CPUGPUMemory* dparams;

	bool trainRun;
	GPU *gpu;
public:
	DNNLayer(GPU *_gpu, int _batchSize, int _inputWidth, int _outputWidth, int _numParams, float _initVal, float _stepSize);
	~DNNLayer();

	virtual void Forward(CPUGPUMemory* input);
	virtual void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
	void ResetDeltaInput();

	int GetInputWidth();
	int GetOutputWidth();
	CPUGPUMemory* GetOutput();
	CPUGPUMemory* GetDeltaInput();
	CPUGPUMemory* GetParams();
	void MakeStep();

	void RemoveDeltaInput();

	void SetTrainRun();
	void SetTestRun();
};

#endif
