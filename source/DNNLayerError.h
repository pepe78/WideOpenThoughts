#ifndef DNNLAYERRORCUH
#define DNNLAYERRORCUH

#include "CPUGPUMemory.h"
#include "GPU.h"

class DNNLayerError
{
private:
	CPUGPUMemory *error;
	CPUGPUMemory *deltaInput;

	int inputWidth, batchSize;
	bool square;
	GPU *gpu;
public:
	DNNLayerError(GPU *_gpu, int _inputWidth, int _batchSize, bool _square);
	~DNNLayerError();

	double ComputeError(CPUGPUMemory* output, CPUGPUMemory *expectedOutput);
	CPUGPUMemory* GetDeltaInput();
};

#endif
