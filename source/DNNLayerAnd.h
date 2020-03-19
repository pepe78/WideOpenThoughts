#ifndef DNNLAYERANDCUH
#define DNNLAYERANDCUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerAnd : public DNNLayer
{
public:
	DNNLayerAnd(GPU *_gpu, int _inputWidth, int _outputWidth, int _batchSize, float _initVal, float _stepSize);
	~DNNLayerAnd();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
