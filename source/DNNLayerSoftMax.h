#ifndef DNNLAYERSOFTMAX
#define DNNLAYERSOFTMAX

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerSoftMax : public DNNLayer
{
public:
	DNNLayerSoftMax(GPU *_gpu, int _inputWidth, int _batchSize);
	~DNNLayerSoftMax();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
