#ifndef DNNLAYERRELUCUH
#define DNNLAYERRELUCUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerRelu : public DNNLayer
{
public:
	DNNLayerRelu(GPU *_gpu, int _inputWidth, int _batchSize);
	~DNNLayerRelu();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
