#ifndef DNNLAYERSIGMOIDCUH
#define DNNLAYERSIGMOIDCUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerSigmoid : public DNNLayer
{
private:
	float o_min;
	float o_max;
public:
	DNNLayerSigmoid(GPU *_gpu, int _inputWidth, float _min, float _max, int _batchSize);
	~DNNLayerSigmoid();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
