#ifndef DNNLAYERDROPOUTCUH
#define DNNLAYERDROPOUTCUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerDropout : public DNNLayer
{
private:
	CPUGPUMemory *dom;
	float perc;
public:
	DNNLayerDropout(GPU *_gpu, int _inputWidth, int _batchSize, float _perc);
	~DNNLayerDropout();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
