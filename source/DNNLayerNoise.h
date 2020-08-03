#ifndef DNNLAYERNOISECUH
#define DNNLAYERNOISECUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerNoise : public DNNLayer
{
private:
	CPUGPUMemory *dom;
	float noisesize;
public:
	DNNLayerNoise(GPU *_gpu, int _inputWidth, int _batchSize, float _noisesize);
	~DNNLayerNoise();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
