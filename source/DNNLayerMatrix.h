#ifndef DNNLAYERMATRIXCUH
#define DNNLAYERMATRIXCUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerMatrix : public DNNLayer
{
public:
	DNNLayerMatrix(GPU *_gpu, int _inputWidth, int _outputWidth, int _batchSize, float _initVal, float _stepSize);
	~DNNLayerMatrix();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
