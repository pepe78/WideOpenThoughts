#ifndef DNNLAYERAUGMENTMATRIXCUH
#define DNNLAYERAUGMENTMATRIXCUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerAugmentMatrix : public DNNLayer
{
private:
	int x1, x2;
	int numConvolutions;
	int numPics;
public:
	DNNLayerAugmentMatrix(GPU *_gpu, int _numPics, int _x1, int _x2, int _numConvolutions, int _batchSize, float _initVal, float _stepSize);
	~DNNLayerAugmentMatrix();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
