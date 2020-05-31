#ifndef DNNLAYERPREPROCESSCUH
#define DNNLAYERPREPROCESSCUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerPreprocess : public DNNLayer
{
private:
	int x1, x2;
	int x1SamplePoints, x2SamplePoints;
	int flipHor, flipVer;
	float minAngle, maxAngle, minStretch, maxStretch, minNoise, maxNoise;
	CPUGPUMemory *dom;
public:
	DNNLayerPreprocess(GPU *_gpu, int _x1, int _x2, int _batchSize, float _minAngle, float _maxAngle, float _minStretch, float _maxStretch, float _minNoise, float _maxNoise, int _flipHor, int _flipVer, int _x1SamplePoints, int _x2SamplePoints);
	~DNNLayerPreprocess();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
