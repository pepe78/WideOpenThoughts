#ifndef DNNLAYERMOVIEUSERCUH
#define DNNLAYERMOVIEUSERCUH

#include "DNNLayer.h"
#include "GPU.h"

class DNNLayerMovieUser : public DNNLayer
{
private:
	int numMovies, numUsers;
	int vectorWidthMovie, vectorWidthUser;
public:
	DNNLayerMovieUser(GPU *_gpu, int _numMovies, int _numUsers, int _vectorWidthMovie, int _vectorWidthUser, float _initValues, float _stepSize, int _batchSize);
	~DNNLayerMovieUser();

	void Forward(CPUGPUMemory* input);
	void Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput);
};

#endif
