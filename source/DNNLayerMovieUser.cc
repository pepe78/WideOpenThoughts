#include "DNNLayerMovieUser.h"

#include <cstdlib>
#include <cstdio>
#include <cfloat>

DNNLayerMovieUser::DNNLayerMovieUser(GPU *_gpu, int _numMovies, int _numUsers, int _vectorWidthMovie, int _vectorWidthUser, float _initValues, float _stepSize, int _batchSize)
	: DNNLayer(_gpu, _batchSize, 2, _vectorWidthMovie + _vectorWidthUser, _numMovies * _vectorWidthMovie + _numUsers * _vectorWidthUser, _initValues, _stepSize)
{
	numMovies = _numMovies;
	numUsers = _numUsers;
	vectorWidthMovie = _vectorWidthMovie;
	vectorWidthUser = _vectorWidthUser;
}

DNNLayerMovieUser::~DNNLayerMovieUser()
{

}

void DNNLayerMovieUser::Forward(CPUGPUMemory* input)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&numUsers, (void*)&numMovies, (void*)&vectorWidthMovie, (void*)&vectorWidthUser, (void*)&t};
	gpu->Execute((char*)"movieuser_forward", (char*)"pppiiiii\0", pp, input->GetSize() / inputWidth);
}

void DNNLayerMovieUser::Backward(CPUGPUMemory* input, CPUGPUMemory* deltaOutput)
{
	int t = (input->GetSize() / inputWidth);
	void* pp[] = {(void*)dparams->GetGPUMemory(), (void*)deltaOutput->GetGPUMemory(),
		(void*)output->GetGPUMemory(), (void*)input->GetGPUMemory(), (void*)params->GetGPUMemory(), (void*)&numUsers, (void*)&numMovies, (void*)&vectorWidthMovie, (void*)&vectorWidthUser, (void*)&t};
	gpu->Execute((char*)"movieuser_backward", (char*)"pppppiiiii\0", pp, input->GetSize() / inputWidth);
}
