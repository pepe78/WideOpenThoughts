#ifndef DNNCUH
#define DNNCUH

#include "DNNLayer.h"
#include "GPU.h"
#include "DataSet.h"
#include "DNNLayerError.h"

#include <string>
#include <vector>

using namespace std;

class DNN
{
private:
	vector<DNNLayer*> layers;
	DataSet *trainSet;
	DataSet *testSet;
	DNNLayerError *errorLayer;

	double TrainBatch(int batchNum);
	double TestBatch(int batchNum);
	void TrainEpoch();

	void Forward(CPUGPUMemory *firstInput, bool isTrain);
	void BackWard(CPUGPUMemory *firstInput);

	int ComputeCorrect(CPUGPUMemory *expected_output, CPUGPUMemory *output);
	double ComputeNetflix(CPUGPUMemory *expected_output, CPUGPUMemory *output);
	void SaveToFile();
	void LoadFromFile(string &paramFile);

	int epoch;
	int saveEvery;
	bool whereMax;
	bool netflixRun;
	GPU *gpu;
public:
	DNN(GPU *_gpu, string &configFile, string &trainSetFile, string &testSetFile, int _batchSize, string &paramFile, int _saveEvery, string &errorType, string &_whereMax);
	~DNN();

	void Train();
	void Test();
};

#endif
