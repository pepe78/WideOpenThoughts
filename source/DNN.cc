#include "DNN.h"

#include "StringUtils.h"
#include "DNNLayerConvolution.h"
#include "DNNLayerMatrix.h"
#include "DNNLayerMax.h"
#include "DNNLayerSigmoid.h"
#include "DNNLayerSoftMax.h"
#include "DNNLayerRelu.h"
#include "DNNLayerDropout.h"
#include "DNNLayerMovieUser.h"

#include <fstream>
#include <string>
#include <math.h>

using namespace std;

DNN::DNN(GPU *_gpu, string &configFile, string &trainSetFile, string &testSetFile, int _batchSize, string &paramFile, int _saveEvery, string &errorType, string &_whereMax)
{
	gpu = _gpu;
	bool inputIsFloat = true;
	saveEvery = _saveEvery;
	ifstream is(configFile.c_str());
	if (is.is_open())
	{
		string line;
		while (getline(is, line))
		{
			if (line.length() > 0)
			{
				printf("Layer %d: %s\n", (int)layers.size(), line.c_str());
				vector<string> parts;
				split_without_space(line, parts, ',');

				if (parts[0].compare("convolution") == 0)
				{
					if (parts.size() != 9)
					{
						fprintf(stderr, "wrong setup of convolution layer!\n");
						exit(-1);
					}
					int numPics = convertToInt(parts[1]);
					int x1 = convertToInt(parts[2]);
					int x2 = convertToInt(parts[3]);
					int numConvo = convertToInt(parts[4]);
					int y1 = convertToInt(parts[5]);
					int y2 = convertToInt(parts[6]);
					float initVal = convertToFloat(parts[7]);
					float stepSize = convertToFloat(parts[8]);

					DNNLayer *curLayer = new DNNLayerConvolution(_gpu, numPics, x1, x2, numConvo, y1, y2, _batchSize, initVal, stepSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("matrix") == 0)
				{
					if (parts.size() != 5)
					{
						fprintf(stderr, "wrong setup of matrix layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);
					int outpWidth = convertToInt(parts[2]);
					float initVal = convertToFloat(parts[3]);
					float stepSize = convertToFloat(parts[4]);

					DNNLayer *curLayer = new DNNLayerMatrix(_gpu, inpWidth, outpWidth, _batchSize, initVal, stepSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("sigmoid") == 0)
				{
					if (parts.size() != 2 && parts.size() != 4)
					{
						fprintf(stderr, "wrong setup of sigmoid layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);
					float o_min = 0;
					float o_max = 1;

					if (parts.size() == 4)
					{
						o_min = convertToFloat(parts[2]);
						o_max = convertToFloat(parts[3]);
					}

					DNNLayer *curLayer = new DNNLayerSigmoid(_gpu, inpWidth, o_min, o_max, _batchSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("relu") == 0)
				{
					if (parts.size() != 2)
					{
						fprintf(stderr, "wrong setup of relu layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);

					DNNLayer *curLayer = new DNNLayerRelu(_gpu, inpWidth, _batchSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("softmax") == 0)
				{
					if (parts.size() != 2)
					{
						fprintf(stderr, "wrong setup of softmax layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);

					DNNLayer *curLayer = new DNNLayerSoftMax(_gpu, inpWidth, _batchSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("dropout") == 0)
				{
					if (parts.size() != 3)
					{
						fprintf(stderr, "wrong setup of dropout layer!\n");
						exit(-1);
					}
					int inpWidth = convertToInt(parts[1]);
					float perc = convertToFloat(parts[2]);

					DNNLayer *curLayer = new DNNLayerDropout(_gpu, inpWidth, _batchSize, perc);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("max") == 0)
				{
					if (parts.size() != 6)
					{
						fprintf(stderr, "wrong setup of max layer!\n");
						exit(-1);
					}
					int numPics = convertToInt(parts[1]);
					int x1 = convertToInt(parts[2]);
					int x2 = convertToInt(parts[3]);
					int d1 = convertToInt(parts[4]);
					int d2 = convertToInt(parts[5]);

					DNNLayer *curLayer = new DNNLayerMax(_gpu, numPics, x1, x2, d1, d2, _batchSize);
					layers.push_back(curLayer);
				}
				else if (parts[0].compare("movieuser") == 0)
				{
					if (parts.size() != 7)
					{
						fprintf(stderr, "wrong setup of movieuser layer!\n");
						exit(-1);
					}
					int numMovies = convertToInt(parts[1]);
					int numUsers = convertToInt(parts[2]);
					int vectorWidthMovie = convertToInt(parts[3]);
					int vectorWidthUser = convertToInt(parts[4]);
					float initVal = convertToFloat(parts[5]);
					float stepSize = convertToFloat(parts[6]);

					inputIsFloat = false;
					DNNLayer *curLayer = new DNNLayerMovieUser(_gpu, numMovies, numUsers, vectorWidthMovie, vectorWidthUser, initVal, stepSize, _batchSize);
					layers.push_back(curLayer);
				}
				else
				{
					fprintf(stderr, "type of layer %s not implemented yet!\n", parts[0].c_str());
					exit(-1);
				}

				if (layers.size() > 1 && layers[layers.size() - 2]->GetOutputWidth() != layers[layers.size() - 1]->GetInputWidth())
				{
					fprintf(stderr, "outputs of layer does not match input of layer!\n");
					exit(-1);
				}
			}
		}
	}
	trainSet = new DataSet(_gpu, trainSetFile, layers[0]->GetInputWidth(), inputIsFloat, layers[layers.size() - 1]->GetOutputWidth(), true, _batchSize);
	testSet = new DataSet(_gpu, testSetFile, layers[0]->GetInputWidth(), inputIsFloat, layers[layers.size() - 1]->GetOutputWidth(), true, _batchSize);

	errorLayer = new DNNLayerError(_gpu, layers[layers.size() - 1]->GetOutputWidth(), _batchSize, errorType.compare("square") == 0);

	epoch = 0;

	layers[0]->RemoveDeltaInput();
	LoadFromFile(paramFile);

	whereMax = _whereMax.compare("wheremax") == 0;
	netflixRun = _whereMax.compare("netflix") == 0;
}

DNN::~DNN()
{
	for (size_t i = 0; i < layers.size(); i++)
	{
		delete layers[i];
	}
	layers.clear();

	if (trainSet != NULL)
	{
		delete trainSet;
	}
	if (testSet != NULL)
	{
		delete testSet;
	}

	delete errorLayer;
}

void DNN::Train()
{
	while (true)
	{
		printf("Epoch %d \n", epoch);
		TrainEpoch();
		Test();
		SaveToFile();
		epoch++;
	}
}

void DNN::Forward(CPUGPUMemory *firstInput, bool isTrain)
{
	for (size_t i = 0; i < layers.size(); i++)
	{
		if (isTrain)
		{
			layers[i]->SetTrainRun();
		}
		else
		{
			layers[i]->SetTestRun();
		}
		layers[i]->Forward(i == 0 ? firstInput : layers[i - 1]->GetOutput());
		gpu->WaitForGPUToFinish();
	}
}

void DNN::BackWard(CPUGPUMemory *firstInput)
{
	for (int i = (int)layers.size() - 1; i >= 0; i--)
	{
		layers[i]->SetTrainRun();
		layers[i]->ResetDeltaInput();
		layers[i]->Backward(i == 0 ? firstInput : layers[i - 1]->GetOutput(), i == layers.size() - 1 ? errorLayer->GetDeltaInput() : layers[i + 1]->GetDeltaInput());
		gpu->WaitForGPUToFinish();

		layers[i]->MakeStep();
	}
}

int DNN::ComputeCorrect(CPUGPUMemory *expected_output, CPUGPUMemory *output)
{
	int ret = 0;
	int outputWidth = layers[layers.size() - 1]->GetOutputWidth();
	int numSamples = expected_output->GetSize() / outputWidth;

	expected_output->CopyGPUtoCPU();
	output->CopyGPUtoCPU();

	float* eo = (float*)expected_output->GetCPUMemory();
	float* o = (float*)output->GetCPUMemory();
	for (int i = 0; i < numSamples; i++)
	{
		int p1 = 0;
		for (int j = 1; j < outputWidth; j++)
		{
			if (eo[i * outputWidth + p1] < eo[i * outputWidth + j])
			{
				p1 = j;
			}
		}

		int p2 = 0;
		for (int j = 1; j < outputWidth; j++)
		{
			if (o[i * outputWidth + p2] < o[i * outputWidth + j])
			{
				p2 = j;
			}
		}
		if (p1 == p2)
		{
			ret++;
		}
	}

	return ret;
}

double DNN::ComputeNetflix(CPUGPUMemory *expected_output, CPUGPUMemory *output)
{
	double ret = 0;
	int numSamples = expected_output->GetSize();

	expected_output->CopyGPUtoCPU();
	output->CopyGPUtoCPU();

	float* eo = (float*)expected_output->GetCPUMemory();
	float* o = (float*)output->GetCPUMemory();
	for (int i = 0; i < numSamples; i++)
	{
		float o1 = eo[i];
		float o2 = o[i];
		if (o2 < 1.0f)
		{
			o2 = 1.0f;
		}
		if (o2 > 5.0f)
		{
			o2 = 5.0f;
		}

		ret += (o1 - o2)*(o1 - o2);
	}

	return ret;
}

double DNN::TrainBatch(int batchNum)
{
	Forward(trainSet->GetBatchNumber(batchNum)->GetInputs(), true);
	double ret = errorLayer->ComputeError(layers[layers.size() - 1]->GetOutput(), trainSet->GetBatchNumber(batchNum)->GetOutputs());
	BackWard(trainSet->GetBatchNumber(batchNum)->GetInputs());

	return ret;
}

double DNN::TestBatch(int batchNum)
{
	Forward(testSet->GetBatchNumber(batchNum)->GetInputs(), false);
	return errorLayer->ComputeError(layers[layers.size() - 1]->GetOutput(), testSet->GetBatchNumber(batchNum)->GetOutputs());
}

void DNN::TrainEpoch()
{
	vector<int> batchesToDo;
	for (int i = 0; i < trainSet->GetNumBatches(); i++)
	{
		batchesToDo.push_back(i);
	}

	double ret = 0;
	int correct = 0;
	double netflix = 0;
	while(batchesToDo.size() !=0)
	{
		int which = rand() % batchesToDo.size();
		int bn = batchesToDo[which];
		batchesToDo.erase(batchesToDo.begin() + which);
		double curErr = TrainBatch(bn);
		ret += curErr;
		int curCorrect = 0;
		if (whereMax)
		{
			curCorrect = ComputeCorrect(trainSet->GetBatchNumber(bn)->GetOutputs(), layers[layers.size() - 1]->GetOutput());
		}
		if (netflixRun)
		{
			netflix += ComputeNetflix(trainSet->GetBatchNumber(bn)->GetOutputs(), layers[layers.size() - 1]->GetOutput());
		}
		correct += curCorrect;
		printf("Train Batch %d (%d) CurError %lf (%d) Error %lf (%d)\n", bn, (int)batchesToDo.size(), curErr, curCorrect, ret, correct);
	}
	printf("TrainError %lf (%d)\n", ret, correct);
	string txt = convertToString(epoch) + ((string)",") + convertToString((float)ret / (trainSet->GetNumSamples() + 0.0f)) + ((string)",") + convertToString(correct / (trainSet->GetNumSamples() + 0.0f)) + ((string)",")
		+ convertToString((float)sqrt(netflix / (trainSet->GetNumSamples() + 0.0f))) + ((string)",");
	AppendToFile("debug.csv", txt);
}

void DNN::Test()
{
	double ret = 0;
	int correct = 0;
	double netflix = 0;
	for (int i = 0; i < testSet->GetNumBatches(); i++)
	{
		double curErr = TestBatch(i);
		ret += curErr;
		int curCorrect = 0;
		if (whereMax)
		{
			curCorrect = ComputeCorrect(testSet->GetBatchNumber(i)->GetOutputs(), layers[layers.size() - 1]->GetOutput());
		}
		if (netflixRun)
		{
			netflix += ComputeNetflix(testSet->GetBatchNumber(i)->GetOutputs(), layers[layers.size() - 1]->GetOutput());
		}
		correct += curCorrect;
		printf("Test Batch %d CurError %lf (%d) Error %lf (%d)\n", i, curErr, curCorrect, ret, correct);
	}
	printf("TestError %lf (%d)\n", ret, correct);
	string txt = convertToString((float)ret / (testSet->GetNumSamples() + 0.0f)) + ((string)",") + convertToString(correct / (testSet->GetNumSamples()+ 0.0f)) + ((string)",")
		+ convertToString((float)sqrt(netflix / (testSet->GetNumSamples() + 0.0f))) + ((string)"\n");
	AppendToFile("debug.csv", txt);
}

void DNN::SaveToFile()
{
	if (epoch % saveEvery == 0)
	{
		string filename = "params_";
		filename += convertToString(epoch);
		filename += ".bin";
		ofstream os(filename.c_str(), ios::out | ios::binary);
		for (size_t i = 0; i < layers.size(); i++)
		{
			CPUGPUMemory *m = layers[i]->GetParams();
			if (m != NULL)
			{
				m->SaveToFile(os);
			}
		}
		os.close();
	}
}

void DNN::LoadFromFile(string &paramFile)
{
	ifstream is(paramFile.c_str(), ios::in | ios::binary);

	if (is.is_open())
	{
		printf("loading file %s\n", paramFile.c_str());
		for (size_t i = 0; i < layers.size(); i++)
		{
			CPUGPUMemory *m = layers[i]->GetParams();
			if (m != NULL)
			{
				m->LoadFromFile(is);
			}
		}
		is.close();

		string numEp = getNumbersOnly(paramFile);
		epoch = convertToInt(numEp) + 1;
	}
}
