#include "CPUGPUMemory.h"
#include "RandUtils.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

CPUGPUMemory::CPUGPUMemory(GPU *_gpu, bool _is_float, int _size, float _initValues)
{
	gpu = _gpu;
	is_float = _is_float;
	size = _size;

	memCPU = is_float ? (void*)new float[size] : (void*)new int[size];
	memset(memCPU, 0, size * (is_float ? sizeof(float) : sizeof(int)));
	if (_initValues != 0)
	{
		if (is_float)
		{
			float *t = (float*)memCPU;
			for (int i = 0; i < size; i++)
			{
				t[i] = (getRand() * 2.0f - 1.0f) * _initValues;
			}
		}
	}
	memGPU = (void*)gpu->GetMemory(size *  (is_float ? sizeof(float) : sizeof(int)));
	CopyCPUtoGPU();
}

void CPUGPUMemory::Resize(int newSize)
{
	size = newSize;
}

CPUGPUMemory::~CPUGPUMemory()
{
	if (is_float)
	{
		delete[] (float*)memCPU;
	}
	else
	{
		delete[] (int*)memCPU;
	}
	gpu->FreeMemory((cl_mem)memGPU);
}

void CPUGPUMemory::CopyCPUtoGPU()
{
	gpu->CopyCPUtoGPU(memCPU, (cl_mem)memGPU, size * (is_float ? sizeof(float) : sizeof(int)));
}

void CPUGPUMemory::CopyGPUtoCPU()
{
	gpu->CopyGPUtoCPU(memCPU, (cl_mem)memGPU, size * (is_float ? sizeof(float) : sizeof(int)));
}

void* CPUGPUMemory::GetCPUMemory()
{
	return memCPU;
}

void* CPUGPUMemory::GetGPUMemory()
{
	return memGPU;
}

int CPUGPUMemory::GetSize()
{
	return size;
}

void CPUGPUMemory::Reset()
{
	memset(memCPU, 0, size * (is_float ? sizeof(float) : sizeof(int)));
	CopyCPUtoGPU();
}

void CPUGPUMemory::SaveToFile(ofstream &os)
{
	CopyGPUtoCPU();
	os.write((char*)memCPU, size * (is_float ? sizeof(float) : sizeof(int)));
}

void CPUGPUMemory::LoadFromFile(ifstream &is)
{
	is.read((char*)memCPU, size * (is_float ? sizeof(float) : sizeof(int)));
	CopyCPUtoGPU();
}
