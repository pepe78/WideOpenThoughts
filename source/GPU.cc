#include "GPU.h"

#include <cstdlib>
#include <cstdio>
#include <fstream>

using namespace std;

GPU::GPU()
{
	InitGPU();
}

GPU::~GPU()
{
	ReleaseGPU();
}

cl_mem GPU::GetMemory(size_t size)
{
	return clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, NULL);
}

void GPU::FreeMemory(cl_mem mem)
{
	clReleaseMemObject(mem);
}

void GPU::CopyCPUtoGPU(void *cpu_mem, cl_mem gpu_mem, size_t size)
{
	cl_int err = clEnqueueWriteBuffer(queue, gpu_mem, CL_TRUE, 0, size, cpu_mem, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Error copying cpu mem to gpu mem!\n");
		exit(-1);
	}
	WaitForGPUToFinish();
}

void GPU::CopyGPUtoCPU(void *cpu_mem, cl_mem gpu_mem, size_t size)
{
	cl_int err = clEnqueueReadBuffer(queue, gpu_mem, CL_TRUE, 0, size, cpu_mem, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Error copying gpu mem to cpu mem!\n");
		exit(-1);
	}
	WaitForGPUToFinish();
}

void GPU::Execute(char *function_name, char *pars, void **objects, int globalSize)
{
	cl_int err;
	kernel = clCreateKernel(program, function_name, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Error creating kernel!\n");
		exit(-1);
	}

	err = CL_SUCCESS;
	int i = 0;
	while(pars[i]!='\0')
	{
		switch(pars[i])
		{
			case 'p':
				err |= clSetKernelArg(kernel, i, sizeof(cl_mem), &(objects[i]));
				break;
			case 'f':
				err |= clSetKernelArg(kernel, i, sizeof(float), objects[i]);
				break;
			case 'i':
				err |= clSetKernelArg(kernel, i, sizeof(int), objects[i]);
				break;
			default:
				fprintf(stderr, "Not implemented kernel parameter option\n");
				exit(-1);
				break;
		}
		i++;
	}
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Error setting arguments for kernel!\n");
		exit(-1);
	}

	size_t localSize = 256;
	size_t  gs = ((size_t)(globalSize + localSize - 1) / localSize) * localSize;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gs, &localSize, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Error enqueing kernel!\n");
		exit(-1);
	}

	clReleaseKernel(kernel);
}

void GPU::WaitForGPUToFinish()
{
	cl_int err = clFinish(queue);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Error waiting for GPU to finish!\n");
		exit(-1);
	}
}

void GPU::InitGPU()
{
	cl_int err;

	// Bind to platform
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "No platform id!\n");
		exit(-1);
	}

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "No device id!\n");
		exit(-1);
	}

	// Create a context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Context not created!\n");
		exit(-1);
	}

	// Create a command queue
	queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Command queue not created!\n");
		exit(-1);
	}

	// Create the compute program from the source buffer
	ifstream myfile ("./source/kernels.cl");
	int length;
	myfile.seekg(0, std::ios::end);
	length = myfile.tellg();
	myfile.seekg(0, std::ios::beg);
	char *kernelSource = new char[length+1];
	kernelSource[length] = '\0';
	myfile.read(kernelSource, length);
	myfile.close();

	program = clCreateProgramWithSource(context, 1,
	    (const char **) &kernelSource, NULL, &err);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Program creation problem!\n");
		exit(-1);
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		fprintf(stderr, "Kernels did not build!\n");
		size_t length;
    		char buffer[2048];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
		buffer[length+1] = '\0';
		fprintf(stderr, "%s\n", buffer);
		exit(-1);
	}

	fprintf(stdout, "Kernels built succesfully!\n");
}

void GPU::ReleaseGPU()
{
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
