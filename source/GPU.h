#ifndef GPUCUH
#define GPUCUH

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/opencl.h>

class GPU
{
private:
	void InitGPU();
	void ReleaseGPU();

	cl_platform_id cpPlatform;        // OpenCL platform
	cl_device_id device_id;           // device ID
	cl_context context;               // context
	cl_command_queue queue;           // command queue
	cl_program program;               // program
	cl_kernel kernel;                 // kernel
public:
	GPU();
	~GPU();

	void WaitForGPUToFinish();
	cl_mem GetMemory(int size);
	void FreeMemory(cl_mem mem);
	void CopyCPUtoGPU(void *cpu_mem, cl_mem gpu_mem, int size);
	void CopyGPUtoCPU(void *cpu_mem, cl_mem gpu_mem, int size);
	void Execute(char *function_name, char *pars, void **objects, int globalSize);
};

#endif
