#define MAXX1X2 784
#define MAXNUMCONVY1Y2 64
#define PMAXX1X2 784 //1 * 28 * 28
#define MAXINP 5000
#define MAXOUTP 5000

#ifndef NOKERNELS

inline float atomicAdd(volatile __global float* address, const float value){
    float old = value;
    while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f);
    return old;
}

__kernel void make_step_kernel(__global float *pars, __global float *dpars, float stepSize, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		pars[tid] -= dpars[tid] * stepSize;
		dpars[tid] = 0;
	}
}

__kernel void preprocess_forward(__global float *outp, __global float *noise, __global float *inp, int inputWidth, int outputWidth, int x1, int x2, int batchSize, float angle, float sx, float sy, int nx1, int nx2, int trainRun)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		if(trainRun == 1)
		{
			float t_outp[PMAXX1X2];
			float t_tmp[PMAXX1X2];
			
			for(int i=0;i<outputWidth;i++)
			{
				t_outp[i] = 0.0f;
				t_tmp[i] = 0.0f;
			}
		
			for(int i=0;i<nx1;i++)
			{
				for(int j=0;j<nx2;j++)
				{
					int ii = (int)i * x1 / nx1;
					int jj = (int)j * x2 / nx2;
					float p = inp[tid * inputWidth + ii * x2 + jj];
					
					float iii = (i + 0.0f) / (nx1 + 0.0f) - 0.5f;
					float jjj = (j + 0.0f) / (nx2 + 0.0f) - 0.5f;
					
					float iiii = (cos(angle) * iii - sin(angle) * jjj) * sx + 0.5f;
					float jjjj = (sin(angle) * iii + cos(angle) * jjj) * sy + 0.5f;
					
					int fi = (int)((float)iiii * (x1 + 0.0f));
					int fj = (int)((float)jjjj * (x2 + 0.0f));
					
					if(fi>=0 && fi<x1 && fj>=0 && fj<x2)
					{
						t_outp[fi * x2 + fj] = (t_outp[fi * x2 + fj] * t_tmp[fi * x2 + fj] + p) / (t_tmp[fi * x2 + fj] + 1.0f);
						t_tmp[fi * x2 + fj] += 1.0f;
					}
				}
			}
			for(int i=0;i<outputWidth;i++)
			{
				outp[tid * outputWidth + i] = t_outp[i] + noise[tid * outputWidth + i];
			}	
		}
		else
		{
			for(int i=0;i<outputWidth;i++)
			{
				outp[tid * outputWidth + i] = inp[tid * outputWidth + i];
			}	
		}
	}
}

__kernel void convolution_forward(__global float *outp, __global float *inp, __global float *pars, int numPics, int inputWidth, int outputWidth, int numConvolutions, int x1, int x2, int y1, int y2, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		float pics[MAXX1X2];
		for (int i = 0; i < x1 * x2 * numPics; i++)
		{
			pics[i] = inp[tid * inputWidth + i];
		}
		float convos[MAXNUMCONVY1Y2];
		
		int pos = 0;
		for (int c = 0; c < numConvolutions; c++)
		{
			for (int i = 0; i < y1 * y2; i++)
			{
				convos[i] = pars[c * y1 * y2 + i];
			}
			
			for (int p = 0; p < numPics; p++)
			{
				for (int i = 0; i < x1 - y1 + 1; i++)
				{
					for (int j = 0; j < x2 - y2 + 1; j++)
					{
						float tmp = 0;
						for (int k = 0; k < y1; k++)
						{
							for (int l = 0; l < y2; l++)
							{
								tmp += pics[p * x1 * x2 + (i + k) * x2 + (j + l)] * convos[k * y2 + l];
							}
						}
						outp[tid * outputWidth + pos] = tmp;
						pos++;
					}
				}
			}
		}
	}
}

__kernel void convolution_backward(__global float *dinp, __global float *dpars, __global  float *doutp, __global  float *outp, __global  float *inp, __global  float *pars, int numPics, int inputWidth, int outputWidth, int numConvolutions, int x1, int x2, int y1, int y2, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		float pics[MAXX1X2];
		for (int i = 0; i < x1 * x2 * numPics; i++)
		{
			pics[i] = inp[tid * inputWidth + i];
		}
		float convos[MAXNUMCONVY1Y2];
		
		int pos = 0;
		for (int c = 0; c < numConvolutions; c++)
		{
			for (int i = 0; i < y1 * y2; i++)
			{
				convos[i] = pars[c * y1 * y2 + i];
			}
			
			for (int p = 0; p < numPics; p++)
			{
				for (int i = 0; i < x1 - y1 + 1; i++)
				{
					for (int j = 0; j < x2 - y2 + 1; j++)
					{
						float tmp = doutp[tid * outputWidth + pos];
						if (tmp != 0)
						{
							for (int k = 0; k < y1; k++)
							{
								for (int l = 0; l < y2; l++)
								{
									if (dinp != NULL)
									{
										dinp[tid * inputWidth + p * x1 * x2 + (i + k) * x2 + (j + l)] += tmp * convos[k * y2 + l];
									}
									atomicAdd(&(dpars[c * y1 * y2 + k * y2 + l]), tmp * pics[p * x1 * x2 + (i + k) * x2 + (j + l)]);
								}
							}
						}
						pos++;
					}
				}
			}
		}
	}
}

__kernel void dropout_forward(__global float *outp, __global  float *inp, __global  float *dom, int inputWidth, float perc, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			if (dom[tid * inputWidth + i] < 1.0f - perc)
			{
				outp[tid * inputWidth + i] = inp[tid * inputWidth + i];
			}
			else
			{
				outp[tid * inputWidth + i] = 0;
			}
		}
	}
}

__kernel void dropout_backward(__global float *dinp, __global  float *doutp, __global  float *outp, __global  float *inp, __global  float *dom, int inputWidth, float perc, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			if (dom[tid * inputWidth + i] < 1.0f - perc)
			{
				dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i];
			}
		}
	}
}

__kernel void noise_forward(__global float *outp, __global  float *inp, __global  float *dom, int inputWidth, float perc, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			outp[tid * inputWidth + i] = inp[tid * inputWidth + i] + dom[tid * inputWidth + i];
		}
	}
}

__kernel void noise_backward(__global float *dinp, __global  float *doutp, __global  float *outp, __global  float *inp, __global  float *dom, int inputWidth, float perc, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i];
		}
	}
}

__kernel void error_square_kernel(__global float *error, __global float *dinput, __global  float *expOutp, __global  float *outp, int inputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			float tmp = outp[tid * inputWidth + i] - expOutp[tid * inputWidth + i];
			error[tid * inputWidth + i] = tmp * tmp;
			dinput[tid * inputWidth + i] = 2.0f * tmp;
		}
	}
}

__kernel void error_log_kernel(__global float *error, __global float *dinput, __global  float *expOutp, __global float *outp, int inputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			if (expOutp[tid * inputWidth + i] > 0.5f)
			{
				error[tid * inputWidth + i] = -log(0.001f + outp[tid * inputWidth + i]);
				dinput[tid * inputWidth + i] = -1.0f / (0.001f + outp[tid * inputWidth + i]);
			}
			else
			{
				error[tid * inputWidth + i] = -log(1.001f - outp[tid * inputWidth + i]);
				dinput[tid * inputWidth + i] = 1.0f / (1.001f - outp[tid * inputWidth + i]);
			}
		}
	}
}

__kernel void matrix_forward(__global float *outp, __global  float *inp, __global  float *pars, int inputWidth, int outputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		float inp2[MAXINP];
		for(int i=0;i<inputWidth;i++)
		{
			inp2[i] = inp[tid* inputWidth + i];
		}
	
		for (int i = 0; i < outputWidth; i++)
		{
			float tmp = pars[i * (inputWidth + 1)];
			for (int j = 0; j < inputWidth; j++)
			{
				tmp += pars[i * (inputWidth + 1) + 1 + j] * inp2[j];
			}
			outp[tid * outputWidth + i] = tmp;
		}
	}
}

__kernel void matrix_backward(__global float *dinp, __global float *dpars, __global  float *doutp, __global  float *outp, __global  float *inp, __global  float *pars, int inputWidth, int outputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		float inp2[MAXINP];
		for(int i=0;i<inputWidth;i++)
		{
			inp2[i] = inp[tid* inputWidth + i];
		}
		float doutp2[MAXOUTP];
		for(int i=0;i<outputWidth;i++)
		{
			doutp2[i] = doutp[tid * outputWidth + i];
		}
	
		for (int i = 0; i < outputWidth; i++)
		{
			atomicAdd(&(dpars[i * (inputWidth + 1)]), doutp2[i]);
			for (int j = 0; j < inputWidth; j++)
			{
				if (dinp != NULL)
				{
					dinp[tid* inputWidth + j] += doutp2[i] * pars[i * (inputWidth + 1) + 1 + j];
				}
				atomicAdd(&(dpars[i * (inputWidth + 1) + 1 + j]), doutp2[i] * inp2[j]);
			}
		}
	}
}

__kernel void and_forward(__global float *outp, __global  float *inp, __global  float *pars, int inputWidth, int outputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < outputWidth; i++)
		{
			float tmp = pars[i * (inputWidth + 1)];
			for (int j = 0; j < inputWidth; j++)
			{
				tmp *= pow(0.01f + 0.99f * inp[tid* inputWidth + j], pars[i * (inputWidth + 1) + 1 + j]);
			}
			outp[tid * outputWidth + i] = tmp;
		}
	}
}

__kernel void and_backward(__global float *dinp, __global float *dpars, __global  float *doutp, __global  float *outp, __global  float *inp, __global  float *pars, int inputWidth, int outputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < outputWidth; i++)
		{
			atomicAdd(&(dpars[i * (inputWidth + 1)]), doutp[tid * outputWidth + i] * outp[tid * outputWidth + i] / pars[i * (inputWidth + 1)]);
			for (int j = 0; j < inputWidth; j++)
			{
				if (dinp != NULL)
				{
					dinp[tid* inputWidth + j] += doutp[tid * outputWidth + i] * outp[tid * outputWidth + i] * pars[i * (inputWidth + 1) + 1 + j] / (0.01f + 0.99f * inp[tid* inputWidth + j]);
				}
				atomicAdd(&(dpars[i * (inputWidth + 1) + 1 + j]), doutp[tid * outputWidth + i] * outp[tid * outputWidth + i] * log(0.01f + 0.99f * inp[tid* inputWidth + j]));
			}
		}
	}
}

__kernel void max_forward(__global float *outp, __global  float *inp, int numPics, int x1, int x2, int d1, int d2, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int p = 0; p < numPics; p++)
		{
			for (int i1 = 0; i1 < x1 / d1; i1++)
			{
				for (int i2 = 0; i2 < x2 / d2; i2++)
				{
					float tmp = -FLT_MAX;
					for (int j1 = 0; j1 < d1; j1++)
					{
						for (int j2 = 0; j2 < d2; j2++)
						{
							if (tmp < inp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + j1) * x2 + (i2 * d2 + j2)])
							{
								tmp = inp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + j1) * x2 + (i2 * d2 + j2)];
							}
						}
					}
					outp[tid * numPics * (x1 / d1) * (x2 / d2) + p * (x1 / d1) * (x2 / d2) + i1 * (x2 / d2) + i2] = tmp;
				}
			}
		}
	}
}

__kernel void max_backward(__global float *dinp, __global  float *doutp, __global  float *outp, __global  float *inp, int numPics, int x1, int x2, int d1, int d2, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int p = 0; p < numPics; p++)
		{
			for (int i1 = 0; i1 < x1 / d1; i1++)
			{
				for (int i2 = 0; i2 < x2 / d2; i2++)
				{
					float tmp = -FLT_MAX;
					int pos1 = -1;
					int pos2 = -1;
					for (int j1 = 0; j1 < d1; j1++)
					{
						for (int j2 = 0; j2 < d2; j2++)
						{
							if (tmp < inp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + j1) * x2 + (i2 * d2 + j2)])
							{
								tmp = inp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + j1) * x2 + (i2 * d2 + j2)];
								pos1 = j1;
								pos2 = j2;
							}
						}
					}
					dinp[tid * numPics * x1 * x2 + p * x1 * x2 + (i1 * d1 + pos1) * x2 + (i2 * d2 + pos2)] += doutp[tid * numPics * (x1 / d1) * (x2 / d2) + p * (x1 / d1) * (x2 / d2) + i1 * (x2 / d2) + i2];
				}
			}
		}
	}
}

__kernel void movieuser_forward(__global float *outp, __global  int *inp, __global  float *params, int numUsers, int numMovies, int vectorWidthMovie, int vectorWidthUser, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		int m = inp[2 * tid];
		int u = inp[2 * tid + 1];

		for (int i = 0; i < vectorWidthMovie; i++)
		{
			outp[tid * (vectorWidthMovie + vectorWidthUser) + i] = params[m * vectorWidthMovie + i];
		}
		for (int i = 0; i < vectorWidthUser; i++)
		{
			outp[tid * (vectorWidthMovie + vectorWidthUser) + vectorWidthMovie + i] = params[numMovies * vectorWidthMovie + u * vectorWidthUser + i];
		}
	}
}

__kernel void movieuser_backward(__global float *dparams, __global  float *doutp, __global  float *outp, __global  int *inp, __global  float *params, int numUsers, int numMovies, int vectorWidthMovie, int vectorWidthUser, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		int m = inp[2 * tid];
		int u = inp[2 * tid + 1];

		for (int i = 0; i < vectorWidthMovie; i++)
		{
			atomicAdd(&(dparams[m * vectorWidthMovie + i]), doutp[tid * (vectorWidthMovie + vectorWidthUser) + i]);
		}
		for (int i = 0; i < vectorWidthUser; i++)
		{
			atomicAdd(&(dparams[numMovies * vectorWidthMovie + u * vectorWidthUser + i]), doutp[tid * (vectorWidthMovie + vectorWidthUser) + vectorWidthMovie + i]);
		}
	}
}

__kernel void relu_forward(__global float *outp, __global  float *inp, int inputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			if (inp[tid * inputWidth + i] > 0)
			{
				outp[tid * inputWidth + i] = inp[tid * inputWidth + i];
			}
			else
			{
				outp[tid * inputWidth + i] = 0;
			}
		}
	}
}

__kernel void relu_backward(__global float *dinp, __global  float *doutp, __global  float *outp, __global  float *inp, int inputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			if (inp[tid * inputWidth + i] > 0)
			{
				dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i];
			}
			else
			{
				dinp[tid * inputWidth + i] = 0;
			}
		}
	}
}

__kernel void sigmoid_forward(__global float *outp, __global  float *inp, int inputWidth, int batchSize, float o_min, float o_max)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			outp[tid * inputWidth + i] = o_min + (o_max - o_min) / (1.0f + exp(inp[tid * inputWidth + i]));
		}
	}
}

__kernel void sigmoid_backward(__global float *dinp, __global  float *doutp, __global  float *outp, __global  float *inp, int inputWidth, int batchSize, float o_min, float o_max)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		for (int i = 0; i < inputWidth; i++)
		{
			float tmp = (outp[tid * inputWidth + i] - o_min) / (o_max - o_min);
			dinp[tid * inputWidth + i] = doutp[tid * inputWidth + i] * tmp * (tmp - 1.0f) * (o_max - o_min);
		}
	}
}

__kernel void softmax_forward(__global float *outp, __global  float *inp, int inputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		float sum = 0.0f;
		for (int i = 0; i < inputWidth; i++)
		{
			float tmp = exp(inp[tid * inputWidth + i]);
			outp[tid * inputWidth + i] = tmp;
			sum += tmp;
		}
		for (int i = 0; i < inputWidth; i++)
		{
			outp[tid * inputWidth + i] /= sum;
		}
	}
}

__kernel void softmax_backward(__global float *dinp, __global  float *doutp, __global  float *outp, __global  float *inp, int inputWidth, int batchSize)
{
	int tid = get_global_id(0);

	if (tid < batchSize)
	{
		float sum = 0.0f;
		for (int i = 0; i < inputWidth; i++)
		{
			float tmp = exp(inp[tid * inputWidth + i]);
			sum += tmp;
		}
		for (int j = 0; j < inputWidth; j++)
		{
			for (int i = 0; i < inputWidth; i++)
			{
				if (i == j)
				{
					dinp[tid * inputWidth + i] += doutp[tid * inputWidth + j] * (sum - exp(inp[tid * inputWidth + i])) / (sum * sum) * exp(inp[tid * inputWidth + i]);
				}
				else
				{
					dinp[tid * inputWidth + i] -= doutp[tid * inputWidth + j] * exp(inp[tid * inputWidth + j]) / (sum * sum) * exp(inp[tid * inputWidth + i]);
				}
			}
		}
	}
}

#endif
