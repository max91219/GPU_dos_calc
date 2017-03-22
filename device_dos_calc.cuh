/*
 * device_dos_calc.cuh
 *
 *  Created on: 22 Mar 2017
 *      Author: max
 */

//Launch params
#define THREADS_PER_BLOCK 256 //need to be 256 or less due to limitations of Mersene twister in CURAND
#define NUM_BLOCKS 64

//Handles errors from CURAND
static const char *curandGetErrorString(curandStatus_t error)
{
  switch (error)
    {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
    }

  return "<unknown>";
}

//Useful for checking return values of cuda API calls
#define gpu_error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

//Useful for checking return values of CURAND API calls
#define CURAND_error_check(ans) { curandAssert((ans), __FILE__, __LINE__); }

inline void curandAssert(curandStatus code, const char *file, int line, bool abort=true) {
  if (code != CURAND_STATUS_SUCCESS) {
    fprintf(stderr,"CURANDassert: %s %s %d\n", curandGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__device__ int get_global_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_local_id() {
  return threadIdx.x + threadIdx.y * blockDim.x;
}

__device__ double calc_dispersion(double kx, double ky, double kz) {
  return -4.0 * ((cos(kx/2.0f)*cos(ky/2.0f)) +
                  (cos(kz/2.0f)*cos(ky/2.0f)) +
                  (cos(kx/2.0f)*cos(kz/2.0f))) -
    2.0 * 0.25 * (cos(kx) +
                  cos(ky) +
                  cos(kz));
}

__global__ void gen_disp_histogram(curandStateMtgp32 *state, unsigned long long int *local_hist,
									unsigned long long int *global_hist, int samples, int bins,
									double lbe, double ube, double d_eps) {
  int g_id = get_global_id();
  int l_id = get_local_id();

  //calculate where this thread blocks histogram starts
  unsigned long long int *l_hist = local_hist + blockIdx.x * bins;

  double kx, ky, kz, disp;
  int bin_ind;

  for(int s = 0; s < samples; s++) {
    kx = curand_uniform_double(&state[blockIdx.x]) * 2.0 * CUDART_PI;
    ky = curand_uniform_double(&state[blockIdx.x]) * 2.0 * CUDART_PI;
    kz = curand_uniform_double(&state[blockIdx.x]) * 2.0 * CUDART_PI;

    //Some asserts to check we are in bounds
    assert(kx >= 0 && kx <= 2.0 * CUDART_PI);
    assert(ky >= 0 && ky <= 2.0 * CUDART_PI);
    assert(kz >= 0 && kz <= 2.0 * CUDART_PI);

    disp = calc_dispersion(kx, ky, kz);
    assert(disp <= 3.5 && disp >= -13.5);

    bin_ind = __double2int_rd((disp - lbe)/d_eps);

    //Some asserts to check we are in bounds
    assert(bin_ind >= 0);
    assert(bin_ind < bins);

    //Stops collisions when adding to block histogram
    atomicAdd(&l_hist[bin_ind],1);
  }

  //Make sure that all the threads have finished sampling before we
  //add up the global histogram
  __syncthreads();

  //Now we are done generating the local histograms we can fold it into
  //one global histogram from all the blocks.
  int g_bin_id = l_id;
  while (g_bin_id < bins) {

	  atomicAdd(&global_hist[g_bin_id], l_hist[g_bin_id]);

    //keep shifting the block until we have added all the bins
    g_bin_id += blockDim.x;
  }

}
