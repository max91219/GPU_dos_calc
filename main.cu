#include<boost/program_options.hpp>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>

//CUDA includes
#include <assert.h>
#include<curand_kernel.h>
#include <math_constants.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

//Launch params
#define THREADS_PER_BLOCK 256 //1024
#define NUM_BLOCKS 64

//Handles errors from curand
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

unsigned int good_seed(int thread_id) {
  unsigned int random_seed, random_seed_a, random_seed_b;

  std::ifstream file ("/dev/urandom", std::ios::binary);
  if (file.is_open()) {
    char * memblock;
    int size = sizeof(int);
    memblock = new char [size];
    file.read (memblock, size);
    file.close();
    random_seed_a = *reinterpret_cast<int*>(memblock);
    delete[] memblock;
  }

  random_seed_b = std::time(0);
  random_seed = random_seed_a xor random_seed_b xor thread_id;
  return random_seed;
}

//Define the kernels and device functions
__device__ int get_global_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_local_id() {
  return threadIdx.x + threadIdx.y * blockDim.x;
}

__global__ void set_up_rngs(curandState *state, unsigned int *seeds) {
  int id = get_global_id();
  curand_init(seeds[id], 0, 0, &state[id]);
}

__device__ float calc_dispersion(float kx, float ky, float kz) {
  return -4.0f * ((std::cos(kx/2.0f)*std::cos(ky/2.0f)) +
                  (std::cos(kz/2.0f)*std::cos(ky/2.0f)) +
                  (std::cos(kx/2.0f)*std::cos(kz/2.0f))) -
    2.0 * 0.25 * (std::cos(kx) +
                  std::cos(ky) +
                  std::cos(kz));
}

__global__ void gen_disp_histogram(curandStateMtgp32 *state, unsigned int *local_hist, float* global_hist, int samples, int bins, float lbe, float d_eps) {
  int g_id = get_global_id();
  int l_id = get_local_id();

  //copy state from global memory into local memory for faster access
  //curandState local_state = state[g_id];

  //calculate where this thread blocks histogram starts
  unsigned int *l_hist = local_hist + blockIdx.x * bins;

  float kx, ky, kz, disp;
  unsigned int bin_ind;

  for(int s = 0; s < samples; s++) {
    kx = curand_uniform(&state[blockIdx.x]) * 2.0f * CUDART_PI_F;
    ky = curand_uniform(&state[blockIdx.x]) * 2.0f * CUDART_PI_F;
    kz = curand_uniform(&state[blockIdx.x]) * 2.0f * CUDART_PI_F;

    //Some asserts to check we are in bounds
    assert(kx >= 0 && kx <= 2.0f * CUDART_PI_F);
    assert(ky >= 0 && ky <= 2.0f * CUDART_PI_F);
    assert(kz >= 0 && kz <= 2.0f * CUDART_PI_F);

    disp = calc_dispersion(kx, ky, kz);
    assert(disp <= 3.5f && disp >= -13.5f);

    bin_ind = (unsigned int) floor(((disp - lbe) / d_eps));

    //Stops collisions when adding to block histogram
    atomicAdd(&l_hist[bin_ind],1);
  }

  //Make sure that all the threads have finished sampling before we
  //add up the global histogram
  __syncthreads();


  //put the new state back in global memory incase we use it again
  //this stops us generateing the same seqence if we re run the kernel
  //state[g_id] = local_state;

  //Now we are done generateing the local histogram we can fold it into
  //one global histogram from all the blocks.
  //We will need to be careful here of integer and float overflow
  //if we are using alot of samples per thread.
  //minimise this by adding up the bin count times d_eps (small number)
  //not the total bin count as an int
  int g_bin_id = l_id;

  while (g_bin_id < bins) {

    float val = (float)l_hist[g_bin_id] / (float)( (float)samples * (float)THREADS_PER_BLOCK);
    atomicAdd(&global_hist[g_bin_id],val);

    //keep shifting the block untill we have added all the bins
    g_bin_id += blockDim.x;
  }

}

//Main control flow of code
int main(int argc, char* argv[]) {

  // Set up the program options
  namespace po = boost::program_options;
  po::options_description description("Options");
  description.add_options()
    ("o_file,o", po::value<std::string>()->default_value("dos.dat"), "output file name")
    ("n_sample,n", po::value<int>()->default_value(10000000), "number of samples")
    ("num_vals,N", po::value<int>()->default_value(34000), "number of points in the epsilon grid");

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << "USAGE: " << argv[0] << std::endl;
      std::cout << description << std::endl;
      return 0;
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl;
    std::cout << "USAGE: " << argv[0] << std::endl;
    std::cout << description << std::endl;
    return EXIT_FAILURE;
  }

  int N = vm["num_vals"].as<int>();
  int n = vm["n_sample"].as<int>();
  std::string o_file = vm["o_file"].as<std::string>();

  float upper_band_edge = 3.5f;
  float lower_band_edge = -13.5f;
  float d_eps = (upper_band_edge - lower_band_edge) / N;

  std::cout << "num_vals: " << N << std::endl;
  std::cout << "n_sample: " << n << std::endl;
  std::cout << "o_file: " << o_file << std::endl;
  std::cout << "d_eps: " << d_eps << std::endl;
  std::cout << "1/d_eps: " << 1.0/d_eps << std::endl;
  std::cout << "Total number of samples: " << n*(float)(NUM_BLOCKS * THREADS_PER_BLOCK) << std::endl;

  std::vector<float> h_hist(N);
  unsigned int seed = good_seed(time(NULL));

  //initialise cuda stuff
  curandStateMtgp32 *d_states;
  mtgp32_kernel_params *d_kernel_params;
  unsigned int *d_local_hist;
  float *d_global_hist;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //allocate device memory
  gpu_error_check(cudaMalloc((void **)&d_states,  NUM_BLOCKS * sizeof(curandStateMtgp32)));
  gpu_error_check(cudaMalloc((void **)&d_kernel_params, sizeof(mtgp32_kernel_params)));
  gpu_error_check(cudaMalloc(&d_local_hist, NUM_BLOCKS * N * sizeof(unsigned int)));
  gpu_error_check(cudaMalloc(&d_global_hist, N * sizeof(float)));

  std::cout << "Setting up RNG states on the device" << std::endl;
  CURAND_error_check(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, d_kernel_params));
  CURAND_error_check(curandMakeMTGP32KernelState(d_states, mtgp32dc_params_fast_11213, d_kernel_params, NUM_BLOCKS, seed));

  std::cout << "launching " << NUM_BLOCKS << " blocks of " << THREADS_PER_BLOCK << " threads" << std::endl;

  //Set the histograms to zero to start
  gpu_error_check(cudaMemset(d_local_hist, (unsigned int) 0, NUM_BLOCKS * N * sizeof(unsigned int)));
  gpu_error_check(cudaMemset(d_global_hist, (float) 0, N * sizeof(float)));

  //generate the histograms
  std::cout << "Generating Density of States" << std::endl;
  cudaEventRecord(start);
  gen_disp_histogram <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (d_states, d_local_hist, d_global_hist, n, N, lower_band_edge, d_eps);
  gpu_error_check(cudaDeviceSynchronize());
  gpu_error_check(cudaPeekAtLastError());
  cudaEventRecord(stop);

  //Copy then results back
  gpu_error_check(cudaMemcpy(h_hist.data(), d_global_hist, N * sizeof(float), cudaMemcpyDeviceToHost));

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Sampling took: " << milliseconds/1000.0f << " s" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //free device memory
  gpu_error_check(cudaFree(d_states));
  gpu_error_check(cudaFree(d_local_hist));
  gpu_error_check(cudaFree(d_global_hist));

  std::cout << "Writing to file" << std::endl;
  std::ofstream ofile;
  ofile.open(o_file.c_str());
  if (ofile.is_open()){
    for (int i = 0; i < h_hist.size(); i++){
      ofile << (i*d_eps)+lower_band_edge << " " << h_hist[i]/((float)NUM_BLOCKS * d_eps) << std::endl;
    }

    ofile.close();
  }


  return 0;
}
