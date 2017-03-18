#include<boost/program_options.hpp>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>

//CUDA includes
#include<curand_kernel.h>

//Launch params
#define THREADS_PER_BLOCK 1024
#define NUM_BLOCKS 22

//Useful for checking return values of cuda API calls
#define gpu_error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
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

__global__ void gen_disp_histogram(curandState *state, unsigned int *local_hist, float* global_hist, int samples, int bins, float lbe, float d_eps) {
  int g_id = get_global_id();
  int l_id = get_local_id();

  //copy state from global memory into local memory for faster access
  curandState local_state = state[g_id];

  //calculate where this thread blocks histogram starts
  unsigned int *l_hist = local_hist + blockIdx.x * bins;

  float kx, ky, kz;
  unsigned int bin_ind;

  for(int s = 0; s < samples; s++) {
    kx = curand_uniform(&local_state) * 2.0f * M_PI;
    ky = curand_uniform(&local_state) * 2.0f * M_PI;
    kz = curand_uniform(&local_state) * 2.0f * M_PI;

    bin_ind = (unsigned int) floor(((calc_dispersion(kx, ky, kz) - lbe)/d_eps));

    //Stops collisions when adding to block histogram
    atomicAdd(&l_hist[bin_ind],1);
  }

  //Make sure that all the threads have finished sampling before we
  //add up the global histogram
  __syncthreads();


  //put the new state back in global memory incase we use it again
  //this stops us generateing the same seqence if we re run the kernel
  state[g_id] = local_state;

  //Now we are done generateing the local histogram we can fold it into
  //one global histogram from all the blocks.
  //We will need to be careful here of integer and float overflow
  //if we are using alot of samples per thread.
  //minimise this by adding up the bin count times d_eps (small number)
  //not the total bin count as an int
  int g_bin_id = l_id;

  while (g_bin_id < bins) {

    float val = (float)l_hist[g_bin_id] / (float)(samples * (float) THREADS_PER_BLOCK);
    atomicAdd(&global_hist[g_bin_id],val);

    //keep shifting the block untill we have added all the bins
    g_bin_id += blockDim.x;
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

//Main control flow of code
int main(int argc, char* argv[]) {

  // Set up the program options
  namespace po = boost::program_options;
  po::options_description description("Options");
  description.add_options()
    ("o_file,o", po::value<std::string>()->default_value("dos.dat"), "output file name")
    ("n_sample,n", po::value<int>()->default_value(2000000), "number of samples")
    ("num_vals,N", po::value<int>()->default_value(200000), "number of points in the epsilon grid");

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
  std::vector<unsigned int> rng_seeds(NUM_BLOCKS * THREADS_PER_BLOCK);

  //initialise cuda stuff
  curandState *d_states;
  unsigned int *d_local_hist;
  float *d_global_hist;
  unsigned int *d_seeds;

  //Generate the seeds
  std::cout << "Generating Seeds" << std::endl;
  for (int i = 0; i<rng_seeds.size(); i++) rng_seeds[i] = good_seed(i);

  //allocate states and seeds and copy the seeds to the device
  gpu_error_check(cudaMalloc((void **)&d_states,  NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(curandState)));
  gpu_error_check(cudaMalloc(&d_seeds, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned int)));
  gpu_error_check(cudaMemcpy(rng_seeds.data(), d_seeds, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  //Print out lauch params
  std::cout << "launching " << NUM_BLOCKS << " blocks of " << THREADS_PER_BLOCK << " threads" << std::endl;

  //initialise the rng states on the gpu
  set_up_rngs <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (d_states, d_seeds);
  gpu_error_check(cudaPeekAtLastError());

  //Free the seeds on the device
  gpu_error_check(cudaFree(d_seeds));

  //Allocate memory on the device for local and global histograms
  gpu_error_check(cudaMalloc(&d_local_hist, NUM_BLOCKS * N * sizeof(unsigned int)));
  gpu_error_check(cudaMalloc(&d_global_hist, N * sizeof(float)));

  //Set the histograms to zero to start
  gpu_error_check(cudaMemset(d_local_hist, (unsigned int) 0, NUM_BLOCKS * N * sizeof(unsigned int)));
  gpu_error_check(cudaMemset(d_global_hist, (float) 0, N * sizeof(unsigned int)));

  //generate the histograms
  std::cout << "Generating histogram" << std::endl;
  gen_disp_histogram <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (d_states, d_local_hist, d_global_hist, n, N, lower_band_edge, d_eps);
  gpu_error_check(cudaDeviceSynchronize());
  gpu_error_check(cudaPeekAtLastError());

  //Copy then results back
  gpu_error_check(cudaMemcpy(h_hist.data(), d_global_hist, N * sizeof(float), cudaMemcpyDeviceToHost));

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
