#include<boost/program_options.hpp>
#include<iostream>
#include<fstream>
#include<vector>

#include<curand_kernel.h>

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
  return blockIdx.x *blockDim.x + threadIdx.x;
}

__global__ void set_up_rngs(curandState *state) {
  int id = get_global_id();

  curand_init(1337, id, 0, &state[id]);
}

__global__ void gen_uniform(curandState *state, float *results) {
  int id = get_global_id();

  curandState local_state = state[id];
  results[id] = curand_uniform(&local_state);

  state[id] = local_state;
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

  std::vector<float> results_h(NUM_BLOCKS * THREADS_PER_BLOCK);

  //initialise cuda stuff
  curandState *dev_states;
  float *res_d;

  //Allocate memory on the device
  gpu_error_check(cudaMalloc((void **)&dev_states,  NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(curandState)));
  gpu_error_check(cudaMalloc((void **)&res_d,  NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));

  //Print out lauch params
  std::cout << "launching " << NUM_BLOCKS << " of " << THREADS_PER_BLOCK << " threads" << std::endl;

  //initialise the rng states on the gpu
  set_up_rngs <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (dev_states);
  gpu_error_check(cudaPeekAtLastError());

  //generate random numbers
  gen_uniform <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (dev_states, res_d);
  gpu_error_check(cudaPeekAtLastError());

  //copy results back
  gpu_error_check(cudaMemcpy(results_h.data(), res_d, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float), cudaMemcpyDeviceToHost));

  for (float f : results_h) std::cout << f << std::endl;

  //free device memory
  gpu_error_check(cudaFree(dev_states));
  gpu_error_check(cudaFree(res_d));

  // std::ofstream ofile;
  // ofile.open(o_file.c_str());
  // if (ofile.is_open()){

  //   float val;

  //   for (int i = 0; i < dos_res.size(); i++){
  //     val = dos_res[i];
  //     val *= 1.0f/d_eps;
  //     val /= n;
  //     ofile << (i*d_eps)+lower_band_edge << " " << val << std::endl;
  //   }

  //   ofile.close();
  // }


  return 0;
}
