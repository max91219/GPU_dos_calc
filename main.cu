/*
 ============================================================================
 Name        : gpu_dos_calc.cu
 Author      : Max Howell
 Version     :
 Copyright   : 
 Description : CUDA computation of non interacting density of states using
 	 	 	   Monte Carlo.
 ============================================================================
 */

#include<boost/program_options.hpp>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>

//CUDA includes
#include <stdio.h>
#include <assert.h>
#include<curand_kernel.h>
#include <math_constants.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include "device_dos_calc.cuh"

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


int main(int argc, char* argv[]) {

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

  double upper_band_edge = 3.5f;
  double lower_band_edge = -13.5f;
  double d_eps = (upper_band_edge - lower_band_edge) / N;

  std::cout << "num_vals: " << N << std::endl;
  std::cout << "n_sample: " << n << std::endl;
  std::cout << "o_file: " << o_file << std::endl;
  std::cout << "d_eps: " << d_eps << std::endl;
  std::cout << "1/d_eps: " << 1.0/d_eps << std::endl;
  std::cout << "Total number of samples: " << n*(float)(NUM_BLOCKS * THREADS_PER_BLOCK) << std::endl;

  std::vector<unsigned long long int> h_hist(N);
  unsigned int seed = good_seed(time(NULL));

  //initialise CUDA stuff
  curandStateMtgp32 *d_states;
  mtgp32_kernel_params *d_kernel_params;
  unsigned long long int *d_local_hist;
  unsigned long long int *d_global_hist;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //allocate device memory
  gpu_error_check(cudaMalloc((void **)&d_states,  NUM_BLOCKS * sizeof(curandStateMtgp32)));
  gpu_error_check(cudaMalloc((void **)&d_kernel_params, sizeof(mtgp32_kernel_params)));
  gpu_error_check(cudaMalloc(&d_local_hist, NUM_BLOCKS * N * sizeof(unsigned long long int)));
  gpu_error_check(cudaMalloc(&d_global_hist, N * sizeof(unsigned long long int)));

  std::cout << "Setting up RNG states on the device" << std::endl;
  CURAND_error_check(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, d_kernel_params));
  CURAND_error_check(curandMakeMTGP32KernelState(d_states, mtgp32dc_params_fast_11213, d_kernel_params, NUM_BLOCKS, seed));

  std::cout << "launching " << NUM_BLOCKS << " blocks of " << THREADS_PER_BLOCK << " threads" << std::endl;

  //Set the histograms to zero to start
  gpu_error_check(cudaMemset(d_local_hist, 0, NUM_BLOCKS * N * sizeof(unsigned long long int)));
  gpu_error_check(cudaMemset(d_global_hist, 0, N * sizeof(unsigned long long int)));

  //generate the histograms
  std::cout << "Generating Density of States" << std::endl;
  cudaEventRecord(start);
  gen_disp_histogram <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (d_states, d_local_hist, d_global_hist, n, N, lower_band_edge, upper_band_edge, d_eps);
  gpu_error_check(cudaDeviceSynchronize());
  gpu_error_check(cudaPeekAtLastError());
  cudaEventRecord(stop);

  //Copy then results back
  gpu_error_check(cudaMemcpy(h_hist.data(), d_global_hist, N * sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

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
      ofile << (i*d_eps)+lower_band_edge+d_eps << " " << (double)h_hist[i]/((double)NUM_BLOCKS * (double)THREADS_PER_BLOCK * (double)n * d_eps) << std::endl;
    }

    ofile.close();
  }

  return 0;
}

