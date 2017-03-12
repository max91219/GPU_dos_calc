#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <boost/program_options.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <curand_kernel.h>
#include <thrust/extrema.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iterator>
#include <cmath>


struct gen_rand : public thrust::unary_function<unsigned int, float> {

  __device__
  float operator()(unsigned int thread_id) {

    unsigned int seed = thread_id;
    curandState s;
    curand_init(seed, 0, 0, &s);

    return  curand_uniform(&s)* 2.0f * M_PI;
  }

};


struct calc_disp_bins {

  float lower_band_edge;
  float upper_band_edge;
  int n_bins;
  float d_eps;

  __host__ __device__
  calc_disp_bins(float _upper_band_edge = 3.5f, float _lower_band_edge = -13.5f, int _n_bins = 200000) :
    upper_band_edge(_upper_band_edge), lower_band_edge(_lower_band_edge), n_bins(_n_bins) {

    d_eps = (upper_band_edge - lower_band_edge) / (float) n_bins;
  };


  template <typename Tuple>
  __device__
  void operator()(Tuple t) {

    float eps = -4.0f * ((std::cos(thrust::get<0>(t)/2.0f)*std::cos(thrust::get<1>(t)/2.0f)) +
                         (std::cos(thrust::get<2>(t)/2.0f)*std::cos(thrust::get<1>(t)/2.0f)) +
                         (std::cos(thrust::get<0>(t)/2.0f)*std::cos(thrust::get<2>(t)/2.0f))) -
                2.0 * 0.25 * (std::cos(thrust::get<0>(t)) +
                              std::cos (thrust::get<1>(t)) +
                              std::cos(thrust::get<2>(t)));

    thrust::get<3>(t) = floor((eps - lower_band_edge) / d_eps);
  }
};


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

  thrust::device_vector<float> k_x(n);
  thrust::device_vector<float> k_y(n);
  thrust::device_vector<float> k_z(n);
  thrust::device_vector<float> disp_bins(n);
  thrust::device_vector<float> dos(N);
  thrust::host_vector<float> dos_res(N);

  thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(n),
                    k_x.begin(),
                    gen_rand());

  thrust::transform(
                    thrust::make_counting_iterator(n+1),
                    thrust::make_counting_iterator(2*n+1),
                    k_y.begin(),
                    gen_rand());

  thrust::transform(
                    thrust::make_counting_iterator(2*n+2),
                    thrust::make_counting_iterator(3*n+2),
                    k_z.begin(),
                    gen_rand());

  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(k_x.begin(), k_y.begin(), k_z.begin(), disp_bins.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(k_x.end(),   k_y.end(),   k_z.end(),   disp_bins.end())),
                   calc_disp_bins(upper_band_edge, lower_band_edge, N));

  thrust::sort(disp_bins.begin(), disp_bins.end());

  thrust::device_vector<float>::iterator max_iter;
  thrust::device_vector<float>::iterator min_iter;

  max_iter = thrust::max_element(k_x.begin(), k_x.end());
  min_iter = thrust::min_element(k_x.begin(), k_x.end());
  std::cout << "max k_x: " << *max_iter << " min k_x: " << *min_iter << std::endl;

  max_iter = thrust::max_element(k_y.begin(), k_y.end());
  min_iter = thrust::min_element(k_y.begin(), k_y.end());
  std::cout << "max k_y: " << *max_iter << " min k_y: " << *min_iter << std::endl;

  max_iter = thrust::max_element(k_z.begin(), k_z.end());
  min_iter = thrust::min_element(k_z.begin(), k_z.end());
  std::cout << "max k_z: " << *max_iter << " min k_z: " << *min_iter << std::endl;

  max_iter = thrust::max_element(disp_bins.begin(), disp_bins.end());
  min_iter = thrust::min_element(disp_bins.begin(), disp_bins.end());
  std::cout << "max disp: " << *max_iter << " min disp: " << *min_iter << std::endl;

  thrust::upper_bound(disp_bins.begin(),
                      disp_bins.end(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(N),
                      dos.begin());

  thrust::adjacent_difference(dos.begin(),
                              dos.end(),
                              dos.begin());

  dos_res = dos;

  std::ofstream ofile;
  ofile.open(o_file.c_str());

  if (ofile.is_open()){

    float val;

    for (int i = 0; i < dos_res.size(); i++){
      val = dos_res[i];
      val *= 1.0f/d_eps;
      val /= n;
      ofile << (i*d_eps)+lower_band_edge << " " << val << std::endl;
    }

    ofile.close();
  }

  // print sorted array
  //thrust::copy(k_x.begin(), k_x.end(), std::ostream_iterator<float>(std::cout, "\n"));
  //std::cout << std::endl;
  //thrust::copy(k_y.begin(), k_y.end(), std::ostream_iterator<float>(std::cout, "\n"));
  //std::cout << std::endl;
  //thrust::copy(k_z.begin(), k_z.end(), std::ostream_iterator<float>(std::cout, "\n"));
  //std::cout << std::endl;
  //thrust::copy(disp_bins.begin(), disp_bins.end(), std::ostream_iterator<float>(std::cout, "\n"));
  //std::cout << std::endl;
  //thrust::copy(dos.begin(), dos.end(), std::ostream_iterator<float>(std::cout, "\n"));


  return 0;
}
