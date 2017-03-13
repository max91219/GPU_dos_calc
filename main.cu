#include <boost/program_options.hpp>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>

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
