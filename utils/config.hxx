#ifndef VODIGGER_CONFIG

#define VODIGGER_CONFIG


#include <iostream>
#include <string>
#include <memory>
#include <boost/program_options.hpp>
namespace bpo = boost::program_options;

#include <boost/property_tree/ptree.hpp>
namespace bpt = boost::property_tree;

#include <caffe/proto/caffe.pb.h>

#include "../vodigger.hxx"


namespace vodigger {

	// decides if the current combination of arguments is valid and good to go
	bool is_valid_args(boost::program_options::variables_map&) noexcept;

	// Function parsing parameters from commandline an handling --help option.
	boost::program_options::variables_map parse_cmd_args(int, const char**) noexcept;

	// Function to parse a JSON config file
	boost::property_tree::ptree parse_config_file(std::istream*) noexcept;

	void solver_param_from_config(caffe::SolverParameter&, const bpt::ptree&);

	void net_param_from_config_and_model(caffe::NetParameter*, Phase, const bpt::ptree&, std::istream*);
}

#endif
