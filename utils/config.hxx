#ifndef VODIGGER_CONFIG

#define VODIGGER_CONFIG


#include <iostream>
#include <string>
#include <boost/program_options.hpp>
namespace bpo = boost::program_options;

#include <boost/property_tree/ptree.hpp>
namespace bpt = boost::property_tree;


namespace vodigger {

	// decides if the current combination of arguments is valid and good to go
	bool is_valid_args(boost::program_options::variables_map&) noexcept;

	// Function parsing parameters from commandline an handling --help option.
	boost::program_options::variables_map parse_cmd_args(int, const char**) noexcept;

	// Function to parse a JSON config file
	boost::property_tree::ptree parse_config_file(std::istream*) noexcept;

}

#endif
