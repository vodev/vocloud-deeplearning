#include "../version.hxx"
#include "config.hxx"

#include <glog/logging.h>

#include <boost/property_tree/json_parser.hpp>
namespace bjson = boost::property_tree::json_parser;


namespace vodigger {


bool is_valid_args(bpo::variables_map& args) noexcept {
    return args.count("source") > 0 &&
           args.count("help") == 0;
}


void print_help_args(bpo::options_description& cmd_opts, std::string title) noexcept {
    if(!title.empty()) std::cout << title << std::endl;
    std::cout << "usage: vodigger [options] SOURCE" << std::endl << std::endl;
    std::cout << cmd_opts << "\n";
}


bool has_integrity(const boost::property_tree::ptree& params)
{
    // save the state to a variable so all checks proceed at the same time
    bool result = true;
    // either saved model or data source to train a model, not both
    if(params.get<std::string>("parameters.model", "").empty())
    {
        LOG(ERROR) << "parameters.model is mandatory";
        return false;
    }
    return result;
}


boost::program_options::variables_map parse_cmd_args(int argc, const char** argv) noexcept
{
    // Define visible command line arguments
    bpo::options_description cmd_opts("Command line options");
    cmd_opts.add_options()
        ("help,h", "produce help message")
        ("createdb,d",  bpo::value<std::string>(), "path where to create leveldb from given (csv) source")
        ("config,c", bpo::value<std::string>()->default_value("config.json"), "name of a config file")
    ;

    // Hidden options only set types for positional arguments and should not be displayed in help
    bpo::options_description hid_opts("Hidden options");
    hid_opts.add_options()
        ("source", bpo::value<std::string>(), "source directory/archive with data and config file")
    ;

    // Declare positional arguments
    bpo::positional_options_description pos_opts;
    pos_opts.add("source", 1); // position arg "source" accepts only one parameter

    // Merge commandline_options and hidden_options for parsing
    bpo::options_description all_opts;
    all_opts.add(cmd_opts).add(hid_opts);

    bpo::variables_map args;
    try {
        bpo::store(bpo::command_line_parser(argc, argv).options(all_opts).positional(pos_opts).run(), args);
        bpo::notify(args);
        if (!is_valid_args(args)) {
           print_help_args(cmd_opts, std::string("VO-Digger version ") + VODIGGER_VERSION);
           return bpo::variables_map();  // return an empty set of arguments
        }
    }
    catch(const bpo::unknown_option& ex) {
        print_help_args(cmd_opts, ex.what());
        return bpo::variables_map();  // return an empty set of arguments
    }

    return args;
}


bpt::ptree parse_config_file(std::istream* iconf) noexcept{
    bpt::ptree pt;
    try {
        bjson::read_json(*iconf, pt);
    } catch (const bjson::json_parser_error& ex) {
        std::cerr << ex.what() << std::endl;
        return bpt::ptree();  // return an empty ptree
    }
    if(!has_integrity(pt)) {
        return bpt::ptree();
    }
    return pt;
}


// namespace vodigger end
}
