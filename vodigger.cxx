#include <iostream>
using std::endl;

#include <iterator>
#include <memory>
#include <string>
using std::string;

#include <glog/logging.h>

#include <boost/program_options.hpp>
namespace bpo = boost::program_options;
#include <boost/property_tree/ptree.hpp>
namespace bpt = boost::property_tree;

#include "utils/config.hxx"
#include "inputs/source_factory.hxx"
#include "inputs/feeder_factory.hxx"
#include "solvers/solver.hxx"
#include "solvers/solver_caffe.hxx"
using namespace vodigger;


int main(int argc, const char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    std::ostream_iterator<string> newlineit {std::cout, "\n"};

    // parse command line arguments
    bpo::variables_map args = parse_cmd_args(argc, argv);
    if(args.empty()) return 0;

    // load the source (folder.archive ...)
    std::shared_ptr<Source> source {source_factory(args["source"].as<string>())};
    CHECK(source) << "Cannot find any source handler for the path specified";

    // obtain a config file
    std::unique_ptr<std::istream> iconf {source->read(args["config"].as<string>())};
    CHECK(iconf) << "Config file " << args["config"].as<string>() << " wasn't found";

    // parse the config file
    bpt::ptree conf = parse_config_file(iconf.get());
    CHECK(!conf.empty()) << "Error parsing the config file";

    // feeder will try to load data into memory or create/get caffe-readable database
    std::shared_ptr<Feeder> feeder {feeder_factory(source, conf)};
    CHECK(feeder) << "No feeder can hadle type: " << conf.get<std::string>("data.type");

    // if the user selected --cretedb <path> then just create database ?and quit?
    if(args.count("createdb") > 0) {
        feeder->create_dbs(args["createdb"].as<std::string>());
        return 0;
    }

    std::unique_ptr<Solver> solver {new SolverCaffe(conf, feeder, source)};
    solver->run(std::cout);

    return 0;
}
