#include "../version.hxx"
#include "config.hxx"

#include <glog/logging.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <boost/property_tree/json_parser.hpp>
namespace bjson = boost::property_tree::json_parser;


namespace vodigger {


bool is_valid_args(bpo::variables_map& args) noexcept
{
    return args.count("source") >= 0 && // source can be ommited ... we use CWD then
           args.count("help") == 0;
}


void print_help_args(bpo::options_description& cmd_opts, std::string title) noexcept
{
    if(!title.empty()) std::cout << title << std::endl;
    std::cout << "usage: vodigger [options] <SOURCE>" << std::endl << std::endl;
    std::cout << cmd_opts << "\n";
}


bool has_integrity(const boost::property_tree::ptree& params) noexcept
{
    // save the state to a variable so all checks proceed at the same time
    bool result = true;
    // either saved model or data source to train a model, not both
    if(params.get<std::string>("params.model", "").empty())
    {
        LOG(ERROR) << "params.model is mandatory";
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
        ("train", "train network specified by solver, model and config file")
        ("test", bpo::value<std::string>(), "test network with specified weight file (can be relative to the source dir)")
        ("dump", bpo::value<std::string>(), "dump parameters of network specified by weight file (can be relative to the source dir)")
        ("guess", bpo::value<std::string>(), "use pretrained network to classify unknown data (can be relative to the source dir)")
        ("time", "benchmark network by performing params.benchmark.iter forward and backward passes")
        ("weights,w", bpo::value<std::string>(), "snapshot of model")
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


bpt::ptree parse_config_file(std::istream* iconf) noexcept
{
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

void solver_param_from_config(caffe::SolverParameter& solver, const bpt::ptree& conf)
{
    std::string lr_policies[] = {"fixed", "step", "exp", "inv", "multistep", "poly", "sigmoid"};

    solver.set_type(conf.get("params.solver", "SGD"));
    solver.set_max_iter(conf.get("params.train.iter", 1000));
    solver.set_display(conf.get("params.train.display", int(solver.max_iter()/5)));
    if(!conf.get<std::string>("data.test", "").empty()) {
        solver.set_test_interval(conf.get("params.train.test_interval", int(solver.max_iter()/5)));
        solver.add_test_iter(conf.get("params.train.test_iter", 5));
    }
    solver.set_base_lr(conf.get("params.train.base_lr", 0.01));
    solver.set_lr_policy(conf.get("params.train.lr_policy", lr_policies[1]));
    solver.set_stepsize(conf.get("params.train.stepsize", int(solver.max_iter()/10)));
    solver.set_gamma(conf.get<float>("params.train.gamma",0.9));
    if(conf.get<float>("params.train.momentum", -1) != -1)
        solver.set_momentum(conf.get<float>("params.train.momentum",0.9));
    solver.set_snapshot_prefix(conf.get<std::string>("name", "model"));
    solver.set_snapshot_after_train(conf.get("params.train.snapshot_after_train", true));
    if (conf.get<std::string>("params.mode", "CPU") == "GPU")
        solver.set_solver_mode(::caffe::SolverParameter_SolverMode(1));
    else
        solver.set_solver_mode(::caffe::SolverParameter_SolverMode(0));
}


/**
* Creates a NetParameter message so either it can be passed to a solver or a Net can be constructed
* from it.
* It constructs TRAIN and TEST input layers (always BigDataLayer) and adds either accuracy
* layer for testing or ArgMax layer for classification (if there is only on class then we suppose
* a regression task and therefor print out just the value - no ArgMax layer added)
*/
void net_param_from_config_and_model(caffe::NetParameter *net, Phase phase,
                                     const bpt::ptree& conf, std::istream* pfile)
{
    std::string traindata = conf.get<std::string>("data.train.file", "");
    std::string testdata = conf.get<std::string>("data.test.file", "");
    std::string guessdata = conf.get<std::string>("data.guess.file", "");
    std::string prefix;

    caffe::NetParameter model;

    if(phase == Phase::TRAIN) {
        CHECK(!traindata.empty()) << "Specify training data";
        net->mutable_state()->set_phase(caffe::Phase::TRAIN);
        prefix = "data.train";
    }
    if(phase == Phase::TEST) {
        CHECK(!testdata.empty()) << "Specify testing data";
        net->mutable_state()->set_phase(caffe::Phase::TEST);
        prefix = "data.test";
    }
    if(phase == Phase::GUESS) {
        CHECK(!guessdata.empty()) << "Specify guessing data";
        net->mutable_state()->set_phase(caffe::Phase::TRAIN);
        prefix = "data.guess";
    }

    if(phase == Phase::TRAIN || phase == Phase::GUESS)
    {
        // init TRAIN input layer using BigDataLayer
        caffe::LayerParameter *input_layer = net->add_layer();
        input_layer->set_name("input");
        input_layer->set_type("BigData");
        caffe::NetStateRule *state = input_layer->add_include();
            state->set_phase((caffe::Phase::TRAIN));
        caffe::BigDataParameter *big_data_param = new caffe::BigDataParameter();
            big_data_param->set_chunk_size(conf.get<float>(prefix + ".chunk_size"));
            big_data_param->set_header(conf.get<int>(prefix+".header", 0));
            big_data_param->set_data_start(conf.get<int>(prefix+".start"));
            big_data_param->set_data_end(conf.get<int>(prefix+".end"));
            input_layer->add_top()->assign("data");
            // set label only for the TRAINing phase
            if(phase == Phase::TRAIN) {
                big_data_param->set_label(conf.get<int>(prefix+".label"));
                input_layer->add_top()->assign("labels");
                big_data_param->set_source(traindata);
            } else {
                big_data_param->set_source(guessdata);
                input_layer->add_top()->assign("ids");
            }
            if(!conf.get<std::string>("data.separator", "").empty())
                big_data_param->set_separator(conf.get<std::string>("data.separator"));
            if(!conf.get<std::string>("data.newline", "").empty())
                big_data_param->set_newline(conf.get<std::string>("data.newline"));
        input_layer->set_allocated_big_data_param(big_data_param);
    }

    if(phase == Phase::TEST)
    {
        // init TEST input layer using BigDataLayer
        caffe::LayerParameter *input_layer = net->add_layer();
        input_layer->set_name("test_input");
        input_layer->set_type("BigData");
        caffe::NetStateRule *state = input_layer->add_include();
            state->set_phase((caffe::Phase::TEST));
        caffe::BigDataParameter *big_data_param = new caffe::BigDataParameter();
            big_data_param->set_chunk_size(conf.get<float>(prefix+".chunk_size"));
            big_data_param->set_header(conf.get<int>(prefix+".header", 0));
            big_data_param->set_data_start(conf.get<int>(prefix+".start"));
            big_data_param->set_data_end(conf.get<int>(prefix+".end"));
        input_layer->add_top()->assign("data");
            big_data_param->set_label(conf.get<int>(prefix+".label"));
        input_layer->add_top()->assign("labels");
            if(!conf.get<std::string>("data.separator", "").empty())
                big_data_param->set_separator(conf.get<std::string>("data.separator"));
            if(!conf.get<std::string>("data.newline", "").empty())
                big_data_param->set_newline(conf.get<std::string>("data.newline"));
            big_data_param->set_source(testdata);
        input_layer->set_allocated_big_data_param(big_data_param);
    }

    // add whole model definition from a file
    if(pfile != nullptr)
    {
        google::protobuf::io::IstreamInputStream iis {pfile};
        google::protobuf::TextFormat::Merge(&iis, net);
    }
    if(pfile != nullptr) delete pfile;

    // append output layers necessary for statistics
    if(phase == Phase::TRAIN || phase == Phase::TEST)
    {
        // append necesary layer for training and testing -- a loss layer (if it isn't there already)
        bool has_loss = false;
        for(int i = 0; i < net->layer_size(); ++i)
            if(net->layer(i).type() == "SoftmaxWithLoss")
                has_loss = true;
        if(!has_loss)
        {
            caffe::LayerParameter *loss_layer = net->add_layer();
            loss_layer->set_name("loss");
            loss_layer->set_type("SoftmaxWithLoss");
            loss_layer->add_bottom()->assign("output");
            loss_layer->add_bottom()->assign("labels");
            loss_layer->add_top()->assign("loss");
        }
    }

    if(phase == Phase::TEST)
    {
        // append necesary layer for testing -- an accuracy layer (if it isn't there already)
        bool has_accuracy = false;
        for(int i = 0; i < net->layer_size(); ++i)
            if(net->layer(i).type() == "Accuracy")
                has_accuracy = true;
        if(!has_accuracy)
        {
            caffe::LayerParameter *accuracy_layer = net->add_layer();
            accuracy_layer->set_name("accuracy");
            accuracy_layer->set_type("Accuracy");
            accuracy_layer->add_bottom()->assign("output");
            accuracy_layer->add_bottom()->assign("labels");
            accuracy_layer->add_top()->assign("accuracy");
            caffe::NetStateRule *state = accuracy_layer->add_include();
                state->set_phase((caffe::Phase::TEST));
        }
    }

    if(phase == Phase::GUESS || phase == Phase::TEST)
    {
        // append ArgMax layer for ConfMatrix for testing and results reporting for guessing
        bool has_argmax = false;
        for(int i = 0; i < net->layer_size(); ++i)
            if(net->layer(i).type() == "ArgMax")
                has_argmax = true;
        if(!has_argmax)
        {
            caffe::LayerParameter *argmax_layer = net->add_layer();
            argmax_layer->set_name("argmax");
            argmax_layer->set_type("ArgMax");
            argmax_layer->add_bottom()->assign("output");
            argmax_layer->add_top()->assign("result");
            caffe::ArgMaxParameter *argmax_param = new caffe::ArgMaxParameter();
                argmax_param->set_out_max_val(true);
                argmax_param->set_top_k(1);
            argmax_layer->set_allocated_argmax_param(argmax_param);
            // include in both TRAIN (guess) and TEST phase
        }
    }
}

// namespace vodigger end
}
