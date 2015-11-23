#include <iomanip>
#include <iostream>
using std::endl;

#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <random>
#include <map>
#include <string>
using std::string;
using std::pair;

#include <glog/logging.h>

#include <boost/smart_ptr.hpp>
#include "boost/algorithm/string.hpp"
#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;
#include <boost/program_options.hpp>
namespace bpo = boost::program_options;
#include <boost/property_tree/ptree.hpp>
namespace bpt = boost::property_tree;

#include "caffe/caffe.hpp"
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;

#include "utils/config.hxx"
#include "inputs/source_factory.hxx"

using namespace vodigger;



// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void copy_layers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}


// Train / Finetune a model.
int train(const bpo::variables_map& args, const bpt::ptree& conf, std::shared_ptr<Source> source)
{
    CHECK(!conf.get<std::string>("params.model", "").empty())
        << "Need a model definition to train.";

    caffe::SolverParameter solver_param;
    caffe::NetParameter *train_net_param = new caffe::NetParameter(), *test_net_param = nullptr;

    solver_param_from_config(solver_param, conf);
    net_param_from_config_and_model(train_net_param, Phase::TRAIN,
                                    conf, source->read(conf.get<std::string>("params.model")));
    solver_param.set_allocated_train_net_param(train_net_param);

    if(!conf.get<std::string>("data.test.file", "").empty()) {
        // we have testing data so we can build a testing network
        test_net_param = solver_param.add_test_net_param();
        net_param_from_config_and_model(test_net_param, Phase::TEST,
                                        conf, source->read(conf.get<std::string>("params.model")));
    }

    boost::shared_ptr<caffe::Solver<float> > solver(
        caffe::SolverRegistry<float>::CreateSolver(solver_param));

    LOG(INFO) << "Starting Optimization";
    if (args.count("snapshot") > 0)
    {
        LOG(INFO) << "Resuming from " << args["snapshot"].as<std::string>();
        solver->Solve(args["snapshot"].as<std::string>());
    }
    else
    {
        std::string weights_file;
        if (args.count("weights") > 0) {
            weights_file = args["weights"].as<std::string>();
            copy_layers(&*solver, weights_file);
        }
        solver->Solve();
    }
    LOG(INFO) << "Optimization Done.";
    return 0;
}


int confusion_max_key(const std::map<std::pair<int, int>, int>& conf)
{
    int maxval = 0;
    for(auto it = conf.cbegin(); it != conf.cend(); ++it) {
        if(it->first.first > maxval) maxval = it->first.first;
        if(it->first.second > maxval) maxval = it->first.second;
    }
    return maxval;
}


// Test: score a model.
void test(const bpo::variables_map& args, const bpt::ptree& conf, std::shared_ptr<Source> source,
          std::ostream& results)
{
    CHECK(!conf.get<std::string>("params.model", "").empty())
        << "Need a model definition to score (config params.model)";

    CHECK(args.count("test") > 0 ||
          !conf.get<std::string>("params.weights", "").empty())
        << "Need model weights to score (config params.weights or argument --weights)";

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    // Instantiate the caffe net.
    std::string weights_file;
    if(args.count("test") > 0) {
        weights_file = args["test"].as<std::string>();
    }
    if(weights_file.empty() && !conf.get<std::string>("params.weights", "").empty()) {
        weights_file = conf.get<std::string>("params.weights");
    }

    caffe::NetParameter net_param;
    net_param_from_config_and_model(&net_param, Phase::TEST, conf,
                                    source->read(conf.get<std::string>("params.model")));
    Net<float> net {net_param};
    net.CopyTrainedLayersFrom(source->absolutize(weights_file));

    int iterations = conf.get("params.test.iter", 50);
    LOG(INFO) << "Running for " << iterations << " iterations (adjust by config params.test.iter).";
    std::vector<Blob<float>* > bottom_vec;
    std::vector<int> test_score_output_id;
    std::vector<float> test_score;
    std::map<pair<int,int>, int> confusion;
    float loss = 0;
    float chance_of_printing = 1.0/iterations;
    int print_samples = 30;

    for (int iter = 0; iter < iterations; ++iter) {
        float iter_loss;
        const std::vector<Blob<float>*>& result =
                net.Forward(bottom_vec, &iter_loss);
        const float* labels = net.blob_by_name("labels")->cpu_data();
        loss += iter_loss;
        int idx = 0;
        for (int j = 0; j < result.size(); ++j) {
            const std::string& output_name = net.blob_names()[
                    net.output_blob_indices()[j]];
            const float* result_vec = result[j]->cpu_data();
            if(result[j]->count() > 2)
            {
                int outer_num = result[j]->count(0, 1);
                int channels = result[j]->count(1, 2);
                int inner_num = result[j]->count(2);
                int printed_results = 0;
                // means we are getting the guesses to the labels
                for(int onum = 0; onum < outer_num; ++onum)
                {
                    // print out just few random samples ... don't flood us with data
                    if(printed_results < int(print_samples*chance_of_printing) && distribution(generator) < chance_of_printing)
                    {
                        std::ostringstream oss;
                        oss << int(labels[onum]);
                        oss << std::setprecision(1) << "[";
                        for (int inum = 0; inum < inner_num; ++inum) {
                            oss << result_vec[onum*channels*inner_num+inum] << ", ";
                        }
                        oss << "\b\b] prob: [";
                        if(channels > 1) {
                            for (int inum = 0; inum < inner_num; ++inum) {
                                oss << result_vec[onum*channels*inner_num+inum+inner_num] << ", ";
                            }
                        }
                        oss << "\b\b]";
                        LOG(INFO) << "Example from (" << output_name << " layer): " << oss.str();
                        ++printed_results;
                    }
                    // compute confusion matrix
                    confusion[std::make_pair(int(labels[onum]), int(result_vec[onum*channels*inner_num]))] += 1;
                }
            }
            else
            {
                for (int k = 0; k < result[j]->count(); ++k, ++idx) {
                    const float score = result_vec[k];
                    if (iter == 0) {
                        test_score.push_back(score);
                        test_score_output_id.push_back(j);
                    } else {
                        test_score[idx] += score;
                    }
                }
            }
        }
    }
    loss /= iterations;
    LOG(INFO) << "Loss: " << loss;
    for (int i = 0; i < test_score.size(); ++i) {
        const std::string& output_name = net.blob_names()[
                net.output_blob_indices()[test_score_output_id[i]]];
        const float loss_weight =
                net.blob_loss_weights()[net.output_blob_indices()[i]];
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / iterations;
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                                            << " = " << loss_weight * mean_score << " loss)";
        }
        LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
    }
    // print out confusion matrix
    LOG(INFO) << "Confusion matrix size: " << confusion.size();
    std::ostringstream oss;
    int axis = confusion_max_key(confusion);
    for(int i = 0; i <= axis; ++i) {
        for(int j = 0; j <= axis; ++j) {
            oss << std::setw(4) << confusion[std::make_pair(i, j)];
        }
        oss << std::endl;
    }
    LOG(INFO) << std::endl << oss.str();
}


// Time: benchmark the execution time of a model.
int time(const bpo::variables_map& args, const bpt::ptree& conf, std::shared_ptr<Source> source)
{
    CHECK(!conf.get<std::string>("params.model", "").empty())
        << "Need a model definition to score (config params.model)";

    if(conf.get("params.benchmark.iter", -1) == -1)
        LOG(WARNING) << "Set params.benchmark.iter in config file. Using default 50";

    // Instantiate the caffe net.
    Net<float> caffe_net(source->absolutize(conf.get<std::string>("params.model")),
                         caffe::TRAIN);

    // Do a clean forward and backward pass, so that memory allocation are done
    // and future iterations will be more stable.
    LOG(INFO) << "Performing Forward";
    // Note that for the speed benchmark, we will assume that the network does
    // not take any input blobs.
    float initial_loss;
    caffe_net.Forward(std::vector<Blob<float>*>(), &initial_loss);
    LOG(INFO) << "Initial loss: " << initial_loss;
    LOG(INFO) << "Performing Backward";
    caffe_net.Backward();

    const std::vector<boost::shared_ptr<Layer<float> > >& layers = caffe_net.layers();
    const std::vector<std::vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
    const std::vector<std::vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
    const std::vector<std::vector<bool> >& bottom_need_backward =
            caffe_net.bottom_need_backward();
    int iterations = conf.get("params.benchmark.iter", 50);

    LOG(INFO) << "*** Benchmark begins ***";
    LOG(INFO) << "Testing for " << iterations << " iterations.";
    caffe::Timer total_timer;
    total_timer.Start();
    caffe::Timer forward_timer;
    caffe::Timer backward_timer;
    caffe::Timer timer;
    std::vector<double> forward_time_per_layer(layers.size(), 0.0);
    std::vector<double> backward_time_per_layer(layers.size(), 0.0);
    double forward_time = 0.0;
    double backward_time = 0.0;
    for (int j = 0; j < iterations; ++j) {
        caffe::Timer iter_timer;
        iter_timer.Start();
        forward_timer.Start();
        for (int i = 0; i < layers.size(); ++i) {
            timer.Start();
            // Although Reshape should be essentially free, we include it here
            // so that we will notice Reshape performance bugs.
            layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
            layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
            forward_time_per_layer[i] += timer.MicroSeconds();
        }
        forward_time += forward_timer.MicroSeconds();
        backward_timer.Start();
        for (int i = layers.size() - 1; i >= 0; --i) {
            timer.Start();
            layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                                                    bottom_vecs[i]);
            backward_time_per_layer[i] += timer.MicroSeconds();
        }
        backward_time += backward_timer.MicroSeconds();
        LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
            << iter_timer.MilliSeconds() << " ms.";
    }
    LOG(INFO) << "Average time per layer: ";
    for (int i = 0; i < layers.size(); ++i) {
        const caffe::string& layername = layers[i]->layer_param().name();
        LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
            "\tforward: " << forward_time_per_layer[i] / 1000 /
            iterations << " ms.";
        LOG(INFO) << std::setfill(' ') << std::setw(10) << layername    <<
            "\tbackward: " << backward_time_per_layer[i] / 1000 /
            iterations << " ms.";
    }
    total_timer.Stop();
    LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
        iterations << " ms.";
    LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
        iterations << " ms.";
    LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
        iterations << " ms.";
    LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
    LOG(INFO) << "*** Benchmark ends ***";
    return static_cast<int>(total_timer.MilliSeconds());
}


void dump(const bpo::variables_map& args, const bpt::ptree& conf, std::shared_ptr<Source> source)
{
    CHECK(!conf.get<std::string>("params.model", "").empty())
        << "Need a model definition to score (config params.model)";

    CHECK(args.count("dump") > 0)
        << "Need model weights to dump (argument --weights)";

    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);

    // Instantiate the caffe net.
    std::string weights_file = args["dump"].as<std::string>();

    Net<float> caffe_net(conf.get<std::string>("params.model"), caffe::TRAIN);
    caffe_net.CopyTrainedLayersFrom(weights_file);

    const std::vector<std::string> names = caffe_net.layer_names();
    const std::vector<boost::shared_ptr<Blob<float> > > params = caffe_net.params();
    const std::vector<int> param_owners = caffe_net.param_owners();

    for(int i = 0; i < params.size(); ++i)
    {
        std::ostringstream iss;
        iss << "Owner: " << param_owners[i] << " shape: " << params[i]->shape_string() << std::endl;
        int outer_num = params[i]->count(0,1);
        int inner_num = params[i]->count(1);
        const float *data = params[i]->cpu_data();
        for(int n=0; n<outer_num; ++n) {
            for(int j=0; j<inner_num;++j) {
                iss << data[n*inner_num + j] << " ";
            }
            iss << std::endl;
        }
        LOG(INFO) << iss.str();
    }
}


void guess(const bpo::variables_map& args, const bpt::ptree& conf, std::shared_ptr<Source> source,
           std::ostream& results)
{
    std::istream *src = source->read(conf.get<std::string>("data.guess.file"));
    std::vector<Blob<float>* > empty_bottom_vec;
    char buff[255];
    float id = 0;
    const char separator = conf.get<std::string>("data.guess.separator",
                                      conf.get<std::string>("data.separator", ",")).c_str()[0];
    const char newline = conf.get<std::string>("data.guess.newline",
                                    conf.get<std::string>("data.newline", "\n")).c_str()[0];
    const int header = conf.get<int>("data.guess.header", 0);
    const int id_col = conf.get<int>("data.guess.id", conf.get<int>("data.guess.ref", 0));
    for(int i=0; i < header; ++i) src->ignore(2147483647, newline);

    // find a file holding trained network
    std::string weights_file;
    if(args.count("guess") > 0) weights_file = args["guess"].as<std::string>();
    if(weights_file.empty() && !conf.get<std::string>("params.weights", "").empty()) {
        weights_file = conf.get<std::string>("params.weights");
    }
    // we will construct only one network in TRAIN state
    caffe::NetParameter net_param;
    net_param_from_config_and_model(&net_param, Phase::GUESS,
                                    conf, source->read(conf.get<std::string>("params.model")));
    // Instantiate the caffe net
    Net<float> net {net_param};
    net.CopyTrainedLayersFrom(source->absolutize(weights_file));
    // run until IDs reset
    while(true) {
        size_t i = 0;
        // get results from one forward pass (nullptr is 'loss' value - we don't compute it now)
        net.Forward(empty_bottom_vec);
        const float *guess = net.blob_by_name("result")->cpu_data();
        const size_t guesses = net.blob_by_name("result")->count(0,1);
        const float *ids = net.blob_by_name("ids")->cpu_data();
        for(; i < guesses; ++i) {
           if(ids[i] < id) break;               // if we loop in data, break
           for(int c=0; c<id_col;++c) src->ignore(2147483647, separator);  // find position of ID/REF
           src->getline(buff, 255, separator);  // obtain REF; we expect it to be in the first column
           if(!src->good()) break;
           src->ignore(2147483647, newline);    // skip the rest of row
           results << buff << separator << guess[2*i] << separator << guess[2*i+1] << newline;
           id = ids[i];
        }
        if(i < guesses || !src->good()) break;
        results.flush();
    }
    delete src;
}


int main(int argc, const char *argv[])
{
    bfs::path cwd;     // in case of different SOURCE than CWD this saves original CWD
    std::string cwds;  // holds the string of CWD (either from SOURCE or command CWD)

    // parse command line arguments
    bpo::variables_map args = parse_cmd_args(argc, argv);
    if(args.empty()) return 0;

    if(args.count("source") > 0) {
        cwds = args["source"].as<string>();
    } else {
        cwds = bfs::current_path().string();
        FLAGS_logtostderr = 1;   // in case of not specified source - print LOG to stderr
    }

    // init logging and output
    google::InitGoogleLogging(argv[0]);

    // load the source (folder.archive ...)
    std::shared_ptr<Source> source {source_factory(cwds)};
    CHECK(source) << "Cannot find any source handler for the path specified";

    // obtain a config file
    std::unique_ptr<std::istream> iconf {source->read(args["config"].as<string>())};
    CHECK(iconf->good()) << "Config file " << args["config"].as<string>() << " wasn't found";

    // parse the config file
    bpt::ptree conf = parse_config_file(iconf.get());
    CHECK(!conf.empty()) << "Error parsing the config file";

    int num_params = args.count("train") +
                     args.count("test") +
                     args.count("time") +
                     args.count("dump") +
                     args.count("guess");

    CHECK_GE(num_params, 1) << "Specify either --test, --train, --time, --dump or --guess";

    // if the user selected --cretedb <path> then just create database ?and quit?
    // if(args.count("createdb") > 0) {
    //     // feeder will try to load data into memory or create/get caffe-readable database
    //     std::shared_ptr<Feeder> feeder {feeder_factory(source, conf)};
    //     CHECK(feeder) << "No feeder can hadle type: " << conf.get<std::string>("data.type");
    //     feeder->create_dbs(args["createdb"].as<std::string>());
    //     return 0;
    // }

    // if we have source specified then change CWD to the path in the argumentd

    if(args.count("source") > 0) {
        cwd = bfs::current_path();
        bfs::current_path(bfs::path(args["source"].as<string>()));
    }

    // Set device id and mode
    // If the gpu flag is not provided, allow the mode and device to be set
    // in the solver prototxt.
    if (conf.get<std::string>("params.mode", "CPU") == "GPU")
    {
        LOG(INFO) << "Dynamic decision about device ID " << 0;    // just a joke
        Caffe::SetDevice(0);
        Caffe::set_mode(Caffe::GPU);
    } else {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    }

    if(args.count("train") >= 1) {
        train(args, conf, source);
    }
    if(args.count("test") >= 1) {
        test(args, conf, source, std::cout);
    }
    if(args.count("time") >= 1) {
        time(args, conf, source);
    }
    if(args.count("dump") >= 1) {
        dump(args, conf, source);
    }
    if(args.count("guess") >= 1) {
        std::ostream *results = source->write(conf.get("output.guess", "output.csv"));
        guess(args, conf, source, *results);
        // dynamic_cast<std::ofstream*>(results)->close();
        delete results;
    }

    // if source specified change CWD back to the original
    if(args.count("source") > 0){
        bfs::current_path(cwd);
    }
    return 0;
}
