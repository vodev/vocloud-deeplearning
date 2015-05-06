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

#include "utils/config.hxx"
#include "inputs/source_factory.hxx"
#include "inputs/feeder_factory.hxx"
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
    CHECK(!conf.get<std::string>("parameters.solver", "").empty())
        << "Need a solver definition to train.";

    // CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
    //         << "Give a snapshot to resume training or weights to finetune "
    //         "but not both.";


    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(
        source->absolutize(conf.get<std::string>("parameters.solver")), &solver_param);

    solver_param.set_net(source->absolutize(conf.get<std::string>("parameters.model")));
    if(solver_param.has_snapshot_prefix()) {
        solver_param.set_snapshot_prefix(source->absolutize(solver_param.snapshot_prefix()));
    }

    // If the gpu flag is not provided, allow the mode and device to be set
    // in the solver prototxt.
    if (conf.get<std::string>("parameters.mode", "CPU") == "GPU")
    {
        LOG(INFO) << "Dynamic decision about device ID " << 0;    // just a joke
        Caffe::SetDevice(0);
        Caffe::set_mode(Caffe::GPU);
    } else {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    }

    LOG(INFO) << "Initializing solver from parameters: " << std::endl
              << solver_param.DebugString();
    boost::shared_ptr<caffe::Solver<float> > solver(caffe::GetSolver<float>(solver_param));

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
int test(const bpo::variables_map& args, const bpt::ptree& conf, std::shared_ptr<Source> source)
{
    CHECK(!conf.get<std::string>("parameters.model", "").empty())
        << "Need a model definition to score (config parameters.model)";

    CHECK(args.count("test") > 0 ||
          !conf.get<std::string>("parameters.weights", "").empty())
        << "Need model weights to score (config parameters.weights or argument --weights)";

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    // Set device id and mode
    // If the gpu flag is not provided, allow the mode and device to be set
    // in the solver prototxt.
    if (conf.get<std::string>("parameters.mode", "CPU") == "GPU")
    {
        LOG(INFO) << "Dynamic decision about device ID " << 0;    // just a joke
        Caffe::SetDevice(0);
        Caffe::set_mode(Caffe::GPU);
    } else {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    }

    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(
        source->absolutize(conf.get<std::string>("parameters.solver")), &solver_param);

    // Instantiate the caffe net.
    std::string weights_file;
    if(args.count("test") > 0) {
        weights_file = args["test"].as<std::string>();
    }
    if(weights_file.empty() && !conf.get<std::string>("parameters.weights", "").empty()) {
        weights_file = conf.get<std::string>("parameters.weights");
    }

    Net<float> caffe_net(conf.get<std::string>("parameters.model"), caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(weights_file);

    int iterations = conf.get("parameters.test_iters", 50);
    LOG(INFO) << "Running for " << iterations << " iterations (adjust by config parameters.test_iters).";
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
                caffe_net.Forward(bottom_vec, &iter_loss);
        const float* labels = caffe_net.blob_by_name("label")->cpu_data();
        loss += iter_loss;
        int idx = 0;
        for (int j = 0; j < result.size(); ++j) {
            const std::string& output_name = caffe_net.blob_names()[
                    caffe_net.output_blob_indices()[j]];
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
        const std::string& output_name = caffe_net.blob_names()[
                caffe_net.output_blob_indices()[test_score_output_id[i]]];
        const float loss_weight =
                caffe_net.blob_loss_weights()[caffe_net.output_blob_indices()[i]];
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

    return 0;
}


// // Time: benchmark the execution time of a model.
// int time(const bpt::ptree& conf, std::shared_ptr<Feeder> feeder, std::shared_ptr<Source> source) {
//   CHECK_GT(model.size(), 0) << "Need a model definition to time.";

//   // Set device id and mode
//   if (FLAGS_gpu >= 0) {
//     LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
//     Caffe::SetDevice(FLAGS_gpu);
//     Caffe::set_mode(Caffe::GPU);
//   } else {
//     LOG(INFO) << "Use CPU.";
//     Caffe::set_mode(Caffe::CPU);
//   }
//   // Instantiate the caffe net.
//   Net<float> caffe_net(FLAGS_model, caffe::TRAIN);

//   // Do a clean forward and backward pass, so that memory allocation are done
//   // and future iterations will be more stable.
//   LOG(INFO) << "Performing Forward";
//   // Note that for the speed benchmark, we will assume that the network does
//   // not take any input blobs.
//   float initial_loss;
//   caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
//   LOG(INFO) << "Initial loss: " << initial_loss;
//   LOG(INFO) << "Performing Backward";
//   caffe_net.Backward();

//   const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
//   const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
//   const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
//   const vector<vector<bool> >& bottom_need_backward =
//       caffe_net.bottom_need_backward();
//   LOG(INFO) << "*** Benchmark begins ***";
//   LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
//   Timer total_timer;
//   total_timer.Start();
//   Timer forward_timer;
//   Timer backward_timer;
//   Timer timer;
//   std::vector<double> forward_time_per_layer(layers.size(), 0.0);
//   std::vector<double> backward_time_per_layer(layers.size(), 0.0);
//   double forward_time = 0.0;
//   double backward_time = 0.0;
//   for (int j = 0; j < FLAGS_iterations; ++j) {
//     Timer iter_timer;
//     iter_timer.Start();
//     forward_timer.Start();
//     for (int i = 0; i < layers.size(); ++i) {
//       timer.Start();
//       // Although Reshape should be essentially free, we include it here
//       // so that we will notice Reshape performance bugs.
//       layers[i]->Reshape(bottom_vecs[i], top_vecs[i]);
//       layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
//       forward_time_per_layer[i] += timer.MicroSeconds();
//     }
//     forward_time += forward_timer.MicroSeconds();
//     backward_timer.Start();
//     for (int i = layers.size() - 1; i >= 0; --i) {
//       timer.Start();
//       layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
//                           bottom_vecs[i]);
//       backward_time_per_layer[i] += timer.MicroSeconds();
//     }
//     backward_time += backward_timer.MicroSeconds();
//     LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
//       << iter_timer.MilliSeconds() << " ms.";
//   }
//   LOG(INFO) << "Average time per layer: ";
//   for (int i = 0; i < layers.size(); ++i) {
//     const caffe::string& layername = layers[i]->layer_param().name();
//     LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
//       "\tforward: " << forward_time_per_layer[i] / 1000 /
//       FLAGS_iterations << " ms.";
//     LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
//       "\tbackward: " << backward_time_per_layer[i] / 1000 /
//       FLAGS_iterations << " ms.";
//   }
//   total_timer.Stop();
//   LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
//     FLAGS_iterations << " ms.";
//   LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
//     FLAGS_iterations << " ms.";
//   LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
//     FLAGS_iterations << " ms.";
//   LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
//   LOG(INFO) << "*** Benchmark ends ***";
//   return 0;
// }



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

    CHECK_GE(args.count("train") + args.count("test"), 1)
        << "Specify either --test or --train";

    // if the user selected --cretedb <path> then just create database ?and quit?
    if(args.count("createdb") > 0) {
        // feeder will try to load data into memory or create/get caffe-readable database
        std::shared_ptr<Feeder> feeder {feeder_factory(source, conf)};
        CHECK(feeder) << "No feeder can hadle type: " << conf.get<std::string>("data.type");
        feeder->create_dbs(args["createdb"].as<std::string>());
        return 0;
    }

    // ugly hack ... change CWD to the path in the argumentd
    bfs::path cwd = bfs::current_path();
    bfs::current_path(bfs::path(args["source"].as<string>()));

    if(args.count("train") >= 1) {
        train(args, conf, source);
    }
    if(args.count("test") >= 1) {
        test(args, conf, source);
    }

    // ugly hack ... revert the previous CWD
    bfs::current_path(cwd);
    return 0;
}
