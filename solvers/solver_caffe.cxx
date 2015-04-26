#include "solver_caffe.hxx"

#include <cmath>
#include <vector>
using std::vector;

#include <boost/optional.hpp>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <boost/smart_ptr.hpp>

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

#include <caffe/blob.hpp>
using caffe::Blob;
#include <caffe/data_layers.hpp>
#include <caffe/util/math_functions.hpp>


namespace vodigger {


void SolverCaffe::compute_right_size(std::shared_ptr<Feeder>& feeder,
                                     Phase phase,
                                     caffe::MemoryDataParameter *mdp)
{
  mdp->set_batch_size(feeder->nums(phase, BATCH)); // this needs tuning though!
  mdp->set_channels(feeder->nums(phase, CHANNEL));
  mdp->set_height(feeder->nums(phase, HEIGHT));
  mdp->set_width(feeder->nums(phase, WIDTH));
}


SolverCaffe::SolverCaffe(const bpt::ptree& properties,
                         std::shared_ptr<Feeder>& feeder,
                         std::shared_ptr<Source>& source) : Solver(properties, feeder, source)
{
  // construct path to trained model
  bfs::path snapshot_path = bfs::path(properties.get<std::string>("parameters.model")).replace_extension(bfs::path(".protobin"));
  snapshot_filename_ = snapshot_path.string();
  // read solver parameters
  CHECK(source->exists(properties.get<std::string>("parameters.solver"))) <<
    "Solver file not readable";
  {
    std::istream* solver_stream = source->read(properties.get<std::string>("parameters.solver"));
    google::protobuf::io::IstreamInputStream gis {solver_stream};

    google::protobuf::TextFormat::Parse(&gis, &params_);
    delete solver_stream;
  }
  // set solver mode and device - mode [default CPU], device [default 0]
  if(params_.has_solver_mode() && params_.solver_mode() == caffe::SolverParameter::GPU) {
      if(params_.has_device_id()) caffe::Caffe::SetDevice(params_.device_id());
      else caffe::Caffe::SetDevice(0);
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
  // read network parameters
  CHECK(source->exists(properties.get<std::string>("parameters.model"))) <<
    "Model file not readable";
  caffe::NetParameter net_params, net_weights;
  {
    std::istream* model_stream = source->read(properties.get<std::string>("parameters.model"));
    google::protobuf::io::IstreamInputStream gis {model_stream};

    google::protobuf::TextFormat::Parse(&gis, &net_params);
    delete model_stream;
  }

  CHECK(std::string("MemoryData").compare(net_params.layer(0).type()) == 0) <<
    "First layer has to be of type=\"MemoryData\"";

  caffe::MemoryDataParameter* mdp = net_params.mutable_layer(0)->mutable_memory_data_param();
  if(source_->exists(snapshot_filename_))
  {
    if(!properties.get<std::string>("data.guess", "").empty())
      this->compute_right_size(feeder, GUESS, mdp);
    else
      this->compute_right_size(feeder, TEST, mdp);
  } else {
    this->compute_right_size(feeder, TRAIN, mdp);
  }

  if(std::string("MemoryData").compare(net_params.layer(1).type()) == 0)
  {
    mdp = net_params.mutable_layer(1)->mutable_memory_data_param();
    if(source_->exists(snapshot_filename_))
    {
      if(!properties.get<std::string>("data.test", "").empty())
        this->compute_right_size(feeder, TEST, mdp);
      else
        this->compute_right_size(feeder, GUESS, mdp);
    } else {
      this->compute_right_size(feeder, TRAIN, mdp);
    }
  }

  // first, set TRAIN state of the network
  caffe::NetState net_state;
  net_state.set_phase(caffe::TRAIN);
  net_params.mutable_state()->CopyFrom(net_state);
  // instantiate TRAIN net
  net_ = new caffe::Net<float>(net_params);
  // if there is pretrained model then update the main model (net_) with those weights
  if(source_->exists(snapshot_filename_))
  {
    std::istream* snapshot_stream = source->read(this->snapshot_filename_);
    google::protobuf::io::IstreamInputStream gis {snapshot_stream};
    google::protobuf::TextFormat::Parse(&gis, &net_weights);
    this->net_->CopyTrainedLayersFrom(net_weights);
  }

  const vector<boost::shared_ptr<Blob<float> > >& net_data = this->net_->params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_data.size(); ++i) {
    const vector<int>& shape = net_data[i]->shape();
    history_.push_back(std::shared_ptr<Blob<float> >(new Blob<float>(shape)));
    update_.push_back(std::shared_ptr<Blob<float> >(new Blob<float>(shape)));
    temp_.push_back(std::shared_ptr<Blob<float> >(new Blob<float>(shape)));
  }

  // instantiate testing net (with accuracy layer and stuff)
  net_state.set_phase(caffe::TEST);
  net_params.mutable_state()->CopyFrom(net_state);
  test_net_ = new caffe::Net<float>(net_params);

}


SolverCaffe::~SolverCaffe()
{
    if(this->net_ != nullptr) {
      delete this->net_;
      this->net_ = nullptr;
    }
    if(this->test_net_ != nullptr) {
      delete this->test_net_;
      this->test_net_ = nullptr;
    }
}


void SolverCaffe::run(std::ostream& output)
{
  if(!source_->exists(snapshot_filename_))
  {
    CHECK_GT(feeder_->nums(TRAIN), 0) << "There are no data to train and no pre-trained model!";
    train_(output);
    snapshot_();
  }
  // now we have to have a trained network
  if(feeder_->nums(TEST) > 0) test_(output);
  if(feeder_->nums(GUESS) > 0) guess_(output);
}


void SolverCaffe::snapshot_()
{
  caffe::NetParameter net_data;
  net_->ToProto(&net_data, false);
  std::ostream *sout = source_->write(snapshot_filename_);
  CHECK(net_data.SerializeToOstream(sout));
  delete sout;
}


void SolverCaffe::train_(std::ostream& output)
{
  float loss = .0, smoothloss = .0;
  float rate = 0.01;
  LOG(INFO) << "Starting network training";

  // obtain a reference to the input layer so we can sneak in some data into it's memory
  boost::shared_ptr<caffe::Layer<float> > input_layer = this->net_->layers()[0];
  caffe::MemoryDataLayer<float> *input_memory_layer = dynamic_cast<caffe::MemoryDataLayer<float>*>(input_layer.get());

  input_memory_layer->set_batch_size(feeder_->nums(TRAIN));
  input_memory_layer->Reset( feeder_->data(TRAIN), feeder_->labels(TRAIN), feeder_->nums(TRAIN) );

  for(size_t iter = 0; iter <= params_.max_iter(); ++iter)
  {
    net_->ForwardPrefilled(&loss);
    net_->Backward();
    this->GDS_(iter);
    net_->Update();

    smoothloss = ((smoothloss * iter) + loss) / (iter + 1); // average loss over all recorded so far
    if(params_.has_display() && iter % params_.display() == 0) {
      output << "Iteration " << iter
             << ", loss = " << loss << " (smooth: " << smoothloss << ")" << std::endl;
    }

    if(params_.test_interval() != 0 && iter % params_.test_interval() == 0) {
      test_(output);
    }

  } // end for(iter)
}


void SolverCaffe::GDS_(int iter)
{
    // get the learning rate
    float rate = this->params_.base_lr() * std::pow(params_.gamma(),
                                                    int(iter / this->params_.stepsize()));

    // compute update value of network parameters (weights)
    const std::vector<boost::shared_ptr<Blob<float> > >& net_params = this->net_->params();
    const std::vector<float>& net_params_lr = this->net_->params_lr();
    const std::vector<float>& net_params_weight_decay =
        this->net_->params_weight_decay();

    // ClipGradients();
    float momentum = this->params_.momentum();
    float weight_decay = this->params_.weight_decay();
    switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        for (int param_id = 0; param_id < net_params.size(); ++param_id) {
          // Compute the value to history, and then copy them to the blob's diff.
          float local_rate = rate * net_params_lr[param_id];
          float local_decay = weight_decay * net_params_weight_decay[param_id];

          if (local_decay) {
            // perform L2 regularization
            // with added weight decay
            caffe::caffe_axpy(net_params[param_id]->count(),
                local_decay,
                net_params[param_id]->cpu_data(),
                net_params[param_id]->mutable_cpu_diff());
          }

          caffe::caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                    net_params[param_id]->cpu_diff(), momentum,
                    history_[param_id]->mutable_cpu_data());
          // copy
          caffe::caffe_copy(net_params[param_id]->count(),
              history_[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        }
        break;
      case caffe::Caffe::GPU:
    #ifndef CPU_ONLY
        for (int param_id = 0; param_id < net_params.size(); ++param_id) {
          // Compute the value to history, and then copy them to the blob's diff.
          float local_rate = rate * net_params_lr[param_id];
          float local_decay = weight_decay * net_params_weight_decay[param_id];

          if (local_decay) {
            // perform L2 regularization
            // with added weight decay
            caffe::caffe_gpu_axpy(net_params[param_id]->count(),
                local_decay,
                net_params[param_id]->gpu_data(),
                net_params[param_id]->mutable_gpu_diff());
          }

          caffe::caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
                    net_params[param_id]->gpu_diff(), momentum,
                    history_[param_id]->mutable_gpu_data());
          // copy
          caffe::caffe_copy(net_params[param_id]->count(),
              history_[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        }
    #else
        // NO_GPU;
    #endif
        break;
      default:
        LOG(FATAL) << "Unknown caffe mode: " << caffe::Caffe::mode();
      }
}

void SolverCaffe::test_(std::ostream& output)
{
  float loss = 0;
  this->test_net_->ShareTrainedLayersWith(net_);
  // initialize data of MemoryDataLayer by supplying a vector of data
  boost::shared_ptr<caffe::Layer<float> > input_layer = this->test_net_->layers()[0];
  caffe::MemoryDataLayer<float> *input_memory_layer = dynamic_cast<caffe::MemoryDataLayer<float>*>(input_layer.get());

  input_memory_layer->set_batch_size(feeder_->nums(TEST));
  input_memory_layer->Reset( feeder_->data(TEST), feeder_->labels(TEST), feeder_->nums(TEST) );

  test_net_->ForwardPrefilled(&loss);

  // Get probabilities
  // const boost::shared_ptr<Blob<float> >& labels = test_net_->blob_by_name("guess");
  // const float* labels_data = labels->cpu_data();

  const vector<Blob<float>*>& result = test_net_->output_blobs();
  int score_index = 0;
  for (int j = 0; j < result.size(); ++j) {
    const float* result_vec = result[j]->cpu_data();
    const string& output_name =
        test_net_->blob_names()[test_net_->output_blob_indices()[j]];
    const float loss_weight =
        test_net_->blob_loss_weights()[test_net_->output_blob_indices()[j]];
    for (int k = 0; k < result[j]->count(); ++k) {
      output << "  Test net output #"
             << score_index++ << ": " << output_name << " = "
             << result_vec[k];
      if (loss_weight) {
        output << " (*loss weight = " << loss_weight
               << " => " << loss_weight * result_vec[k] << " loss)";
      }
      output << std::endl;
    }
  }
}


void SolverCaffe::guess_(std::ostream& output)
{
  float loss = 0;
  float *zeroes, *labels;

  // initialize data of MemoryDataLayer by supplying a vector of data
  boost::shared_ptr<caffe::Layer<float> > input_layer = this->net_->layers()[0];
  caffe::MemoryDataLayer<float> *input_memory_layer = dynamic_cast<caffe::MemoryDataLayer<float>*>(input_layer.get());

  labels = feeder_->labels(GUESS);
  zeroes = new float[feeder_->nums(GUESS)];
  std::memset(zeroes, (float)0, sizeof(*zeroes) * feeder_->nums(GUESS));

  // prefill data into TRAINed network
  input_memory_layer->set_batch_size(feeder_->nums(GUESS));
  input_memory_layer->Reset( feeder_->data(GUESS), zeroes, feeder_->nums(GUESS) );

  std::vector<Blob<float>*> dataout = net_->ForwardPrefilled(&loss);

  output << "============= guess results ==============" << std::endl;
  for(int b = 0; b < dataout.size(); ++b)
  {
    output << b << "th blob of shape " << dataout[b]->shape_string() << std::endl;

    // let's make LOSS as a special case
    if(dataout[b]->count() == 1) {
      output << "LOSS: " << dataout[b]->data_at(0,0,0,0) << std::endl;
      continue;
    }

    int outer_num = dataout[b]->count(0, 1);
    int prob_step = dataout[b]->count(1, 2);
    int inner_num = dataout[b]->count(2);
    const float* results = dataout[b]->cpu_data();
    for(int i = 0; i < outer_num; i+=prob_step) {
      for (int j = 0; j < inner_num; ++j) {
        output << results[i*inner_num+j];
        if(prob_step > 1) output << " (" << int(results[i*inner_num+j+prob_step]*100) << "%)";
        output << ", ";
      }
      output << "real label: " << int(labels[i]) << std::endl;
    }
    output << "=============================" << std::endl;
  }

}



}

