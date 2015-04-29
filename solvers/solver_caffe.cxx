#include "solver_caffe.hxx"

#include <cmath>
#include <sstream>
#include <random>
#include <vector>
using std::vector;

#include <boost/optional.hpp>

#include <google/protobuf/message.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <boost/smart_ptr.hpp>

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

#include <caffe/blob.hpp>
using caffe::Blob;
#include <caffe/data_layers.hpp>
#include <caffe/util/io.hpp>
#include <caffe/util/math_functions.hpp>


namespace vodigger {


void SolverCaffe::compute_right_size(std::shared_ptr<Feeder>& feeder,
                                     Phase phase,
                                     caffe::MemoryDataParameter *data_param)
{
  data_param->set_batch_size(feeder->nums(phase, BATCH)); // this needs tuning though!
  data_param->set_channels(feeder->nums(phase, CHANNEL));
  data_param->set_height(feeder->nums(phase, HEIGHT));
  data_param->set_width(feeder->nums(phase, WIDTH));
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
  caffe::NetParameter net_params;
  {
    std::istream* model_stream = source->read(properties.get<std::string>("parameters.model"));
    google::protobuf::io::IstreamInputStream gis {model_stream};

    google::protobuf::TextFormat::Parse(&gis, &net_params);
    delete model_stream;
  }
  // if there is a pretrained model then put it into net_
  if(source->exists(snapshot_filename_))
  {
    net_params.ParseFromIstream(source->read(this->snapshot_filename_));
    // net_weights.MergePartialFromCodedStream(coded_input)
    // this->net_->CopyTrainedLayersFrom(net_weights);
  }


  feed_ = (std::string("MemoryData").compare(net_params.layer(0).type()) == 0);

  CHECK(properties.get<std::string>("data.train", "").empty() ||
        source->exists(properties.get<std::string>("data.train"))) << "Train data not readable";
  CHECK(properties.get<std::string>("data.test", "").empty() ||
        source->exists(properties.get<std::string>("data.test"))) << "Test data not readable";
  CHECK(properties.get<std::string>("data.guess", "").empty() ||
        source->exists(properties.get<std::string>("data.guess"))) << "Guess data not readable";

  CHECK(source->exists(properties.get<std::string>("data.train")) ||
        source->exists(snapshot_filename_)) << "Either training file or a snapshot has to exist";

  bool has_snapshot = source_->exists(snapshot_filename_);
  Phase train_phase = has_snapshot ? TEST : TRAIN;

  caffe::NetState net_state;

  if(feed_) {
    caffe::MemoryDataParameter *dp = net_params.mutable_layer(0)->mutable_memory_data_param();
    // batch size is the critical attribute here
    train_batch_ = properties.get<int>("data.batch", feeder->nums(train_phase, BATCH));
    dp->set_batch_size(train_batch_);
    dp->set_channels(feeder->nums(train_phase, CHANNEL));
    dp->set_height(feeder->nums(train_phase, HEIGHT));
    dp->set_width(feeder->nums(train_phase, WIDTH));
  } else {
    if(has_snapshot)
      net_params.mutable_layer(0)->mutable_big_data_param()->set_source(
          source->absolutize(properties.get<std::string>("data.test")));
    else
      net_params.mutable_layer(0)->mutable_big_data_param()->set_source(
          source->absolutize(properties.get<std::string>("data.train")));
  }

  // guess and train have the same phase -- TRAIN
  net_state.set_phase(caffe::TRAIN);
  net_params.mutable_state()->CopyFrom(net_state);

  net_ = new caffe::Net<Dtype>(net_params);

  // if there are data to test on, instantiate a testing network
  if(!properties.get<std::string>("data.test", "").empty())
  {
    // in case of having BigDataLayer first ...
    // set up correct shape for TEST input layer ... can be layer #1 or #0
    if(std::string("BigData").compare(net_params.layer(1).type()) == 0)
    {
      net_params.mutable_layer(1)->mutable_big_data_param()->set_source(
        source->absolutize(properties.get<std::string>("data.test")));
    }
    else if(std::string("BigData").compare(net_params.layer(0).type()) == 0)
    {
      net_params.mutable_layer(0)->mutable_big_data_param()->set_source(
        source->absolutize(properties.get<std::string>("data.test")));
    }

    // in case of having MemoryLayer first ...
    caffe::MemoryDataParameter *mdp = nullptr;
    // set up correct shape for TEST input layer ... can be layer #1 or #0
    if(std::string("MemoryData").compare(net_params.layer(1).type()) == 0)
      mdp = net_params.mutable_layer(1)->mutable_memory_data_param();
    else if(std::string("MemoryData").compare(net_params.layer(0).type()) == 0)
      mdp = net_params.mutable_layer(0)->mutable_memory_data_param();

    if(mdp != nullptr) {
      // batch size is the critical attribute here
      test_batch_ = properties.get<int>("data.batch", feeder->nums(TEST, BATCH));
      mdp->set_batch_size(test_batch_);
      mdp->set_channels(feeder->nums(TEST, CHANNEL));
      mdp->set_height(feeder->nums(TEST, HEIGHT));
      mdp->set_width(feeder->nums(TEST, WIDTH));
    }

    // now common part again
    net_state.set_phase(caffe::TEST);
    net_params.mutable_state()->CopyFrom(net_state);
    test_net_ = new caffe::Net<Dtype>(net_params);
  }

  if(feed_ && test_net_ != nullptr)
  {
    // initialize data of MemoryDataLayer by supplying a vector of data
    boost::shared_ptr<caffe::Layer<Dtype> > input_layer = this->test_net_->layers()[0];
    caffe::MemoryDataLayer<Dtype> *input_memory_layer = dynamic_cast<caffe::MemoryDataLayer<Dtype>*>(input_layer.get());

    input_memory_layer->Reset( feeder_->data(TEST), feeder_->labels(TEST), feeder_->nums(TEST) );
  }

  const vector<boost::shared_ptr<Blob<Dtype> > >& net_data = this->net_->params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_data.size(); ++i) {
    const vector<int>& shape = net_data[i]->shape();
    history_.push_back(std::shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(std::shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(std::shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}



SolverCaffe::~SolverCaffe() noexcept
{
    if(this->net_ != nullptr) delete this->net_;
    if(this->test_net_ != nullptr) delete this->test_net_;
}



void SolverCaffe::run(std::ostream& output)
{
  if(!this->source_->exists(snapshot_filename_))
  {
    train_(output);
    snapshot_();
  }
  // now we have to have a trained network
  if(test_net_ != nullptr)
  {
    size_t full_test_iter = this->feeder_->nums(TEST, BATCH) / test_batch_;
    test_(full_test_iter, output);
  }
  // guess_(output);
}



void SolverCaffe::snapshot_()
{
  caffe::NetParameter net_data;
  net_->ToProto(&net_data, false);
  std::ostream *sout = this->source_->write(snapshot_filename_);
  CHECK(net_data.SerializeToOstream(sout));
  delete sout;
}



void SolverCaffe::train_(std::ostream& output)
{
  Dtype loss = .0, smoothloss = .0;
  vector<Blob<Dtype>*> bottom_vec;
  LOG(INFO) << "Starting network training";

  // obtain a reference to the input layer so we can sneak in some data into it's memory
  if(feed_)
  {
    boost::shared_ptr<caffe::Layer<Dtype> > input_layer = this->net_->layers()[0];
    caffe::MemoryDataLayer<Dtype> *input_memory_layer = dynamic_cast<caffe::MemoryDataLayer<Dtype>*>(input_layer.get());

    input_memory_layer->Reset( feeder_->data(TRAIN), feeder_->labels(TRAIN), feeder_->nums(TRAIN) );
  }


  for(size_t iter = 1; iter <= params_.max_iter(); ++iter)
  {
    if(feed_) {
      net_->ForwardPrefilled(&loss);
      net_->Backward();
    }
    else {
      net_->ForwardBackward(bottom_vec);
    }
    this->GDS_(iter);
    net_->Update();

    smoothloss = ((smoothloss * iter) + loss) / (iter + 1); // average loss over all recorded so far
    if(params_.has_display() && iter % params_.display() == 0) {
      output << "Iteration " << iter
             << ", loss = " << loss << " (smooth: " << smoothloss << ")" << std::endl;
    }

    if(params_.test_interval() != 0 && iter % params_.test_interval() == 0) {
      test_(1, output);
    }

  } // end for(iter)
}



void SolverCaffe::GDS_(size_t iter)
{
    // get the learning rate
    float rate = this->params_.base_lr() * std::pow(params_.gamma(),
                                                    int(iter / this->params_.stepsize()));

    // compute update value of network parameters (weights)
    const std::vector<boost::shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
    const std::vector<Dtype>& net_params_lr = this->net_->params_lr();
    const std::vector<Dtype>& net_params_weight_decay =
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



void SolverCaffe::test_(size_t iter, std::ostream& output)
{
  Dtype loss = 0;
  this->test_net_->ShareTrainedLayersWith(net_);
  vector<Blob<Dtype>*> bottom_vec;

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  for(size_t cur_iter=0; cur_iter < iter; ++cur_iter)
  {
    LOG(INFO) << "Test iteration #" << cur_iter;

    if(feed_) test_net_->ForwardPrefilled(&loss);
    else test_net_->Forward(bottom_vec, &loss);

    // Get probabilities
    // const boost::shared_ptr<Blob<Dtype> >& labels = test_net_->blob_by_name("guess");
    // const Dtype* labels_data = labels->cpu_data();

    const vector<Blob<Dtype>*>& result = test_net_->output_blobs();
    boost::shared_ptr<Blob<Dtype> > label_blob = test_net_->blob_by_name("label");
    const Dtype* labels = label_blob->cpu_data();

    int score_index = 0;
    for (int num = 0; num < result.size(); ++num)
    {
      const Dtype* result_vec = result[num]->cpu_data();
      const string& output_name =
          test_net_->blob_names()[test_net_->output_blob_indices()[num]];
      const float loss_weight =
          test_net_->blob_loss_weights()[test_net_->output_blob_indices()[num]];
      if(result[num]->count() > 2)
      {
        LOG(INFO) << output_name << " output blob of shape " << result[num]->shape_string();
        int outer_num = result[num]->count(0, 1);
        int prob_step = result[num]->count(1, 2);
        int inner_num = result[num]->count(2);
        int printed_results = 0;
        const Dtype* results = result[num]->cpu_data();
        for(int inum = 0; inum < outer_num, printed_results < 20; inum+=prob_step) {
          if(distribution(generator) < 0.1) continue;
          std::ostringstream oss;
          for (int onum = 0; onum < inner_num; ++onum) {
            oss << results[inum*inner_num+onum];
            if(prob_step > 1) oss << " (" << int(results[inum*inner_num+onum+prob_step]*100) << "%)";
            oss << ", ";
          }
          oss << "real label: " << int(labels[num]);
          LOG(INFO) << oss.str();
          ++printed_results;
        }
      }
      else
      {
        for (int k = 0; k < result[num]->count(); ++k)
        {
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
    if(true)
    {
      for(auto name: test_net_->layer_names())
      {
        boost::shared_ptr<caffe::Layer<Dtype> > layer = test_net_->layer_by_name(name);
        vector<boost::shared_ptr<Blob<Dtype> > > weights = layer->blobs();
        LOG(INFO) << "Layer " << name << " has " << weights.size() << " blobs";
        for (int i=0; i<weights.size(); ++i) {
          std::ostringstream oss;
          const Dtype* wdata = weights[i]->cpu_data();
          for(int w=0; w<weights[i]->count(); ++w) oss << wdata[w] << ",";
          LOG(INFO) << "#" << i << " blob, shape "
                    << weights[i]->shape_string()
                    << ", values: " << oss.str();
        }
      }
    }
  }
}


/**
* It is necessary to do as many iterations as there are chunks of source data
**/

void SolverCaffe::guess_(std::ostream& output)
{
  Dtype loss = 0;
  vector<Blob<Dtype>*> bottom_vec;

  // Dtype *zeroes, *labels;

  // initialize data of MemoryDataLayer by supplying a vector of data
  // boost::shared_ptr<caffe::Layer<Dtype> > input_layer = this->net_->layers()[0];
  // caffe::MemoryDataLayer<Dtype> *input_memory_layer = dynamic_cast<caffe::MemoryDataLayer<Dtype>*>(input_layer.get());

  // labels = feeder_->labels(GUESS);
  // zeroes = new Dtype[feeder_->nums(GUESS)];
  // std::memset(zeroes, (Dtype)0, sizeof(*zeroes) * feeder_->nums(GUESS));

  // prefill data into TRAINed network
  // input_memory_layer->set_batch_size(feeder_->nums(GUESS));
  // input_memory_layer->Reset( feeder_->data(GUESS), zeroes, feeder_->nums(GUESS) );

  net_->Forward(bottom_vec, &loss);
  std::vector<Blob<Dtype>*> dataout = net_->output_blobs();

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
    const Dtype* results = dataout[b]->cpu_data();
    for(int i = 0; i < outer_num; i+=prob_step) {
      for (int onum = 0; onum < inner_num; ++onum) {
        output << results[i*inner_num+onum];
        if(prob_step > 1) output << " (" << int(results[i*inner_num+onum+prob_step]*100) << "%)";
        output << ", ";
      }
      output << std::endl;
      //output << "real label: " << int(labels[i]) << std::endl;
    }
    output << "=============================" << std::endl;
  }

}



}

