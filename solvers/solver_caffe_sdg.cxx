#include <iostream>
#include <cmath>
#include <sstream>
#include <random>
#include <vector>
using std::vector;

#include <google/protobuf/message.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <boost/smart_ptr.hpp>
using boost::shared_ptr;

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

#include <caffe/blob.hpp>
using caffe::Blob;
#include <caffe/net.hpp>
using caffe::Net;
#include <caffe/util/io.hpp>
#include <caffe/util/math_functions.hpp>

#include "solver_caffe_sdg.hxx"


namespace vodigger
{


template <typename Dtype>
void SolverCaffeSDG<Dtype>::Init(const caffe::SolverParameter& param)
{
  LOG(INFO) << "Initializing solver from parameters: " << std::endl
            << param.DebugString();
  this->param_ = param;
  this->param_.set_net(source_->absolutize(props_.get<std::string>("parameters.model")));
  if(this->param_.has_snaphshot_prefix()) {
    this->param_.set_snaphshot_prefix(source_->absolutize(this->param_->snaphshot_prefix()));
  }
  CHECK_GE(this->param_.average_loss(), 1) << "average_loss should be non-negative.";
  if (this->param_.random_seed() >= 0) {
    caffe::Caffe::set_random_seed(this->param_.random_seed());
  }
  // Scaffolding code
  this->InitTrainNet();
  this->InitTestNets();
  LOG(INFO) << "Solver scaffolding done.";
  this->iter_ = 0;
  this->current_step_ = 0;
}


template <typename Dtype>
void SolverCaffeSDG<Dtype>::Test(const int test_net_id)
{
  std::cout << "TESTING GUYS!!!!" << std::endl;
  LOG(INFO) << "Iteration " << this->iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(this->test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(this->net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = this->test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < this->param_.test_iter(test_net_id); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (this->param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (this->param_.test_compute_loss()) {
    loss /= this->param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const std::string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    std::ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / this->param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
        << mean_score << loss_msg_stream.str();
  }
}



template <typename Dtype>
void SolverCaffeSDG<Dtype>::Solve(const char* resume_file)
{
  LOG(INFO) << "Solving " << this->net_->name();
  LOG(INFO) << "Learning Rate Policy: " << this->param_.lr_policy();

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    this->Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  this->Step(this->param_.max_iter() - this->iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (this->param_.snapshot_after_train()
      && (!this->param_.snapshot() || this->iter_ % this->param_.snapshot() != 0)) {
    this->Snapshot();
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    Dtype loss;
    this->net_->ForwardPrefilled(&loss);
    LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss;
  }
  if (this->param_.test_interval() && this->iter_ % this->param_.test_interval() == 0) {
    this->TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}



template <typename Dtype>
void SolverCaffeSDG<Dtype>::TestAll() {
  for (int test_net_id = 0; test_net_id < this->test_nets_.size(); ++test_net_id) {
    this->Test(test_net_id);
  }
}



}
