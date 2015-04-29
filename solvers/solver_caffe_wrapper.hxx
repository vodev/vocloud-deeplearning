#ifndef VODIGGER_SOLVER_CAFFE_WRAPPER
#define VODIGGER_SOLVER_CAFFE_WRAPPER

#include "../vodigger.hxx"
#include "../inputs/feeder.hxx"
#include <string>
#include <memory>

#include <caffe/solver.hpp>
#include <caffe/proto/caffe.pb.h>
#include <boost/property_tree/ptree.hpp>
namespace bpt = boost::property_tree;


namespace vodigger
{


template<typename Dtype>
class SolverCaffeWrapper : public Solver
{
  caffe::Solver<Dtype>* solver_ = nullptr;
  caffe::Net<Dtype> net_ = nullptr;
public:
  SolverCaffeWrapper(const bpt::ptree& props,
                     std::shared_ptr<Feeder>& feeder,
                     std::shared_ptr<Source>& source) : Solver(props, feeder, source) {}

  virtual ~SolverCaffeWrapper() noexcept {
    if (solver_ != nullptr) delete solver_;
    if (net_ != nullptr) delete net_;
  }

  virtual void train(std::ostream& output);
  virtual void test(std::ostream& output);

protected:

};



}

#endif
