#ifndef VODIGGER_SOLVER_CAFFE_SDG
#define VODIGGER_SOLVER_CAFFE_SDG

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
class SolverCaffeSDG : public caffe::SGDSolver<Dtype>
{
  std::shared_ptr<Source> source_;
  std::shared_ptr<Feeder> feeder_;
  const bpt::ptree& props_;

public:
  SolverCaffeSDG(const bpt::ptree& props,
                 std::shared_ptr<Feeder>& feeder,
                 std::shared_ptr<Source>& source) :
    props_(props),
    source_(source),
    feeder_(feeder),
    caffe::SGDSolver<Dtype>(source->absolutize(props.get<std::string>("parameters.solver")))
    {}

  void Init(const caffe::SolverParameter& param);
  void Solve(const char* resume_file = NULL) override;

protected:
  void Test(const int test_net_id);
  void TestAll();

};



}

#endif
