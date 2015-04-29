#ifndef VODIGGER_SOLVER
#define VODIGGER_SOLVER

#include "../vodigger.hxx"
#include "../inputs/feeder.hxx"

#include <memory>

#include <boost/property_tree/ptree.hpp>
namespace bpt = boost::property_tree;


namespace vodigger {

class Solver
{
protected:
    std::shared_ptr<Source> source_;
    std::shared_ptr<Feeder> feeder_;
    const bpt::ptree& props_;

    Solver(const bpt::ptree& properties,
           std::shared_ptr<Feeder>& feeder,
           std::shared_ptr<Source>& source) : props_(properties), feeder_(feeder), source_(source)
    {}

    Solver(Solver&&) = delete;
    Solver(const Solver&) = delete;

public:
    virtual ~Solver() noexcept {}
    virtual void run(std::ostream& output) = 0;
};


}

#endif
