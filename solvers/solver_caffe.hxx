#ifndef VODIGGER_SOLVER_CAFFE
#define VODIGGER_SOLVER_CAFFE

#include <string>
using std::string;
#include <memory>

#include <caffe/net.hpp>

#include "solver.hxx"

#include "../vodigger.hxx"
#include "../inputs/source.hxx"
#include "../inputs/feeder.hxx"


namespace vodigger {


class SolverCaffe : public Solver
{
	typedef float Dtype;

	caffe::Net<Dtype> *net_ = nullptr, *test_net_ = nullptr;
	caffe::SolverParameter params_;

	// Filename where is supposed to be a model snapshot (if it isn't there then we create it)
	std::string snapshot_filename_;

	// following atributes are for momentum/snapshot/restore purposes
	std::vector<std::shared_ptr<caffe::Blob<Dtype> > > history_, update_, temp_;

	// split functionality into more functions so it isn't a mess
	void train_(std::ostream& output);
	void test_(std::ostream& output);
	void guess_(std::ostream& output);
	void snapshot_();
	// perform gradient descent on trained network's parameters
	void GDS_(int iter);

public:
	/* Find the caffe solver by name (proto files in solvers/ forlder) and instantiate it */
	SolverCaffe(const bpt::ptree&, std::shared_ptr<Feeder>&, std::shared_ptr<Source>&);

	virtual ~SolverCaffe() noexcept;

	virtual void run(std::ostream& output);
};



}

#endif
