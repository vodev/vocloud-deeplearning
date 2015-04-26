#ifndef VODIGGER_FEEDER
#define VODIGGER_FEEDER

#include <string>
#include <memory>

#include <boost/property_tree/ptree.hpp>
namespace bpt = boost::property_tree;

#include <caffe/blob.hpp>

#include "source.hxx"
#include "../vodigger.hxx"

/**
*
*/

namespace vodigger {

/**
* Iterator going through source and returning tuples (key, data)
*/


class Feeder // : public std::iterator<std::input_iterator_tag, std::pair<std::string,array<float> > >
{
	Feeder(const Feeder&) = delete;
	Feeder(Feeder&&) = delete;

	protected:

	std::shared_ptr<Source> source_;
	Feeder(std::shared_ptr<Source>& source, bpt::ptree& params) : source_(source) {};

	public:

	/**
	* Creates a LevelDB database from given sources (test&train).
	* @param db_path: string absolute file path (without extension) where to create the database
	*/
	virtual void create_dbs(const std::string&) = 0;

	/**
	* Return data/label/num so it can be fed into MemoryDataLayer::Reset function
	*/
	virtual float* data(Phase phase)  = 0;
	virtual float* labels(Phase phase) = 0;
	virtual size_t nums(Phase phase, Shape shape = BATCH) = 0;


	virtual ~Feeder() {}

};


}

#endif
