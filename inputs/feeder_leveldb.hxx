#ifndef VODIGGER_FEEDER_LEVELDB
#define VODIGGER_FEEDER_LEVELDB


#include "feeder.hxx"

/**
*
*/

namespace vodigger {

/**
* Iterator going through source and returning tuples (key, data)
*/


class FeederLevelDB  : public Feeder
{
	// paths to data sources (databases in this case)
	std::array<std::string, 3> data_source_;

public:

	static bool handles(const std::string& type) {return std::string("leveldb").compare(type) == 0;}

	FeederLevelDB(std::shared_ptr<Source>& source, bpt::ptree& params);

	virtual void create_dbs(const std::string&) { /* DB already created */ }

	/**
	* Return data/label/num so it can be fed into MemoryDataLayer::Reset function
	*/
	virtual float* data(Phase phase) {return nullptr;}
	virtual float* labels(Phase phase) {return nullptr;}
	virtual size_t nums(Phase phase, Shape shape = BATCH) {return 0;}


};


}

#endif
