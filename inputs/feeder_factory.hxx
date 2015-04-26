#ifndef VODDIGER_FEEDER_FACTORY

#define VODDIGER_FEEDER_FACTORY


#include "feeder.hxx"
#include "source.hxx"


namespace vodigger {

/**Factory pattern constructor which decides what type of file is given and opens it.
*/
Feeder* feeder_factory(std::shared_ptr<Source>&, bpt::ptree&);

// namespace vodigger end
}


#endif
