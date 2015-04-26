#ifndef VODDIGER_SOURCE_FACTORY

#define VODDIGER_SOURCE_FACTORY


#include "source.hxx"
#include "source_folder.hxx"


namespace vodigger {

/**Factory pattern constructor which decides what type of file is given and opens it.
*/
Source* source_factory(const std::string);


// namespace vodigger end
}


#endif
