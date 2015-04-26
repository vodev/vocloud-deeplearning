#ifndef VODIGGER_SOURCE_FOLDER

#define VODIGGER_SOURCE_FOLDER

#include "source.hxx"
#include <map>

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;


namespace vodigger {


class SourceFolder : public Source {

	bfs::path folder_;
	// key is always path RELATIVE to the source folder
	std::map<std::string, bfs::path> paths_;

public:
	SourceFolder(const std::string&);

	virtual std::istream* read (const std::string&);
	virtual std::ostream* write(const std::string&);

	virtual ~SourceFolder() noexcept = default;

	/**
	* Function returns true if can handle specified pathname
	*/
	static bool handles(const std::string&);
};


}

#endif
