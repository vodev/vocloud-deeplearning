#ifndef VODIGGER_SOURCE

#define VODIGGER_SOURCE

#include <algorithm>
#include <istream>
#include <string>
#include <vector>

#include "../vodigger.hxx"


namespace vodigger {


/**
 * An abstract class representing a contained holding more files. So in practice it can be
 * a folder or an archive.
 */
class Source {

protected:
    Source() = default; // it's an abstract class - disallow public construction
    std::vector<std::string> files_;

    Source(const Source&) = delete;      // we will NOT define copy constructor
    Source(Source&&)      = delete;      // we will NOT define move constructor

public:

    virtual ~Source() noexcept    = default;

    virtual std::istream* read (const std::string&) = 0;
    virtual std::ostream* write(const std::string&) = 0;
    virtual bool exists (const std::string& filename)
    {
        return std::find(files_.begin(), files_.end(), filename) != files_.end();
    }

    virtual const std::vector<std::string>& files() {return this->files_;}
};


// namespace vodigger end
}


#endif
