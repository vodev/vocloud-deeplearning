#include "source_folder.hxx"

#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>

#include <glog/logging.h>

#include "../utils/string.hxx"

namespace vodigger {


SourceFolder::SourceFolder(const std::string& dirname) : Source(), folder_(bfs::canonical(bfs::path(dirname)))
{
  try
  {
    if( bfs::exists(this->folder_) && bfs::is_directory(this->folder_) ) {
      size_t bp_len = this->folder_.string().size() + 1;  // length of base_path (to be stripped from file paths)
      for(auto entry = bfs::directory_iterator(this->folder_); entry != bfs::directory_iterator(); ++entry) {
        // directory_entry (result of iterator) yields always absolute path!
        bfs::path p = (*entry).path();
        std::string fp = p.string().substr(bp_len); // get relative file path
        this->files_.push_back(fp);
        this->paths_[fp] = std::move(p);
        LOG(INFO) << fp;
      }
    }
    else
    {
      throw std::invalid_argument("source doesn't exist or isn't a folder");
    }
  }
  catch (const bfs::filesystem_error& ex)
  {
    std::cerr << ex.what() << std::endl;
  }
}


bool SourceFolder::handles(const std::string& pathname)
{
  return bfs::is_directory(bfs::path(pathname));
}


std::istream* SourceFolder::read(const std::string& filename)
{
  try {
    // binary or not?
    std::ios_base::openmode flags = std::ios::in | std::ios::binary | std::ios::trunc;
    if( endswith(filename, "csv") || endswith(filename, "json") || endswith(filename, "txt") ) {
      flags = std::ios::in;
    }
    return new std::ifstream(this->paths_[filename].string(), flags);
  } catch (const std::out_of_range& ex) {
    return nullptr;
  }
}

std::ostream* SourceFolder::write(const std::string& filename)
{
  bfs::path ap = (this->folder_ / bfs::path(filename));
  LOG(INFO) << "Requested writing to " << ap.string() << std::endl;
  CHECK(!bfs::exists(ap)) << "Can't write to file '" << ap.string() << "'', already exists";
  return new std::ofstream(ap.native(), std::ios::binary | std::ios::trunc);
}


}
