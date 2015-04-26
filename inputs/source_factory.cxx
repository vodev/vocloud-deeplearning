#include "source_factory.hxx"

#include "source_folder.hxx"


namespace vodigger {


Source* source_factory(const std::string path) {
	if(SourceFolder::handles(path)) return new SourceFolder(path);
	return nullptr;
}


}
