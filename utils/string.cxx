#include "string.hxx"
#include "../vodigger.hxx"


namespace vodigger {


bool endswith(const std::string& str, const std::string& end)
{
	if( str.length() < end.length()) return false;
	return str.compare(str.length() - end.length(), end.length(), end) == 0;
}


std::string phaseToString(Phase phase)
{
	switch(phase)
	{
		case TRAIN: return std::string("train");
		case TEST: return std::string("test");
		case GUESS: return std::string("guess");
	}
}


}
