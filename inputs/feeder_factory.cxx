#include "feeder_factory.hxx"

#include "feeder_csv.hxx"
#include "feeder_leveldb.hxx"


namespace vodigger {


Feeder* feeder_factory(std::shared_ptr<Source>& source, bpt::ptree& params)
{
	std::string type = params.get<std::string>("data.type");
	if (FeederCsv::handles(type)) return new FeederCsv(source, params);
	if (FeederLevelDB::handles(type)) return new FeederLevelDB(source, params);
	return nullptr;
}


}
