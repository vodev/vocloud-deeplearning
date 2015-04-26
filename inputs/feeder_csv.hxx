#ifndef VODIGGER_FEEDER_CSV
#define VODIGGER_FEEDER_CSV

#include <istream>
#include <vector>
#include <array>

#include "feeder.hxx"
#include "../vodigger.hxx"
#include "../utils/string.hxx"


namespace vodigger {



class FeederCsv : public Feeder {

	char delim_ = ',';
	bool has_header_ = true;
	// indices of columns in CSV files
	int id_, label_, data_start_, data_end_;
	// path to sources for phases train/test/guess
	std::array<std::string, 3> data_source_;
	// data handling
	std::array<float*, 3> labels_;		// pointers to flat data
	std::array<float*, 3> data_;		// pointers to matching labels (one per datum)
	std::array<int, 3> nums_;			// number of "samples" in data

	void create_db(const std::string&, const std::vector<std::istream*>&);
	void prefetch_data(Phase phase);

public:
	static bool handles(const std::string& type) {return std::string("csv").compare(type) == 0;}

	FeederCsv(std::shared_ptr<Source>& source, bpt::ptree& params);
	virtual ~FeederCsv();

	virtual void create_dbs(const std::string& db_filename);

	virtual float* data(Phase phase);
	virtual float* labels(Phase phase);
	virtual size_t nums(Phase phase, Shape shape = BATCH);

};



}


#endif
