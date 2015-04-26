
// This script converts the MNIST dataset to a lmdb (default) or
// leveldb (--backend=leveldb) format used by caffe to load data.
// Usage:
//    convert_mnist_data [FLAGS] input_image_file input_label_file
//                        output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <stdint.h>
#include <sys/stat.h>

#include <vector>
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <sstream>
#include <cstring>  // std::memcpy
#include <string>
using std::string;

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;


#include <caffe/proto/caffe.pb.h>
/* defines
message Datum {
  optional int32 channels = 1;
  optional int32 height = 2;
  optional int32 width = 3;
  // the actual image data, in bytes
  optional bytes data = 4;
  optional int32 label = 5;
  // Optionally, the datum could also hold float data.
  repeated float float_data = 6;
  // If true data contains an encoded image that need to be decoded
  optional bool encoded = 7 [default = false];
}
*/

#include "feeder_csv.hxx"
#include "../utils/string.hxx"

#define VD_MAX(a, b) ((a)>(b)?(a):(b))


namespace vodigger {


FeederCsv::FeederCsv(std::shared_ptr<Source>& source, bpt::ptree& params) : Feeder(source, params)
{
  this->delim_ = params.get("data.delimiter", ',');
  this->has_header_ = params.get("data.header", true);
  this->id_ = params.get<int>("data.id", -1);
  this->label_ = params.get<int>("data.label", -1);
  this->data_start_ = params.get<int>("data.start", -1);
  this->data_end_ = params.get<int>("data.end", -1);

  this->data_.fill(nullptr);
  this->labels_.fill(nullptr);
  this->nums_.fill(0);

  this->data_source_[TRAIN] = params.get<std::string>("data.train", "");
  this->data_source_[TEST]  = params.get<std::string>("data.test", "");
  this->data_source_[GUESS] = params.get<std::string>("data.guess", "");
}


FeederCsv::~FeederCsv()
{
  for(float* d: this->data_) if(d != nullptr) delete[] d;
  for(float* l: this->labels_) if(l != nullptr) delete[] l;
}


float* FeederCsv::data(Phase phase)
{
  if(this->data_[phase] == nullptr) this->prefetch_data(phase);
  return this->data_[phase];
}


float* FeederCsv::labels(Phase phase)
{
  if(this->labels_[phase] == nullptr) this->prefetch_data(phase);
  return this->labels_[phase];
}


size_t FeederCsv::nums(Phase phase, Shape shape)
{
  switch(shape) {
    case BATCH:
      if(this->nums_[phase] == 0) this->prefetch_data(phase);
      return this->nums_[phase];
    case CHANNEL:
    case HEIGHT:
      return 1;
    case WIDTH:
      return this->data_end_ - this->data_start_ + 1;
  }
}


void FeederCsv::prefetch_data(Phase phase)
{
  if(this->data_[phase] != nullptr) return;  // data already loaded

  if(this->data_source_[phase].empty())
  {
    this->nums_[phase] = 0;
    return;
  }

  // spectrum has rows/height equal to 1
  // spectrum has cols/width set from a config file
  uint32_t cols = this->data_end_ - this->data_start_ + 1;
  // values read from sources
  int label = 0;
  std::string id, value, header, line;
  // consecutive indices
  size_t sourcei = 0, rowi = 0, coli = 0, datai = 0, maxdatai = 0;
  // handle data memory fiddling
  float *data = new float[cols]; // one row of data at time
  float *labels;                 // will be the returned labels array
  std::vector<float*>  data_vec;  // holds all read data
  std::vector<float> labels_vec;  // accumulate labels
  // open all streams to all sources in this phase (later it really might return more sources)
  std::vector<std::istream*> streams {this->source_->read(this->data_source_[phase])};
  // streams the current line read from source
  std::istringstream linestream;

  CHECK_GT(streams.size(), 0) << "No data files provided!";

  CHECK_GT(cols, 0) << "Wrong data.start or data.end setting";
  CHECK_NE(this->label_, this->data_start_) << "Wrong data.label or data.start setting";

  LOG(INFO) << "CSV settings -- delimiter: '" << this->delim_
            << "' label: " << this->label_
            << " datacols: [" << this->data_start_ << ", " << this->data_end_
            << "] header: " << (this->has_header_ ? "yes" : "no");

  for(std::istream *stream: streams) {
    // strip header if set
    if(this->has_header_) {
      std::getline(*stream, header);
      size_t colcount = count(header.begin(), header.end(), this->delim_);
      if(colcount < VD_MAX(this->data_end_, this->label_) ) {
        continue;
      }
    }

    // iterate through all rows in the file
    while(stream->good())
    {
      // parse every file by rows
      std::getline(*stream, line);
      linestream.str(line);
      linestream.clear(); // clear any bad flags
      // iterate through columns and assign them to the right structures
      coli = 0;
      datai = 0;
      while(linestream.good())
      {
        std::getline(linestream, value, this->delim_);
        if(coli == this->label_) {
          // we got the label
          label = atoi(value.c_str());
        } else if (coli >= this->data_start_ && coli <= this->data_end_) {
          // we got data value
          data[datai] = atof(value.c_str());
          ++datai;
        }
        else if (coli == this->id_) {
          id = value;
        } else {
          // unknown column index
        }
        ++coli;
      }
      // only if we parsed a valid row (with sufficient data in it)
      if(datai == cols) {
        data_vec.push_back(data);
        labels_vec.push_back(label);
        data = new float[cols];
        ++rowi;
      } else {
        LOG(INFO) << "Discarting invalid row: " << line;
      }
      maxdatai = VD_MAX(datai, maxdatai);
    }
    ++sourcei;
    delete stream;
  }
  LOG(INFO) << "Processed " <<  maxdatai << " cols and " << rowi << " valid rows in " << sourcei << " files.";

  // there will be always stale chunk of memory
  delete[] data;
   // create one huge chunk of memory which will hold the data
  data = new float[rowi*cols];
  for(size_t i = 0; i < data_vec.size(); ++i)
  {
    std::memcpy(data+(i*cols), data_vec[i], cols*sizeof(*data));
    delete[] data_vec[i];
  }
  labels = new float[labels_vec.size()];
  std::memcpy(labels, labels_vec.data(), labels_vec.size() * sizeof(*labels));

  // save the result into class' internals
  this->data_[phase] = data;
  this->labels_[phase] = labels;
  this->nums_[phase] = rowi;
}



void FeederCsv::create_dbs(const string& db_filename)
{
  int label_index_backup = this->label_;
  std::array<Phase, 3> phases {TEST, TRAIN, GUESS};
  bfs::path basepath {db_filename};
  if(!bfs::exists(basepath)) bfs::create_directory(basepath);

  for(Phase phase: phases)
  {
    if(phase == GUESS) this->label_ = -1;
    if(this->data_source_[phase].compare("") != 0)
    {
      std::vector<std::istream*> streams {this->source_->read(this->data_source_[phase])};
      this->create_db(db_filename + "/" + phaseToString(phase), streams);
      delete streams[0];
    }
    if(phase == GUESS) this->label_ = label_index_backup;
  }
}


void FeederCsv::create_db(const std::string& db_filename, const std::vector<std::istream*>& streams)
{
  uint32_t num_items = streams.size();
  uint32_t rows = 1;                                        // spectrum has one row and many columns
  uint32_t cols = this->data_end_ - this->data_start_ + 1;  // obtain #cols from the config file
  std::string header, line;

  CHECK_GT(num_items, 0) << "No data files provided!";

  CHECK_GT(cols, 0) << "Wrong data.start or data.end setting";
  CHECK_NE(this->label_, this->data_start_) << "Wrong data.label or data.start setting";

  LOG(INFO) << "CSV settings -- delimiter: '" << this->delim_
            << "' label: " << this->label_
            << " datacols: [" << this->data_start_ << ", " << this->data_end_
            << "] header: " << (this->has_header_ ? "yes" : "no");

  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  // Open db
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename << ". Does it already exist?";
  batch = new leveldb::WriteBatch();

  // Storing to db
  string id, value, serialized;
  int label = 0;
  size_t sourcei = 0, rowi = 0, coli = 0, datai = 0, maxdatai = 0;
  float *data = new float[cols];

  caffe::Datum datum;
  datum.set_channels(1);
  datum.set_height(rows);
  datum.set_width(cols);

  std::istringstream linestream;

  for(std::istream* stream: streams) {

    // strip header if set
    if(this->has_header_) {
      std::getline(*stream, header);
      size_t colcount = count(header.begin(), header.end(), this->delim_);
      if(colcount < VD_MAX(this->data_end_, this->label_) ) {
        LOG(WARNING) << "Given source file for " << db_filename << " does not contain any data";
        continue;
      }
    }

    // iterate through all rows in the file
    while(stream->good())
    {
      // parse every file by rows
      std::getline(*stream, line);
      linestream.str(line);
      linestream.clear(); // clear any bad flags

      // iterate through columns and assign them to the right structures
      coli = 0;
      datai = 0;
      datum.clear_float_data();
      while(linestream.good())
      {
        std::getline(linestream, value, this->delim_);
        if(coli == this->label_) {
          // we got the label
          label = atoi(value.c_str());
        } else if (coli >= this->data_start_ && coli <= this->data_end_) {
          // we got data value
          // data[datai] = atof(value.c_str());
          datum.add_float_data(atof(value.c_str()));
          ++datai;
        }
        else if (coli == this->id_) {
          id = value;
        } else {
          // unknown column index
        }
        ++coli;
      }
      // only if we parsed a valid row (with sufficient data in it)
      if(datai == cols) {
        datum.set_label(label);
        // datum.set_data(data, sizeof(*data)*cols);
        datum.SerializeToString(&serialized);
        batch->Put(id, serialized);
        ++rowi;
      } else {
        LOG(INFO) << "Discarting invalid row: " << line;
      }
      maxdatai = VD_MAX(datai, maxdatai);
    }
    ++sourcei;

    // write the data into DB when we are done current file
    db->Write(leveldb::WriteOptions(), batch);
    delete batch;
    batch = new leveldb::WriteBatch();

  }

  LOG(INFO) << "Processed " <<  maxdatai << " data cols and " << rowi << " valid rows in " << sourcei << " files.";

  delete[] data;
  delete batch;
  delete db;
}


}
