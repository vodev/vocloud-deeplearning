
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

#include "feeder_leveldb.hxx"


namespace vodigger {

FeederLevelDB::FeederLevelDB(std::shared_ptr<Source>& source, bpt::ptree& params) : Feeder(source, params)
{
  this->data_source_[TRAIN] = params.get<std::string>("data.train", "");
  this->data_source_[TEST]  = params.get<std::string>("data.test", "");
  this->data_source_[GUESS] = params.get<std::string>("data.guess", "");
}

}
