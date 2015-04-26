
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

#include <fstream>  // NOLINT(readability/streams)
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

#include <fitsio.h>

#include "../utils/string.hxx"


void FeederFits::create_db(const string& db_filename) {

  google::CHECK(bfs::exists(bfs::path(db_filename))) << "DB file already exists!"

  SourceFolder *sf = dynamic_cast<SourceFolder>(this->source_->get());
  google::CHECK_NOTNULL(sf) << "FITS can work only with files in folders!"

  vector<string> datafiles( self->source_->files().size() - 1 );

  std::copy_if(self->source_->files().begin(), self->source_->files().end(),
    std::back_inserter(datafiles), [](const string& fn) {return endswith(fn, "fits");});

  uint32_t num_items = datafiles.size();
  uint32_t rows = 1;  // spectrum has one row and many columns
  uint32_t cols = 0;  // unknown yet 
  uint32_t status;    // only for fitsio purposes

  fitsfile *sample = nullptr;
  fits_open_data(&sample, sample);
  cols = fits_get_num_rows(&sample, &cols, &status);
  fits_close(sample);

  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;

  // Open db
  LOG(INFO) << "Opening leveldb " << db_filename;
  leveldb::Status status = leveldb::DB::Open(
      options, db_filename, &db);
  CHECK(status.ok()) << "Failed to open leveldb " << db_filename
      << ". Is it already existing?";
  batch = new leveldb::WriteBatch();
  
  for(const string& filename: self->source_->files()) {
    // don't process non-fots files
    if( !endswith(filename, "fits") || !endswith(filename, "fit") ) continue;


  }


  // Storing to db
  char label;
  char* pixels = new char[rows * cols];
  int count = 0;
  const int kMaxKeyLength = 10;
  char key_cstr[kMaxKeyLength];
  string value;

  Datum datum;
  datum.set_channels(1);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    image_file.read(pixels, rows * cols);
    label_file.read(&label, 1);
    datum.set_data(pixels, rows*cols);
    datum.set_label(label);
    snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
    datum.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    batch->Put(keystr, value);

    if (++count % 1000 == 0) {
      // Commit txn
      db->Write(leveldb::WriteOptions(), batch);
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    delete batch;
    delete db;
    LOG(ERROR) << "Processed " << count << " files.";
  }
  delete pixels;
}

int main(int argc, char** argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("This script converts the MNIST dataset to\n"
        "the lmdb/leveldb format used by Caffe to load data.\n"
        "Usage:\n"
        "    convert_mnist_data [FLAGS] input_image_file input_label_file "
        "output_db_file\n"
        "The MNIST dataset could be downloaded at\n"
        "    http://yann.lecun.com/exdb/mnist/\n"
        "You should gunzip them after downloading,"
        "or directly use data/mnist/get_mnist.sh\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const string& db_backend = FLAGS_backend;

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/mnist/convert_mnist_data");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(argv[1], argv[2], argv[3], db_backend);
  }
  return 0;
}
