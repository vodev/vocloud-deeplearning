cmake_minimum_required(VERSION 2.8)
project(VO-digger)

# make for target
# cmake -DCMAKE_BUILD_TYPE=Release|Debug|RelWithDebInfo

# define outcomes
add_library(sources OBJECT source_folder.cxx source_factory.cxx)
# add_library(feeders OBJECT feeder_csv.cxx feeder_leveldb.cxx feeder_factory.cxx)
