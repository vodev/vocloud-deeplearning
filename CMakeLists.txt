cmake_minimum_required(VERSION 2.8)
project(VO-digger)

# make for target
# cmake -DCMAKE_BUILD_TYPE=Release|Debug|RelWithDebInfo
add_subdirectory(utils)
add_subdirectory(inputs)
add_subdirectory(solvers)

# define outcomes
add_executable(vodigger vodigger.cxx
	$<TARGET_OBJECTS:utils>
	$<TARGET_OBJECTS:sources>
	$<TARGET_OBJECTS:feeders>
	$<TARGET_OBJECTS:solvers>
)
add_dependencies(vodigger utils sources feeders solvers)

# set globally that we compile as c++11
set(VODDIGER_CXX_FLAGS "-std=c++11")

# set that while cmaking as: cmake -DCPU_ONLY=ON .
if(CPU_ONLY)
  set(VODDIGER_CXX_FLAGS "${VODDIGER_CXX_FLAGS} -DCPU_ONLY")
  message("-- Compiling CPU support only (CPU_ONLY specified)")
endif()

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${VODDIGER_CXX_FLAGS} CACHE INTERNAL "cmake_cxx_flags")

# find necessary libraries and link them
target_link_libraries(vodigger m nsl)  # those two libs exists for sure


# more at /usr/share/cmake-2.8/Modules
find_package(Boost REQUIRED program_options system filesystem)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  target_link_libraries(vodigger ${Boost_LIBRARIES})
endif()


find_library(LIB_LEVELDB leveldb)
if(${LIB_LEVELDB} STREQUAL LIB_LEVELDB-NOTFOUND)
	message(FATAL_ERROR "leveldb library not found")
endif()
target_link_libraries(vodigger leveldb)


find_library(LIB_GLOG glog)
if(${LIB_GLOG} STREQUAL LIB_GLOG-NOTFOUND)
	message(FATAL_ERROR "glog library not found")
endif()
target_link_libraries(vodigger glog)


find_library(LIB_CAFFE caffe)
if(${LIB_CAFFE} STREQUAL LIB_CAFFE-NOTFOUND)
	find_library(LIB_CAFFE_NV caffe-nv)
	if(${LIB_CAFFE_NV} STREQUAL LIB_CAFFE_NV-NOTFOUND)
		message(FATAL_ERROR "CAFFE library not found")
	else()
		target_link_libraries(vodigger caffe-nv)
	endif()
else()
	target_link_libraries(vodigger caffe)
endif()


find_library(LIB_PROTOBUF protobuf)
if(${LIB_PROTOBUF} STREQUAL LIB_PROTOBUF-NOTFOUND)
	message(FATAL_ERROR "protobuf library not found")
endif()
target_link_libraries(vodigger protobuf)