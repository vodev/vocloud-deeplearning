cmake_minimum_required(VERSION 2.8)
project(VO-digger)

# define outcomes
add_library(solvers OBJECT solver_caffe.cxx solver_caffe_sdg.cxx)
