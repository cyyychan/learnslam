cmake_minimum_required( VERSION 2.8 )
project( test )

include_directories( "/usr/include/eigen3" )

add_executable( test test.cpp )