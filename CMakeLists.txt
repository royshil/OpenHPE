cmake_minimum_required(VERSION 2.6)

PROJECT( OpenHPE_proj )
FIND_PACKAGE( OpenCV REQUIRED )

SET( MY_SOURCES main.cpp bg_fg_blobs.cpp tnc.c simple_tnc.c )
MESSAGE(STATUS "sources: ${MY_SOURCES}")

ADD_EXECUTABLE( OpenHPE_test ${MY_SOURCES} )
TARGET_LINK_LIBRARIES( OpenHPE_test ${OpenCV_LIBS} )