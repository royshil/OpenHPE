cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

SET( MY_SOURCES main.cpp bg_fg_blobs.cpp tnc.c simple_tnc.c )
ADD_EXECUTABLE( OpenHPE_test ${MY_SOURCES} )

PROJECT( OpenHPE_proj )
FIND_PACKAGE( OpenCV REQUIRED )

IF(OpenCV_FOUND)
	MESSAGE(STATUS "opencv found")
	set(LIBS ${LIBS} ${OpenCV_LIBS})
ENDIF(OpenCV_FOUND)

FIND_PACKAGE(freenect)
IF(freenect_FOUND)
	MESSAGE(STATUS "freenect found")
	set(LIBS ${LIBS} ${freenect_LIBS} )
ENDIF(freenect_FOUND)


MESSAGE(STATUS "sources: ${MY_SOURCES}")
MESSAGE(STATUS "libs: ${LIBS}")

TARGET_LINK_LIBRARIES( OpenHPE_test ${LIBS} )
