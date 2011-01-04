# - Try to find Freenect
# Once done, this will define
#
#  freenect_FOUND - system has freenect
#  freenect_INCLUDE_DIRS - the freenect include directories
#  freenect_LIBRARIES - link these to use freenect

include(LibFindMacros)

MESSAGE(STATUS "Try to find freenect")

# Dependencies
#libfind_package(libfreenect freenect)

MESSAGE(STATUS "Try using pkg-config")

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(freenect_PKGCONF freenect)

MESSAGE(STATUS "pkg-config found: ${freenect_PKGCONF_INCLUDE_DIRS}")

# Include dir
find_path(freenect_INCLUDE_DIR
  NAMES libfreenect.h
  PATHS ${freenect_PKGCONF_INCLUDE_DIRS} /usr/local/include/
)

MESSAGE(STATUS "freenect includes found: ${freenect_INCLUDE_DIR}")

# Finally the library itself
find_library(freenect_LIBRARY
  NAMES freenect
  PATHS ${freenect_PKGCONF_LIBRARY_DIRS} /usr/local/lib/
)

MESSAGE(STATUS "freenect libs found: ${freenect_LIBRARY}")

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(freenect_PROCESS_INCLUDES freenect_INCLUDE_DIR)
set(freenect_PROCESS_LIBS freenect_LIBRARY)
libfind_process(freenect)