# Install script for directory: /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/core/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/IRL/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/miscellaneous/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/MultiHeat/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/policy/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Rocky/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/solvers/cmake_install.cmake")
  include("/home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD/cmake_install.cmake")

endif()

