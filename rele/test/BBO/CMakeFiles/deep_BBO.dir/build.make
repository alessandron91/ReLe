# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alessandro/Scrivania/ReLe/src/ReLe/rele

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alessandro/Scrivania/ReLe/src/ReLe/rele

# Include any dependencies generated for this target.
include test/BBO/CMakeFiles/deep_BBO.dir/depend.make

# Include the progress variables for this target.
include test/BBO/CMakeFiles/deep_BBO.dir/progress.make

# Include the compile flags for this target's objects.
include test/BBO/CMakeFiles/deep_BBO.dir/flags.make

test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o: test/BBO/CMakeFiles/deep_BBO.dir/flags.make
test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o: test/BBO/DeepBBOTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o -c /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO/DeepBBOTest.cpp

test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.i"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO/DeepBBOTest.cpp > CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.i

test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.s"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO/DeepBBOTest.cpp -o CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.s

test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o.requires:

.PHONY : test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o.requires

test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o.provides: test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o.requires
	$(MAKE) -f test/BBO/CMakeFiles/deep_BBO.dir/build.make test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o.provides.build
.PHONY : test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o.provides

test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o.provides.build: test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o


# Object files for target deep_BBO
deep_BBO_OBJECTS = \
"CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o"

# External object files for target deep_BBO
deep_BBO_EXTERNAL_OBJECTS =

test/BBO/deep_BBO: test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o
test/BBO/deep_BBO: test/BBO/CMakeFiles/deep_BBO.dir/build.make
test/BBO/deep_BBO: librele.a
test/BBO/deep_BBO: /usr/lib/libarmadillo.so
test/BBO/deep_BBO: /usr/lib/x86_64-linux-gnu/libnlopt.so
test/BBO/deep_BBO: /usr/lib/x86_64-linux-gnu/libboost_system.so
test/BBO/deep_BBO: /usr/lib/x86_64-linux-gnu/libboost_timer.so
test/BBO/deep_BBO: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
test/BBO/deep_BBO: test/BBO/CMakeFiles/deep_BBO.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable deep_BBO"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/deep_BBO.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/BBO/CMakeFiles/deep_BBO.dir/build: test/BBO/deep_BBO

.PHONY : test/BBO/CMakeFiles/deep_BBO.dir/build

test/BBO/CMakeFiles/deep_BBO.dir/requires: test/BBO/CMakeFiles/deep_BBO.dir/DeepBBOTest.cpp.o.requires

.PHONY : test/BBO/CMakeFiles/deep_BBO.dir/requires

test/BBO/CMakeFiles/deep_BBO.dir/clean:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO && $(CMAKE_COMMAND) -P CMakeFiles/deep_BBO.dir/cmake_clean.cmake
.PHONY : test/BBO/CMakeFiles/deep_BBO.dir/clean

test/BBO/CMakeFiles/deep_BBO.dir/depend:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/BBO/CMakeFiles/deep_BBO.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/BBO/CMakeFiles/deep_BBO.dir/depend

