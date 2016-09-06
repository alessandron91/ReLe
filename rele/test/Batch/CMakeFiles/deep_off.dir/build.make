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
include test/Batch/CMakeFiles/deep_off.dir/depend.make

# Include the progress variables for this target.
include test/Batch/CMakeFiles/deep_off.dir/progress.make

# Include the compile flags for this target's objects.
include test/Batch/CMakeFiles/deep_off.dir/flags.make

test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o: test/Batch/CMakeFiles/deep_off.dir/flags.make
test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o: test/Batch/DeepBatchTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o -c /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch/DeepBatchTest.cpp

test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deep_off.dir/DeepBatchTest.cpp.i"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch/DeepBatchTest.cpp > CMakeFiles/deep_off.dir/DeepBatchTest.cpp.i

test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deep_off.dir/DeepBatchTest.cpp.s"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch/DeepBatchTest.cpp -o CMakeFiles/deep_off.dir/DeepBatchTest.cpp.s

test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o.requires:

.PHONY : test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o.requires

test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o.provides: test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o.requires
	$(MAKE) -f test/Batch/CMakeFiles/deep_off.dir/build.make test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o.provides.build
.PHONY : test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o.provides

test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o.provides.build: test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o


# Object files for target deep_off
deep_off_OBJECTS = \
"CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o"

# External object files for target deep_off
deep_off_EXTERNAL_OBJECTS =

test/Batch/deep_off: test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o
test/Batch/deep_off: test/Batch/CMakeFiles/deep_off.dir/build.make
test/Batch/deep_off: librele.a
test/Batch/deep_off: /usr/lib/libarmadillo.so
test/Batch/deep_off: /usr/lib/x86_64-linux-gnu/libnlopt.so
test/Batch/deep_off: /usr/lib/x86_64-linux-gnu/libboost_system.so
test/Batch/deep_off: /usr/lib/x86_64-linux-gnu/libboost_timer.so
test/Batch/deep_off: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
test/Batch/deep_off: test/Batch/CMakeFiles/deep_off.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable deep_off"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/deep_off.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/Batch/CMakeFiles/deep_off.dir/build: test/Batch/deep_off

.PHONY : test/Batch/CMakeFiles/deep_off.dir/build

test/Batch/CMakeFiles/deep_off.dir/requires: test/Batch/CMakeFiles/deep_off.dir/DeepBatchTest.cpp.o.requires

.PHONY : test/Batch/CMakeFiles/deep_off.dir/requires

test/Batch/CMakeFiles/deep_off.dir/clean:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch && $(CMAKE_COMMAND) -P CMakeFiles/deep_off.dir/cmake_clean.cmake
.PHONY : test/Batch/CMakeFiles/deep_off.dir/clean

test/Batch/CMakeFiles/deep_off.dir/depend:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/Batch/CMakeFiles/deep_off.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/Batch/CMakeFiles/deep_off.dir/depend
