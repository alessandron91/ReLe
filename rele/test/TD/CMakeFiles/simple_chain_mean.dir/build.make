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
include test/TD/CMakeFiles/simple_chain_mean.dir/depend.make

# Include the progress variables for this target.
include test/TD/CMakeFiles/simple_chain_mean.dir/progress.make

# Include the compile flags for this target's objects.
include test/TD/CMakeFiles/simple_chain_mean.dir/flags.make

test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o: test/TD/CMakeFiles/simple_chain_mean.dir/flags.make
test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o: test/TD/SimpleChainMeanReward.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o -c /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD/SimpleChainMeanReward.cpp

test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.i"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD/SimpleChainMeanReward.cpp > CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.i

test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.s"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD/SimpleChainMeanReward.cpp -o CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.s

test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o.requires:

.PHONY : test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o.requires

test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o.provides: test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o.requires
	$(MAKE) -f test/TD/CMakeFiles/simple_chain_mean.dir/build.make test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o.provides.build
.PHONY : test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o.provides

test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o.provides.build: test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o


# Object files for target simple_chain_mean
simple_chain_mean_OBJECTS = \
"CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o"

# External object files for target simple_chain_mean
simple_chain_mean_EXTERNAL_OBJECTS =

test/TD/simple_chain_mean: test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o
test/TD/simple_chain_mean: test/TD/CMakeFiles/simple_chain_mean.dir/build.make
test/TD/simple_chain_mean: librele.a
test/TD/simple_chain_mean: /usr/lib/libarmadillo.so
test/TD/simple_chain_mean: /usr/lib/x86_64-linux-gnu/libnlopt.so
test/TD/simple_chain_mean: /usr/lib/x86_64-linux-gnu/libboost_system.so
test/TD/simple_chain_mean: /usr/lib/x86_64-linux-gnu/libboost_timer.so
test/TD/simple_chain_mean: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
test/TD/simple_chain_mean: test/TD/CMakeFiles/simple_chain_mean.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable simple_chain_mean"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple_chain_mean.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/TD/CMakeFiles/simple_chain_mean.dir/build: test/TD/simple_chain_mean

.PHONY : test/TD/CMakeFiles/simple_chain_mean.dir/build

test/TD/CMakeFiles/simple_chain_mean.dir/requires: test/TD/CMakeFiles/simple_chain_mean.dir/SimpleChainMeanReward.cpp.o.requires

.PHONY : test/TD/CMakeFiles/simple_chain_mean.dir/requires

test/TD/CMakeFiles/simple_chain_mean.dir/clean:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD && $(CMAKE_COMMAND) -P CMakeFiles/simple_chain_mean.dir/cmake_clean.cmake
.PHONY : test/TD/CMakeFiles/simple_chain_mean.dir/clean

test/TD/CMakeFiles/simple_chain_mean.dir/depend:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/TD/CMakeFiles/simple_chain_mean.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/TD/CMakeFiles/simple_chain_mean.dir/depend
