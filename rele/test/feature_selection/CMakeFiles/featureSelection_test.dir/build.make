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
include test/feature_selection/CMakeFiles/featureSelection_test.dir/depend.make

# Include the progress variables for this target.
include test/feature_selection/CMakeFiles/featureSelection_test.dir/progress.make

# Include the compile flags for this target's objects.
include test/feature_selection/CMakeFiles/featureSelection_test.dir/flags.make

test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o: test/feature_selection/CMakeFiles/featureSelection_test.dir/flags.make
test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o: test/feature_selection/FeatureSelectionTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o -c /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection/FeatureSelectionTest.cpp

test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.i"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection/FeatureSelectionTest.cpp > CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.i

test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.s"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection/FeatureSelectionTest.cpp -o CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.s

test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o.requires:

.PHONY : test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o.requires

test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o.provides: test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o.requires
	$(MAKE) -f test/feature_selection/CMakeFiles/featureSelection_test.dir/build.make test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o.provides.build
.PHONY : test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o.provides

test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o.provides.build: test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o


# Object files for target featureSelection_test
featureSelection_test_OBJECTS = \
"CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o"

# External object files for target featureSelection_test
featureSelection_test_EXTERNAL_OBJECTS =

test/feature_selection/featureSelection_test: test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o
test/feature_selection/featureSelection_test: test/feature_selection/CMakeFiles/featureSelection_test.dir/build.make
test/feature_selection/featureSelection_test: librele.a
test/feature_selection/featureSelection_test: /usr/lib/libarmadillo.so
test/feature_selection/featureSelection_test: /usr/lib/x86_64-linux-gnu/libnlopt.so
test/feature_selection/featureSelection_test: /usr/lib/x86_64-linux-gnu/libboost_system.so
test/feature_selection/featureSelection_test: /usr/lib/x86_64-linux-gnu/libboost_timer.so
test/feature_selection/featureSelection_test: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
test/feature_selection/featureSelection_test: test/feature_selection/CMakeFiles/featureSelection_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable featureSelection_test"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/featureSelection_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/feature_selection/CMakeFiles/featureSelection_test.dir/build: test/feature_selection/featureSelection_test

.PHONY : test/feature_selection/CMakeFiles/featureSelection_test.dir/build

test/feature_selection/CMakeFiles/featureSelection_test.dir/requires: test/feature_selection/CMakeFiles/featureSelection_test.dir/FeatureSelectionTest.cpp.o.requires

.PHONY : test/feature_selection/CMakeFiles/featureSelection_test.dir/requires

test/feature_selection/CMakeFiles/featureSelection_test.dir/clean:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection && $(CMAKE_COMMAND) -P CMakeFiles/featureSelection_test.dir/cmake_clean.cmake
.PHONY : test/feature_selection/CMakeFiles/featureSelection_test.dir/clean

test/feature_selection/CMakeFiles/featureSelection_test.dir/depend:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/feature_selection/CMakeFiles/featureSelection_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/feature_selection/CMakeFiles/featureSelection_test.dir/depend
