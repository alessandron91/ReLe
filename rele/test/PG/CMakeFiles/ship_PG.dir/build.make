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
include test/PG/CMakeFiles/ship_PG.dir/depend.make

# Include the progress variables for this target.
include test/PG/CMakeFiles/ship_PG.dir/progress.make

# Include the compile flags for this target's objects.
include test/PG/CMakeFiles/ship_PG.dir/flags.make

test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o: test/PG/CMakeFiles/ship_PG.dir/flags.make
test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o: test/PG/ShipSteeringPGTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o -c /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/ShipSteeringPGTest.cpp

test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.i"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/ShipSteeringPGTest.cpp > CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.i

test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.s"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/ShipSteeringPGTest.cpp -o CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.s

test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o.requires:

.PHONY : test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o.requires

test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o.provides: test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o.requires
	$(MAKE) -f test/PG/CMakeFiles/ship_PG.dir/build.make test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o.provides.build
.PHONY : test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o.provides

test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o.provides.build: test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o


# Object files for target ship_PG
ship_PG_OBJECTS = \
"CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o"

# External object files for target ship_PG
ship_PG_EXTERNAL_OBJECTS =

test/PG/ship_PG: test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o
test/PG/ship_PG: test/PG/CMakeFiles/ship_PG.dir/build.make
test/PG/ship_PG: librele.a
test/PG/ship_PG: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
test/PG/ship_PG: /usr/lib/libarmadillo.so
test/PG/ship_PG: /usr/lib/x86_64-linux-gnu/libnlopt.so
test/PG/ship_PG: /usr/lib/x86_64-linux-gnu/libboost_system.so
test/PG/ship_PG: /usr/lib/x86_64-linux-gnu/libboost_timer.so
test/PG/ship_PG: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
test/PG/ship_PG: test/PG/CMakeFiles/ship_PG.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ship_PG"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ship_PG.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/PG/CMakeFiles/ship_PG.dir/build: test/PG/ship_PG

.PHONY : test/PG/CMakeFiles/ship_PG.dir/build

test/PG/CMakeFiles/ship_PG.dir/requires: test/PG/CMakeFiles/ship_PG.dir/ShipSteeringPGTest.cpp.o.requires

.PHONY : test/PG/CMakeFiles/ship_PG.dir/requires

test/PG/CMakeFiles/ship_PG.dir/clean:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && $(CMAKE_COMMAND) -P CMakeFiles/ship_PG.dir/cmake_clean.cmake
.PHONY : test/PG/CMakeFiles/ship_PG.dir/clean

test/PG/CMakeFiles/ship_PG.dir/depend:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/CMakeFiles/ship_PG.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/PG/CMakeFiles/ship_PG.dir/depend

