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
include test/PG/CMakeFiles/pursuer_PG.dir/depend.make

# Include the progress variables for this target.
include test/PG/CMakeFiles/pursuer_PG.dir/progress.make

# Include the compile flags for this target's objects.
include test/PG/CMakeFiles/pursuer_PG.dir/flags.make

test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o: test/PG/CMakeFiles/pursuer_PG.dir/flags.make
test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o: test/PG/PursuerPGTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o -c /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/PursuerPGTest.cpp

test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.i"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/PursuerPGTest.cpp > CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.i

test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.s"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/PursuerPGTest.cpp -o CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.s

test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o.requires:

.PHONY : test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o.requires

test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o.provides: test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o.requires
	$(MAKE) -f test/PG/CMakeFiles/pursuer_PG.dir/build.make test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o.provides.build
.PHONY : test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o.provides

test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o.provides.build: test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o


# Object files for target pursuer_PG
pursuer_PG_OBJECTS = \
"CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o"

# External object files for target pursuer_PG
pursuer_PG_EXTERNAL_OBJECTS =

test/PG/pursuer_PG: test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o
test/PG/pursuer_PG: test/PG/CMakeFiles/pursuer_PG.dir/build.make
test/PG/pursuer_PG: librele.a
test/PG/pursuer_PG: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
test/PG/pursuer_PG: /usr/lib/libarmadillo.so
test/PG/pursuer_PG: /usr/lib/x86_64-linux-gnu/libnlopt.so
test/PG/pursuer_PG: /usr/lib/x86_64-linux-gnu/libboost_system.so
test/PG/pursuer_PG: /usr/lib/x86_64-linux-gnu/libboost_timer.so
test/PG/pursuer_PG: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
test/PG/pursuer_PG: test/PG/CMakeFiles/pursuer_PG.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alessandro/Scrivania/ReLe/src/ReLe/rele/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pursuer_PG"
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pursuer_PG.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/PG/CMakeFiles/pursuer_PG.dir/build: test/PG/pursuer_PG

.PHONY : test/PG/CMakeFiles/pursuer_PG.dir/build

test/PG/CMakeFiles/pursuer_PG.dir/requires: test/PG/CMakeFiles/pursuer_PG.dir/PursuerPGTest.cpp.o.requires

.PHONY : test/PG/CMakeFiles/pursuer_PG.dir/requires

test/PG/CMakeFiles/pursuer_PG.dir/clean:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG && $(CMAKE_COMMAND) -P CMakeFiles/pursuer_PG.dir/cmake_clean.cmake
.PHONY : test/PG/CMakeFiles/pursuer_PG.dir/clean

test/PG/CMakeFiles/pursuer_PG.dir/depend:
	cd /home/alessandro/Scrivania/ReLe/src/ReLe/rele && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG /home/alessandro/Scrivania/ReLe/src/ReLe/rele /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG /home/alessandro/Scrivania/ReLe/src/ReLe/rele/test/PG/CMakeFiles/pursuer_PG.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/PG/CMakeFiles/pursuer_PG.dir/depend

