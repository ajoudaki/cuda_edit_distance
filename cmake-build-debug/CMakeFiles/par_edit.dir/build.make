# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /home/amir/Software/clion-2020.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/amir/Software/clion-2020.3/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/amir/CLionProjects/cuda_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/amir/CLionProjects/cuda_test/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/par_edit.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/par_edit.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/par_edit.dir/flags.make

CMakeFiles/par_edit.dir/par_ed.cu.o: CMakeFiles/par_edit.dir/flags.make
CMakeFiles/par_edit.dir/par_ed.cu.o: ../par_ed.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/amir/CLionProjects/cuda_test/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/par_edit.dir/par_ed.cu.o"
	/usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/amir/CLionProjects/cuda_test/par_ed.cu -o CMakeFiles/par_edit.dir/par_ed.cu.o

CMakeFiles/par_edit.dir/par_ed.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/par_edit.dir/par_ed.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/par_edit.dir/par_ed.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/par_edit.dir/par_ed.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target par_edit
par_edit_OBJECTS = \
"CMakeFiles/par_edit.dir/par_ed.cu.o"

# External object files for target par_edit
par_edit_EXTERNAL_OBJECTS =

par_edit: CMakeFiles/par_edit.dir/par_ed.cu.o
par_edit: CMakeFiles/par_edit.dir/build.make
par_edit: CMakeFiles/par_edit.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/amir/CLionProjects/cuda_test/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable par_edit"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/par_edit.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/par_edit.dir/build: par_edit

.PHONY : CMakeFiles/par_edit.dir/build

CMakeFiles/par_edit.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/par_edit.dir/cmake_clean.cmake
.PHONY : CMakeFiles/par_edit.dir/clean

CMakeFiles/par_edit.dir/depend:
	cd /home/amir/CLionProjects/cuda_test/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/amir/CLionProjects/cuda_test /home/amir/CLionProjects/cuda_test /home/amir/CLionProjects/cuda_test/cmake-build-debug /home/amir/CLionProjects/cuda_test/cmake-build-debug /home/amir/CLionProjects/cuda_test/cmake-build-debug/CMakeFiles/par_edit.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/par_edit.dir/depend

