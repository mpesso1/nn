# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/mason/Neurons

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mason/Neurons/build

# Include any dependencies generated for this target.
include source/CMakeFiles/run.dir/depend.make

# Include the progress variables for this target.
include source/CMakeFiles/run.dir/progress.make

# Include the compile flags for this target's objects.
include source/CMakeFiles/run.dir/flags.make

source/CMakeFiles/run.dir/run.cpp.o: source/CMakeFiles/run.dir/flags.make
source/CMakeFiles/run.dir/run.cpp.o: ../source/run.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mason/Neurons/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object source/CMakeFiles/run.dir/run.cpp.o"
	cd /home/mason/Neurons/build/source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/run.cpp.o -c /home/mason/Neurons/source/run.cpp

source/CMakeFiles/run.dir/run.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/run.cpp.i"
	cd /home/mason/Neurons/build/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mason/Neurons/source/run.cpp > CMakeFiles/run.dir/run.cpp.i

source/CMakeFiles/run.dir/run.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/run.cpp.s"
	cd /home/mason/Neurons/build/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mason/Neurons/source/run.cpp -o CMakeFiles/run.dir/run.cpp.s

source/CMakeFiles/run.dir/run.cpp.o.requires:

.PHONY : source/CMakeFiles/run.dir/run.cpp.o.requires

source/CMakeFiles/run.dir/run.cpp.o.provides: source/CMakeFiles/run.dir/run.cpp.o.requires
	$(MAKE) -f source/CMakeFiles/run.dir/build.make source/CMakeFiles/run.dir/run.cpp.o.provides.build
.PHONY : source/CMakeFiles/run.dir/run.cpp.o.provides

source/CMakeFiles/run.dir/run.cpp.o.provides.build: source/CMakeFiles/run.dir/run.cpp.o


source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o: source/CMakeFiles/run.dir/flags.make
source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o: ../library/communication/communication.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mason/Neurons/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o"
	cd /home/mason/Neurons/build/source && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run.dir/__/library/communication/communication.cpp.o -c /home/mason/Neurons/library/communication/communication.cpp

source/CMakeFiles/run.dir/__/library/communication/communication.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run.dir/__/library/communication/communication.cpp.i"
	cd /home/mason/Neurons/build/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mason/Neurons/library/communication/communication.cpp > CMakeFiles/run.dir/__/library/communication/communication.cpp.i

source/CMakeFiles/run.dir/__/library/communication/communication.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run.dir/__/library/communication/communication.cpp.s"
	cd /home/mason/Neurons/build/source && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mason/Neurons/library/communication/communication.cpp -o CMakeFiles/run.dir/__/library/communication/communication.cpp.s

source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o.requires:

.PHONY : source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o.requires

source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o.provides: source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o.requires
	$(MAKE) -f source/CMakeFiles/run.dir/build.make source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o.provides.build
.PHONY : source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o.provides

source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o.provides.build: source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o


# Object files for target run
run_OBJECTS = \
"CMakeFiles/run.dir/run.cpp.o" \
"CMakeFiles/run.dir/__/library/communication/communication.cpp.o"

# External object files for target run
run_EXTERNAL_OBJECTS =

source/run: source/CMakeFiles/run.dir/run.cpp.o
source/run: source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o
source/run: source/CMakeFiles/run.dir/build.make
source/run: liblibs.a
source/run: /usr/local/lib/libopencv_gapi.so.4.5.4
source/run: /usr/local/lib/libopencv_stitching.so.4.5.4
source/run: /usr/local/lib/libopencv_aruco.so.4.5.4
source/run: /usr/local/lib/libopencv_barcode.so.4.5.4
source/run: /usr/local/lib/libopencv_bgsegm.so.4.5.4
source/run: /usr/local/lib/libopencv_bioinspired.so.4.5.4
source/run: /usr/local/lib/libopencv_ccalib.so.4.5.4
source/run: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.4
source/run: /usr/local/lib/libopencv_dnn_superres.so.4.5.4
source/run: /usr/local/lib/libopencv_dpm.so.4.5.4
source/run: /usr/local/lib/libopencv_face.so.4.5.4
source/run: /usr/local/lib/libopencv_freetype.so.4.5.4
source/run: /usr/local/lib/libopencv_fuzzy.so.4.5.4
source/run: /usr/local/lib/libopencv_hdf.so.4.5.4
source/run: /usr/local/lib/libopencv_hfs.so.4.5.4
source/run: /usr/local/lib/libopencv_img_hash.so.4.5.4
source/run: /usr/local/lib/libopencv_intensity_transform.so.4.5.4
source/run: /usr/local/lib/libopencv_line_descriptor.so.4.5.4
source/run: /usr/local/lib/libopencv_mcc.so.4.5.4
source/run: /usr/local/lib/libopencv_quality.so.4.5.4
source/run: /usr/local/lib/libopencv_rapid.so.4.5.4
source/run: /usr/local/lib/libopencv_reg.so.4.5.4
source/run: /usr/local/lib/libopencv_rgbd.so.4.5.4
source/run: /usr/local/lib/libopencv_saliency.so.4.5.4
source/run: /usr/local/lib/libopencv_stereo.so.4.5.4
source/run: /usr/local/lib/libopencv_structured_light.so.4.5.4
source/run: /usr/local/lib/libopencv_superres.so.4.5.4
source/run: /usr/local/lib/libopencv_surface_matching.so.4.5.4
source/run: /usr/local/lib/libopencv_tracking.so.4.5.4
source/run: /usr/local/lib/libopencv_videostab.so.4.5.4
source/run: /usr/local/lib/libopencv_wechat_qrcode.so.4.5.4
source/run: /usr/local/lib/libopencv_xfeatures2d.so.4.5.4
source/run: /usr/local/lib/libopencv_xobjdetect.so.4.5.4
source/run: /usr/local/lib/libopencv_xphoto.so.4.5.4
source/run: /usr/local/boost_1_80_0/stage/lib/libboost_system.so.1.80.0
source/run: /usr/local/boost_1_80_0/stage/lib/libboost_thread.so.1.80.0
source/run: /usr/local/boost_1_80_0/stage/lib/libboost_regex.so.1.80.0
source/run: /usr/local/lib/libopencv_shape.so.4.5.4
source/run: /usr/local/lib/libopencv_highgui.so.4.5.4
source/run: /usr/local/lib/libopencv_datasets.so.4.5.4
source/run: /usr/local/lib/libopencv_plot.so.4.5.4
source/run: /usr/local/lib/libopencv_text.so.4.5.4
source/run: /usr/local/lib/libopencv_ml.so.4.5.4
source/run: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.4
source/run: /usr/local/lib/libopencv_optflow.so.4.5.4
source/run: /usr/local/lib/libopencv_ximgproc.so.4.5.4
source/run: /usr/local/lib/libopencv_video.so.4.5.4
source/run: /usr/local/lib/libopencv_videoio.so.4.5.4
source/run: /usr/local/lib/libopencv_imgcodecs.so.4.5.4
source/run: /usr/local/lib/libopencv_objdetect.so.4.5.4
source/run: /usr/local/lib/libopencv_calib3d.so.4.5.4
source/run: /usr/local/lib/libopencv_dnn.so.4.5.4
source/run: /usr/local/lib/libopencv_features2d.so.4.5.4
source/run: /usr/local/lib/libopencv_flann.so.4.5.4
source/run: /usr/local/lib/libopencv_photo.so.4.5.4
source/run: /usr/local/lib/libopencv_imgproc.so.4.5.4
source/run: /usr/local/lib/libopencv_core.so.4.5.4
source/run: source/CMakeFiles/run.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mason/Neurons/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable run"
	cd /home/mason/Neurons/build/source && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
source/CMakeFiles/run.dir/build: source/run

.PHONY : source/CMakeFiles/run.dir/build

source/CMakeFiles/run.dir/requires: source/CMakeFiles/run.dir/run.cpp.o.requires
source/CMakeFiles/run.dir/requires: source/CMakeFiles/run.dir/__/library/communication/communication.cpp.o.requires

.PHONY : source/CMakeFiles/run.dir/requires

source/CMakeFiles/run.dir/clean:
	cd /home/mason/Neurons/build/source && $(CMAKE_COMMAND) -P CMakeFiles/run.dir/cmake_clean.cmake
.PHONY : source/CMakeFiles/run.dir/clean

source/CMakeFiles/run.dir/depend:
	cd /home/mason/Neurons/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mason/Neurons /home/mason/Neurons/source /home/mason/Neurons/build /home/mason/Neurons/build/source /home/mason/Neurons/build/source/CMakeFiles/run.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : source/CMakeFiles/run.dir/depend

