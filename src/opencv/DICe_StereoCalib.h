// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 Sandia Corporation.
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact: Dan Turner (dzturne@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef DICE_STEREOCALIB_H
#define DICE_STEREOCALIB_H

#include <DICe.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>

namespace DICe {

/// pre-processing for CalPreview
///
/// Here's the way the dot cal target dots are located in the image:
/// First a binary threshold is applied to the image.
/// Then the three marker dots (the doughnut dots) are located in the image.
/// Once the three marker dots are located, the closest two, non-colinear regular dots are found for each marker,
/// these dots define the cardinal directions for each marker dot.
/// Next, the origin, xaxis, and yaxis marker dots are assigned in the following way: the origin is the marker dot
/// that is closest in terms of perpendicular distance from the lines created by the cardinal direction dots and the marker dots.
/// That leaves two marker dots to identify as the x and y axis dots. the xaxis dot is the marker dot furthest from the origin.
/// The same process that was used to identify the origin marker dot is used to identify the regular dot that is opposite the origin
/// in the box defined by the three marker dots. These four points are then used to determine the affine transform from a
/// regular grid to the image locations of the dots. The number of dots in x and y is determined by finding all the dots
/// within a tolerance of the x and y axis created by the origin and the other marker dots.
/// \param image_filename the name of the image to search for cal target points in
/// \param output_image_filename the name of the debugging output file that shows the location of the dots
/// \param pre_process_params parameters that define the binary threshold, etc.
/// \param image_points [out] returns the coordinates of the cal dots in image space
/// \param object_points [out] returns the coordinates of the cal dots on the board (physical or model space)
DICE_LIB_DLL_EXPORT
int pre_process_cal_image(const std::string & image_filename,
  const std::string & output_image_filename,
  Teuchos::RCP<Teuchos::ParameterList> pre_process_params,
  std::vector<cv::Point2f> & image_points,
  std::vector<cv::Point3f> & object_points);

/// free function to perform stereo calibration using opencv
/// \param mode grid pattern type
/// mode = 0: checkerboard, 1: dot grid with marker dots (vic3d targets), 2: dot grid, no marker dots
/// \param imagelist the list of images
/// \param board_width the dimensions of the grid
/// \param board_height the dimensions of the grid
/// \param squareSize the spacing dimension
/// \param useCalibrated
/// \param showRectified
/// \param output_filename output file name to use for calibration parameters
DICE_LIB_DLL_EXPORT
float
StereoCalib(const int mode,
  const std::vector<std::string>& imagelist,
  const int board_width,
  const int board_height,
  const float & squareSize,
  const int_t binary_threshold,
  const bool useCalibrated,
  const bool showRectified,
  const std::string & output_filename);

}// End DICe Namespace

#endif
