// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#ifndef DICE_OPENCVSERVERUTILS_H
#define DICE_OPENCVSERVERUTILS_H


#include <DICe.h>

#include <Teuchos_ParameterList.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace DICe{

/// String parameter name
const char* const opencv_server_io_files = "io_files";
const char* const opencv_server_filters = "filters";

/// Filters
const char* const opencv_server_filter_cal_preview = "cal_preview";
const char* const opencv_server_filter_adaptive_threshold = "adaptive_threshold";
const char* const opencv_server_filter_binary_threshold = "binary_threshold";
const char* const opencv_server_filter_checkerboard_targets = "checkerboard_targets";
const char* const opencv_server_filter_dot_targets = "dot_targets";
const char* const opencv_server_filter_epipolar_line = "epipolar_line";
const char* const opencv_server_filter_tracklib = "tracklib";
const char* const opencv_server_filter_none = "none"; // used to display the original image with no filter
const char* const opencv_server_filter_brightness = "brightness";
const char* const opencv_server_filter_equalize_hist = "equalize_hist";
const char* const opencv_server_filter_background = "background"; // create a background image to subtract from subsequent frames

/// Filter parameters
const char* const opencv_server_filter_mode = "filter_mode";
const char* const opencv_server_preview_threshold = "preview_threshold";
const char* const opencv_server_threshold_mode = "threshold_mode";
const char* const opencv_server_threshold_start = "threshold_start";
const char* const opencv_server_threshold_end = "threshold_end";
const char* const opencv_server_threshold_step = "threshold_step";
const char* const opencv_server_block_size = "block_size";
const char* const opencv_server_min_blob_size = "min_blob_size";
const char* const opencv_server_binary_constant = "binary_constant";
const char* const opencv_server_dot_tol = "dot_tol";
const char* const opencv_server_use_adaptive_threshold = "use_adaptive_threshold";
const char* const opencv_server_epipolar_dot_x = "epipolar_dot_x";
const char* const opencv_server_epipolar_dot_y = "epipolar_dot_y";
const char* const opencv_server_epipolar_is_left = "epipolar_is_left";
const char* const opencv_server_cal_file = "cal_file";
const char* const opencv_server_cine_file_name = "cine_file_name";
const char* const opencv_server_background_file_name = "background_file_name";
const char* const opencv_server_background_ref_frame = "background_ref_frame";
const char* const opencv_server_background_num_frames = "background_num_frames";

/// parse the input string and return a Teuchos ParameterList
DICE_LIB_DLL_EXPORT
Teuchos::ParameterList parse_filter_string(int argc, char *argv[]);

/// run open CV routines using an array of char* input to determine which filters, etc to activate
DICE_LIB_DLL_EXPORT
int_t opencv_server(int argc, char *argv[]);

// filter routines

DICE_LIB_DLL_EXPORT
int_t opencv_binary_threshold(cv::Mat & img,Teuchos::ParameterList & options);
DICE_LIB_DLL_EXPORT
int_t opencv_adaptive_threshold(cv::Mat & img, Teuchos::ParameterList & options);
DICE_LIB_DLL_EXPORT
int_t opencv_checkerboard_targets(cv::Mat & img, Teuchos::ParameterList & options);
// overload that returns the located corner points
DICE_LIB_DLL_EXPORT
int_t opencv_checkerboard_targets(cv::Mat & img, Teuchos::ParameterList & options,
  std::vector<cv::Point2f> & corners);
DICE_LIB_DLL_EXPORT
int_t opencv_dot_targets(cv::Mat & img,
  Teuchos::ParameterList & options,
  std::vector<cv::KeyPoint> & key_points,
  std::vector<cv::KeyPoint> & img_points,
  std::vector<cv::KeyPoint> & grd_points,
  int_t & return_thresh);
// overload that returns the keypoints found in the image
DICE_LIB_DLL_EXPORT
int_t opencv_dot_targets(cv::Mat & img,
  Teuchos::ParameterList & options,
  int_t & return_thresh);

DICE_LIB_DLL_EXPORT
int_t opencv_epipolar_line(cv::Mat & img,
  Teuchos::ParameterList & options,
  const bool first_image);

DICE_LIB_DLL_EXPORT
int_t opencv_create_cine_background_image(Teuchos::ParameterList & options);

// utilities

/// find all the dot markers in the image
void get_dot_markers(cv::Mat img,
  std::vector<cv::KeyPoint> & keypoints,
  int_t thresh,
  bool invert,
  Teuchos::ParameterList & options,
  const int min_size=100);
/// calculates the image to grid and grid to images coefficients based on the current set of good points
void calc_trans_coeff(std::vector<cv::KeyPoint> & imgpoints,
  std::vector<cv::KeyPoint> & grdpoints,
  std::vector<scalar_t> & img_to_grdx,
  std::vector<scalar_t> & img_to_grdy,
  std::vector<scalar_t> & grd_to_imgx,
  std::vector<scalar_t> & grd_to_imgy);
/// filters the possible dots by size, the bounding box and how close they are to the expected grid locations
void filter_dot_markers(std::vector<cv::KeyPoint> dots,
  std::vector<cv::KeyPoint> & img_points,
  std::vector<cv::KeyPoint> & grd_points,
  const std::vector<scalar_t> & grd_to_imgx,
  const std::vector<scalar_t> & grd_to_imgy,
  const std::vector<scalar_t> & img_to_grdx,
  const std::vector<scalar_t> & img_to_grdy,
  const int_t num_fiducials_x,
  const int_t num_fiducials_y,
  float dot_tol,
  cv::Mat img,
  bool draw);
/// reorders the keypoints into origin, x dot, y dot based on the distances between them
void reorder_keypoints(std::vector<cv::KeyPoint> & keypoints,const std::vector<cv::KeyPoint> & dots);
/// creates a quadrilateral that describes the allowable area for valid dots
void create_bounding_box(std::vector<float> & box_x,
  std::vector<float> & box_y,
  const int_t num_fiducials_x,
  const int_t num_fiducials_y,
  const std::vector<scalar_t> & grd_to_imgx,
  const std::vector<scalar_t> & grd_to_imgy,
  const int_t img_w,
  const int_t img_h);
//convert grid locations to image locations
void grid_to_image(const float & grid_x,
  const float & grid_y,
  float & img_x,
  float & img_y,
  const std::vector<scalar_t> & grd_to_imgx,
  const std::vector<scalar_t> & grd_to_imgy,
  const int_t img_w,
  const int_t img_h);
//convert grid locations to image locations
void image_to_grid(const float & img_x,
  const float & img_y,
  float & grid_x,
  float & grid_y,
  const std::vector<scalar_t> & img_to_grdx,
  const std::vector<scalar_t> & img_to_grdy);
/// returns the squared distance between two keypoints
float dist2(cv::KeyPoint pnt1, cv::KeyPoint pnt2);
/// returns indicies of three distances sorted by decending magnitude
void order_dist3(std::vector<float> & dist,
  std::vector<int> & dist_order);
/// checks if a point is in the bounding box quadrilateral
bool is_in_quadrilateral(const float & x,
  const float & y,
  const std::vector<float> & box_x,
  const std::vector<float> & box_y);

} // end DICe namespace

#endif
