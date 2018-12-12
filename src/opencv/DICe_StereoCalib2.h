#pragma once
// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

#ifndef DICE_STEREO_CALIB2_H
#define DICE_STEREO_CALIB2_H

#include <DICe.h>
#include "DICe_ParameterUtilities.h"

#include <DICe_Shape.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <DICe_XMLUtils.h>
#include <Teuchos_Array.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <DICe_CamSystem.h>

#include <vector>

/// \namespace holds the enumerated values and string representations for StereoCalib2 members
namespace DICe_StereoCalib2{
  /// types of calibration targets that may be used
  enum Target_Type {
    CHECKER_BOARD = 0,
    BLACK_ON_WHITE_W_DONUT_DOTS,
    WHITE_ON_BLACK_W_DONUT_DOTS,
    //DON"T ADD ANY BELOW MAX
    MAX_TARGET_TYPE_,
    NO_SUCH_TARGET_TYPE
  };

  const static char * targetTypeStrings[] = {
    "CHECKER_BOARD",
    "BLACK_ON_WHITE_W_DONUT_DOTS",
    "WHITE_ON_BLACK_W_DONUT_DOTS"
  };

  const std::string intersection_file_ID_ = "DICe_XML_Intersection_File"; //string designating a valid intersection file

}



namespace DICe {
  /// \class DICe::StereoCalib2
  /// \brief A class for processing calibration target images and determining camera calibration parameters
  ///
  /// This class uses the OpenCV calibration routine and is currently limited to two cameras
  /// Results from the calibration are sent to a CamSystem object. If one is not supplied the class
  /// uses an internally generated one. CamSystem is also used to write the calibration files.
  /// The extraction routine supports dot targets with 3 "donut" marker dots to specify the axes and
  /// checkerboard calibration targets.

  class DICE_LIB_DLL_EXPORT
    StereoCalib2 {
  public:
    /// \brief constructor with no arguments. It sets the default calibration options and no output to the console
    StereoCalib2() {
      set_Calib_Options();
      verbose_ = false;
    };

    /// \brief constructor with all the arguments needed to run a calibration
    /// \param image_list a list of the calibration target images in [camera][image] order
    /// \param directory the directory with the calibration images.
    /// \param calib_file name of the calibration XML file to write the results (with directory)
    /// \param target type of target used for calibration. the valid values are enumerated in the DICe_StereoCalib2 namespace
    /// \param num_points_x the total number of intersections/dots in the x direction of the calibration target
    /// \param num_points_y the total number of intersections/dots in the y direction of the calibration target
    /// \param point_spacing distance between the intersections/dots. World coordinates will be specified in the units given for this parameter
    /// \param orig_loc_x Optional the location of the origin marker point in the x direction measured by the number points from the left most point (pt 0)
    /// \param orig_loc_y Optional the location of the origin marker point in the y direction measured by the number points from the bottom most point (pt 0)
    /// \param marker_points_x Optional number of spaces from the origin to the x axis marker point
    /// \param marker_points_y Optional number of spaces from the origin to the y axis marker point
    /// \param options Optional the integer value of the combined options if negative the calibration uses the default options
    /// orig_loc_x/y and marker_points_x/y are only needed for the dot patterns and should be set to 0 for the checkerboard
    /// if the origin marker is in the lower left corner of the rectangle of points the orig_loc_x/y is 0,0
    /// if the marker_points number is not specified it is assumed that the marker points ar at the corners of the rectangle
    /// the default options are: use_intrinsic_guess and zero_tangent_dist 
    StereoCalib2(const std::vector<std::vector<std::string>> image_list, const std::string directory, std::string calib_file,
      const DICe_StereoCalib2::Target_Type target, const int num_points_x, const int num_points_y, const double point_spacing,
      const int orig_loc_x = 0, const int orig_loc_y = 0, const int marker_points_x = 0, const int marker_points_y = 0, const int options = -1){
      verbose_ = false;
      set_Image_List(image_list, directory);
      set_Target_Info(target, num_points_x, num_points_y, point_spacing, orig_loc_x, orig_loc_y, marker_points_x, marker_points_y);
      if (options < 0)
        calib_options_ = options;
      else
        set_Calib_Options();
      Extract_Target_Points();
      do_openCV_calibration();
      write_calibration_file(calib_file, true, int_camera_system_);
    };

    // pure virtual destructor
    virtual ~StereoCalib2() {};

    /// \brief set the list of calibration target images
    /// \param image_list a list of the calibration target images in [camera][image] order
    /// \param directory the directory with the calibration images. It will also used as the output directory for the calibration file.
    void
      set_Image_List(const std::vector<std::vector<std::string>> image_list, const std::string directory);

    /// \brief sets information about the calibration target being used
    /// \param target type of target used for calibration. the valid values are enumerated in the DICe_StereoCalib2 namespace
    /// \param num_points_x the total number of intersections/dots in the x direction of the calibration target
    /// \param num_points_y the total number of intersections/dots in the y direction of the calibration target
    /// \param point_spacing distance between the intersections/dots. World coordinates will be specified in the units given for this parameter
    /// \param orig_loc_x Optional the location of the origin marker point in the x direction measured by the number points from the left most point (pt 0)
    /// \param orig_loc_y Optional the location of the origin marker point in the y direction measured by the number points from the bottom most point (pt 0)
    /// \param marker_points_x Optional number of points that specify the x axis in the dot pattern
    /// \param marker_points_y Optional number of points that specify the y axis in the dot pattern
    /// \param options Optional the integer value of the combined options if negative the calibration uses the default options
    /// orig_loc_x/y and marker_points_x/y are only needed for the dot patterns and should be set to 0 for the checkerboard
    /// if the origin marker is in the lower left corner of the rectangle of points the orig_loc_x/y is 0,0
    /// if the marker_points number is not specified it is assumed that the marker points ar at the corners of the rectangle
    void
      set_Target_Info(const DICe_StereoCalib2::Target_Type target, const int num_points_x, const int num_points_y, const double point_spacing,
        const int orig_loc_x = 0, const int orig_loc_y = 0, const int marker_points_x = 0, const int marker_points_y = 0);

    /// \brief turns on/off messages to the console
    void
      set_Verbose(const bool verbose) {
      verbose_ = verbose;
    }
    
    /// \brief extract the points from the calibration image
    /// \param threshold_start Optional starting threshold when looking for the marker dots (only used for dot targets)
    /// \param threshold_end Optional ending threshold when looking for the marker dots (only used for dot targets)
    /// \param threshold_step Optional threshold step when looking for the marker dots (only used for dot targets)
    void
      Extract_Target_Points(int threshold_start = 20, int threshold_end = 250, int threshold_step = 5);

    /// \brief writes the calibration file from the values
    /// \param filename filename (with directory) to write the calibration file to.
    /// \param allfields if true writes all of the fields to the calibration file, even if they have 0 values
    /// \param camsystem the CamSystem object that holds the calibration results
    void
      write_calibration_file(std::string filename, bool allfields, DICe::CamSystem & camsystem);

    /// \brief writes the calibration file from the values (overload)
    /// \param filename filename (with directory) to write the calibration file to.
    /// \param allfields if true writes all of the fields to the calibration file, even if they have 0 values
    /// Note: this form of the function uses the parameters stored in the internal CamSystem object
    void
      write_calibration_file(std::string filename, bool allfields);

    /// \brief set the calibration options used in StereoCalib
    /// Note: calibration opions defined in OpenCV
    /// Calling the function with no parameters sets the default options
    void
      set_Calib_Options(const bool fix_intrinsic = false,
      const bool use_intrinsic_guess = true,
      const bool use_extrinsic_guess = false,
      const bool fix_principal_point = false,
      const bool fix_focal_length = false,
      const bool fix_aspect_ratio = false,
      const bool same_focal_length = false,
      const bool zero_tangent_dist = true,
      const bool fix_k1 = false,
      const bool fix_k2 = false,
      const bool fix_k3 = false,
      const bool fix_k4 = false,
      const bool fix_k5 = false,
      const bool fix_k6 = false,
      const bool rational_model = false,
      const bool thin_prism_model = false,
      const bool fix_s1_s2_s3_s4 = false,
      const bool tilted_model = false,
      const bool fix_taux_tauy = false);

    /// \brief writes a file with the image list and all the intersection points from the calibration images
    /// \param filename the name of the file to write (should end in xml)
    /// \param directory the directory of the file to write. directory + filename should be a complete filepath
    void
      write_intersection_file(std::string filename, std::string directory);

    /// \brief readds a file with the image list and all the intersection points from the calibration images
    /// \param filename the name of the file to write (should end in xml)
    /// \param directory the directory of the file to write. directory + filename should be a complete filepath
    void
      read_intersection_file(std::string filename, std::string directory);

    /// \brief runs the openCV calibration with current set of intersection points and stores the results in a supplied CamSystem
    /// \param camera_system a CamSystem object to store the results
    float
      do_openCV_calibration(DICe::CamSystem & camera_system);
    /// \brief runs the openCV calibration with current set of intersection points and stores the results in the internal CamSystem object
    float
      do_openCV_calibration();

    /// \brief specifies the parameters and turns on/off the creation of images with marked intersection points
    /// \param draw_image turns image drawing on or off
    /// \param dir directory to put the new images into. If dir="" the calibration image directory is used
    /// \param modifier string added to the bas file name to differentiate it from the calibration images
    /// \param extension file extension that also specifies the image type. jpg is used to reduce the image size
    void
      Draw_Intersection_Image(bool draw_images, std::string dir = "", std::string modifier = "_pnts", std::string extension = ".jpg");

    /// \brief returns a summary of the calibration activity
    /// \param summary string that gets filled with the summary
    /// as the calibration system is run summaries of the image list, target parameters, point extraction, and calibration are generated
    /// this allows the use to retrieve and save them if desired.
    void
      get_Calibration_Summary(std::string & summary);

  // Private functions used in StereoCalib2
  private:
    /// extracts points from dot type targets (called by Extract_Target_Points)
    void extract_dot_target_points(int threshold_start, int threshold_end, int threshold_step, int max_scale);
    /// extracts points from checkerboard type targets (called by Extract_Target_Points)
    void extract_checkerboard_intersections();
    /// reorders the keypoints into origin, x dot, y dot based on the distances between them
    void reorder_keypoints(std::vector<cv::KeyPoint> & keypoints);
    /// returns the squared distance between two keypoints
    float dist2(cv::KeyPoint pnt1, cv::KeyPoint pnt2);
    /// returns indicies of three distances sorted by decending magnitude
    void order_dist3(std::vector<float> & dist, std::vector<int> & dist_order);
    /// creates a quadrilateral that describes the allowable area for valid dots
    void create_bounding_box(std::vector<float> & box_x, std::vector<float> & box_y);
    /// checks if a point is in the bounding box quadrilateral
    bool is_in_quadrilateral(const float & x, const float & y, const std::vector<float> & box_x, const std::vector<float> & box_y);
    /// transformation from grid location (interger location with (0,0) at the minimum x minimum y location) to image locations
    void grid_to_image(const float grid_x, const float grid_y, float & img_x, float & img_y);
    /// transformation from image locations to integer grid locations (0,0) at the minimum x minimum y location
    void image_to_grid(const float img_x, const float img_y, float & grid_x, float & grid_y);
    /// uses the openCV blob detector to get possible dot locations
    void get_dot_markers(cv::Mat img, std::vector<cv::KeyPoint> & keypoints, int thresh, bool invert);
    /// calculates the image to grid and grid to images coefficients based on the current set of good points
    void calc_trans_coeff(std::vector<cv::KeyPoint> & imgpoints, std::vector<cv::KeyPoint> & grdpoints);
    /// filters the possible dots by size, the bounding box and how close they are to the expected grid locations
    void filter_dot_markers(int icam, int i_img, std::vector<cv::KeyPoint> dots, std::vector<cv::KeyPoint> & img_points, 
      std::vector<cv::KeyPoint> & grd_points, float dot_tol, cv::Mat out_img, bool draw);
    /// takes the image and grid points and assembles them into object and intersection points in the format required by openCV
    void assemble_intersection_object_points();
    /// clears the arrays of object and intersection points
    void clear_intersection_object_points();
    /// initializes the grid points based on the target information and clears and resizes the image points arrays
    void init_grid_image_points();
    /// saves the string in t_stream_ to the appropriate summary and sends the result to the console if verbose=true
    void send_tstream(std::stringstream & save_stream);
    /// modifies the filename for writing the intersection images
    std::string mod_filename(int i_cam, int i_image);
    /// makes a headder string with a consistant length
    std::string headder(std::string title, char headder_char, int len, int justification);


    //class wide variables
    int calib_options_; //Holds the integer representation of the calibration options
    std::vector<std::string> calib_options_text_; //text vector of all the selected calibration options

    std::vector<std::vector<std::string>> image_list_; //list of target images used for the calibration 
    std::string image_directory_; //directory with the calibration images
    int_t num_cams_; //number of cameras (for future multi camera calibration
    int_t num_images_; //number of image sets in the calibration

    DICe_StereoCalib2::Target_Type target_type_; //type of the target
    int num_points_x_; //number of points in the x direction
    int num_points_y_; //number of points in the y direction
    float point_spacing_; //spacing between the points
    int orig_loc_x_; //x location of the origin point in the grid 
    int orig_loc_y_; //y location of the origin point in the grid 
    int marker_points_x_; //number of points that designate the x axis in a dot pattern 
    int marker_points_y_; //number of points that designate the y axis in a dot pattern 

    bool target_specified_ = false; //has the target been specified
    bool image_list_set_ = false; //has the image list been filled
    bool points_extracted_ = false; //were points extracted from the calibration images
    bool intersections_read_from_file_ = false; //were points read in from an intersection file
    bool calibration_run_ = false; //was the calibration routine run
    bool intersetions_written_to_file_ = false; //were intersection points written out to a file

    bool verbose_; //should output go to the console
    bool draw_intersection_image_; //should intersection point images be generated

    bool valid_intersection_file_; //was the xml format of the intersection file valid
    const int headder_width_ = 70; //standard width of the headders

    cv::Size image_size_; //size of the image 

    std::string intersection_image_dir_; //directory for intersection images
    std::string intersection_image_modifier_; //filename modifier for intersection images
    std::string intersection_image_ext_; //file extension for intersection images

    int num_included_sets_; //number of image sets that are included in the calibration
    std::vector<bool> include_set_; //vector indicating if an image set should be used
    std::vector<int> num_common_pts_; //number of common points in an image set

    //summary streams
    std::stringstream t_stream_;
    std::stringstream image_list_stream_;
    std::stringstream target_info_stream_;
    std::stringstream extraction_stream_;
    std::stringstream read_intersection_file_stream_;
    std::stringstream write_intersection_file_stream_;
    std::stringstream do_calibration_stream_;

    //transformation coefficients
    std::vector<scalar_t> img_to_grdx_;
    std::vector<scalar_t> grd_to_imgx_;
    std::vector<scalar_t> img_to_grdy_;
    std::vector<scalar_t> grd_to_imgy_;

    //intersection and object point storage
    std::vector<std::vector<std::vector<cv::Point2f>>> intersection_points_;
    std::vector<std::vector<cv::Point3f> > object_points_;

    //grid and image point storage
    std::vector<std::vector<cv::Point3f> > grid_points_;
    std::vector<std::vector<std::vector<std::vector<cv::Point2f>>>> image_points_;
    
    //internal CamSystem object 
    DICe::CamSystem int_camera_system_;
  };

}// End DICe Namespace

#endif

