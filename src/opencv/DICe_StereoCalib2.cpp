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

#include <DICe_StereoCalib2.h>


using namespace cv;
using namespace std;

namespace DICe {
  //fill the list of image sets
  void
    StereoCalib2::set_Image_List(const std::vector<std::vector<std::string>> image_list, const std::string directory) {

    //clear the image list flag
    image_list_set_ = false;

    //clear the summary output for the image list
    image_list_stream_ = stringstream();
    //send out a header
    t_stream_ << headder("setting image list sets", '*', headder_width_, 1) << std::endl;
    send_tstream(image_list_stream_);

    //set the number of cameras and images from the list dimensions
    num_cams_ = image_list.size();
    num_images_ = image_list[0].size();

    //throw an exception if the number of cameras in the list is greater than 2
    t_stream_ << "StereoCalib2::set_Image_List: ";
    t_stream_ << num_cams_ << " cameras specified in the image list. Only two camera calibration is supported at this time.";
    //right now we only support two camera calibration
    TEUCHOS_TEST_FOR_EXCEPTION(num_cams_ > 2, std::runtime_error, t_stream_.str());
    t_stream_ = stringstream();

    //copy the image list to the local variables
    image_list_.resize(num_cams_);
    for (int_t i_cam = 0; i_cam < num_cams_; i_cam++) {
      image_list_[i_cam].resize(num_images_);
      for (int_t i_image = 0; i_image < num_images_; i_image++) {
        image_list_[i_cam][i_image] = image_list[i_cam][i_image];
      }
    }
    image_directory_ = directory;

    //create the summary of the list creation
    t_stream_ << "Image directory: " << image_directory_ << std::endl;
    for (int_t i_image = 0; i_image < num_images_; i_image++) {
      t_stream_ << "  Set " << i_image << ": ";
      for (int_t i_cam = 0; i_cam < num_cams_; i_cam++) {
        t_stream_ << image_list_[i_cam][i_image] << "  ";
      }
      t_stream_ << std::endl;
    }
    t_stream_ << headder("end setting image list sets", '*', headder_width_, 1) << std::endl << std::endl;
    send_tstream(image_list_stream_);

    //indicate that the image list has been set
    image_list_set_ = true;

  }

  //set the calibration target information
  void
    StereoCalib2::set_Target_Info(const DICe_StereoCalib2::Target_Type target, const int num_points_x, const int num_points_y, const double point_spacing, 
      const int orig_loc_x, const int orig_loc_y, const int marker_points_x, const int marker_points_y) {
    target_type_ = target;  //type of target set by enum value
    num_points_x_ = num_points_x; //number of points/intersections on the entire plate in the x direction
    num_points_y_ = num_points_y; //number of points/intersections on the entire plate in the y direction
    point_spacing_ = point_spacing; //spacing of the points/intersections in the desired world units
    orig_loc_x_ = orig_loc_x; //location of the origin marker point in the x direction measured by the number points from the left most point (pt 0) 
    orig_loc_y_ = orig_loc_y; //location of the origin marker point in the y direction measured by the number points from the bottom most point (pt 0)
                              // if the origin marker is in the lower left corner the orig_loc_x/y is 0,0
    marker_points_x_ = marker_points_x; //number of points from the origin to the x axis marker point
    marker_points_y_ = marker_points_y; //number of points from the origin to the y axis marker point
    
    //if the number of points between the marker points is not specified assume they are on the outer edges
    //this also works for the checkerboard targets
    if (marker_points_x == 0)           
      marker_points_x_ = num_points_x;  
    if (marker_points_y == 0)
      marker_points_y_ = num_points_y;


    //clear then fill the target info summary 
    target_info_stream_ = stringstream();
    t_stream_ << headder("setting target information", '*', headder_width_, 1) << std::endl;
    t_stream_ << "target type: " << DICe_StereoCalib2::targetTypeStrings[target_type_] << std::endl;
    t_stream_ << "total number of points in the X direction: " << num_points_x_ << std::endl;
    t_stream_ << "total number of points in the Y direction: " << num_points_y_ << std::endl;
    t_stream_ << "point spacing: " << point_spacing << std::endl;
    t_stream_ << "X location of the origin points in grid points: " << orig_loc_x_ << std::endl;
    t_stream_ << "Y location of the origin points in grid points: " << orig_loc_y_ << std::endl;
    t_stream_ << "Number of points in the marked X axis: " << marker_points_x_ << std::endl;
    t_stream_ << "Number of points in the marked Y axis: " << marker_points_y_ << std::endl;
    t_stream_ << headder("end of target information", '*', headder_width_, 1) << std::endl << std::endl;
    send_tstream(target_info_stream_);

    //set the target specified flag
    target_specified_ = true;
  }

  //extract the intersection/dot locations from a calibration target
  void
    StereoCalib2::Extract_Target_Points(int threshold_start, int threshold_end, int threshold_step) {

    //clear the extracted points flag
    points_extracted_ = false;

    //clear the extraction stream and write the headder
    extraction_stream_ = stringstream();
    t_stream_ << headder("extracting calibration points", '*', headder_width_, 1) << std::endl;
    send_tstream(extraction_stream_);

    //check to see if the image list has been filled and the target parameters specified
    if (!target_specified_ || !image_list_set_) {
      t_stream_<< "Error: target not specified or image list not set. Extraction not started";
      send_tstream(extraction_stream_);
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "StereoCalib2::Extract_Target_Points: target not specified or image list not set. Extraction not started");
    }

    //call the appropriate routine depending on the type of target
    switch (target_type_) {
    case DICe_StereoCalib2::CHECKER_BOARD:
      t_stream_ << "extracting checkerboard intersections" << std::endl;
      send_tstream(extraction_stream_);
      extract_checkerboard_intersections();
      break;
    case DICe_StereoCalib2::BLACK_ON_WHITE_W_DONUT_DOTS:
      t_stream_ << "extracting black dots on a white background (three donut dots to specify axis)" << std::endl;
      send_tstream(extraction_stream_);
      extract_dot_target_points(threshold_start, threshold_end, threshold_step, 1);
      break;
    case DICe_StereoCalib2::WHITE_ON_BLACK_W_DONUT_DOTS:
      t_stream_ << "extracting white dots on a black background (three donut dots to specify axis)" << std::endl;
      send_tstream(extraction_stream_);
      extract_dot_target_points(threshold_start, threshold_end, threshold_step, 1);
      break;
    }

    //write the footer
    t_stream_ << headder("end extracting calibration points", '*', headder_width_, 1) << std::endl << std::endl;
    send_tstream(extraction_stream_);

    //set the extracted points flag
    points_extracted_ = true;
    //clear the read from file flag
    intersections_read_from_file_ = false;

  }
  

  //extract the intersection locations from a checkerboard pattern
  void
    StereoCalib2::extract_checkerboard_intersections() {
    Mat out_img;
    bool found;
    cv::Size boardSize; //used by openCV
    std::vector<cv::Point2f> corners; //found corner locations

    //initialize grid and image points
    init_grid_image_points();

    //initialize the include set array
    include_set_.clear();
    include_set_.assign(num_images_, true);

    //set the height and width of the board in intersections
    boardSize.width = num_points_x_;
    boardSize.height = num_points_y_;

    for (int i_image = 0; i_image < num_images_; i_image++) {
      for (int i_cam = 0; i_cam < num_cams_; i_cam++) {

        //put together the file name
        const string & filename = image_directory_ + image_list_[i_cam][i_image];
        t_stream_ << "processing checkerboard cal image: " << image_list_[i_cam][i_image] << std::endl;
        send_tstream(extraction_stream_);

        //read the image
        Mat img = imread(filename, IMREAD_GRAYSCALE);
        if (img.empty()) {
          //if the image is empth mark the set as not used an move on
          t_stream_ << "  image is empty or not found " << std::endl;
          send_tstream(extraction_stream_);
          include_set_[i_image] = false;
        }
        else {
          //the image was found save the size for the calibration
          image_size_ = img.size();
          //copy the image if we are going to save intersection images
          if (draw_intersection_image_) img.copyTo(out_img);
          //find the intersections in the checkerboard
          found = findChessboardCorners(img, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
          if (found) {
            t_stream_ << "checkerboard intersections found" << std::endl;
            send_tstream(extraction_stream_);
            //improve the locations with cornerSubPixel
            //need to check out the window and zero zone parameters to see their effect
            cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
              TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
                30, 0.01));
            //store the points and draw the intersection markers if needed
            int i_pnt = 0;
            for (int i_y = 0; i_y < num_points_y_; i_y++) {
              for (int i_x = 0; i_x < num_points_x_; i_x++) {
                image_points_[i_cam][i_image][i_x][num_points_y_ - 1 - i_y] = corners[i_pnt];
                if (draw_intersection_image_) {
                  circle(out_img, corners[i_pnt], 4, Scalar(0, 0, 0), -1);
                  circle(out_img, corners[i_pnt], 2, Scalar(255, 0, 255), -1);
                }
                i_pnt++;
              }
            }
          }
          else {
            //remove the image from the calibration and proceed with the next image
            include_set_[i_image] = false;
            t_stream_ << "checkerboard intersections were not found" << std::endl;
            send_tstream(extraction_stream_);
          }
          //save the intersection point image if needed
          if (draw_intersection_image_) imwrite(mod_filename(i_cam, i_image), out_img);
        }
      }
    }
  }


  //extract the dot locations from a dot target
  void
    StereoCalib2::extract_dot_target_points(int threshold_start, int threshold_end, int threshold_step, int max_scale) {
    float include_image_set_tol = 0.75; //the search must have found at least 75% of the total to be included
    float dot_tol = 0.25; //how close a dot must be to the projected loation to be considered good
    int i_thresh; //current threshold value
    int i_thresh_first; //first threshold that produced 3 keypoints
    int i_thresh_last; //last threshold that produced 3 keypoints
    int num_included_sets = 0;
    bool keypoints_found;
    stringstream headder_strm;
    Mat out_img; //output image
    num_common_pts_.assign(num_images_, 0);//holds the number of common points keypoints_found for each image

    //initialize grid and image points
    init_grid_image_points();

    //integer grid coordinates of the marker locations
    std::vector<KeyPoint> marker_grid_locs;
    marker_grid_locs.resize(3);
    marker_grid_locs[0].pt.x = orig_loc_x_;
    marker_grid_locs[0].pt.y = orig_loc_y_;
    marker_grid_locs[1].pt.x = orig_loc_x_ + marker_points_x_ - 1;
    marker_grid_locs[1].pt.y = orig_loc_y_;
    marker_grid_locs[2].pt.x = orig_loc_x_;
    marker_grid_locs[2].pt.y = orig_loc_y_ + marker_points_y_ - 1;

    //initialize the include set array
    include_set_.clear();
    include_set_.assign(num_images_, true);

    for (int i_image = 0; i_image < num_images_; i_image++)
    {
      //send out the headder
      headder_strm = stringstream();
      headder_strm << "Starting Image Set " << i_image;
      t_stream_ << headder(headder_strm.str(), '*', headder_width_, 1) << std::endl << std::endl;
      send_tstream(extraction_stream_);

      //go through each of the camera's images (note only two camera calibration is currently supported)
      for (int i_cam = 0; i_cam < num_cams_; i_cam++)
      {
        t_stream_ << "processing cal image: " << image_list_[i_cam][i_image] << std::endl;
        send_tstream(extraction_stream_);
        //get the image
        const string& filename = image_directory_ + image_list_[i_cam][i_image];

        Mat img = imread(filename, IMREAD_GRAYSCALE);

        t_stream_ << "image read" << std::endl;
        send_tstream(extraction_stream_);

        if (img.empty()) {
          t_stream_ << "  image is empty or not found " << std::endl;
          send_tstream(extraction_stream_);
          include_set_[i_image] = false;
        }
        else {
          //save the image size for calibration
          image_size_ = img.size();

          //find the keypoints in the image
          keypoints_found = true;
          std::vector<KeyPoint> keypoints;
          t_stream_ << "  extracting the key points" << std::endl;
          send_tstream(extraction_stream_);

          //try to find the keypoints at different thresholds
          i_thresh_first = 0;
          i_thresh_last = 0;
          for (i_thresh = threshold_start; i_thresh <= threshold_end; i_thresh += threshold_step) {
            //get the dots using an inverted image to get the donut holes 
            get_dot_markers(img, keypoints, i_thresh, true);
            //were three keypoints found?
            if (keypoints.size() != 3) { 
              keypoints_found = false;
              if (i_thresh_first != 0) {
                keypoints_found = true; //keypoints found in a previous pass
                break; // get out of threshold loop
              }
            }
            else
            {
              //save the threshold value if needed
              if (i_thresh_first == 0) {
                i_thresh_first = i_thresh;
              }
              i_thresh_last = i_thresh;
              keypoints_found = true;
            }
          }

          //calculate the average threshold value
          i_thresh = (i_thresh_first + i_thresh_last) / 2;
          //get the key points at the average threshold value
          get_dot_markers(img, keypoints, i_thresh, true);
          //it is possible that this threshold does not have 3 points. 
          //chances are that this indicates p thresholding problem to begin with
          //use the 
          if (keypoints.size() != 3) {
            t_stream_ << "    unable to identify three keypoints" << std::endl;
            t_stream_ << "    other points will not be extracted" << std::endl;
            keypoints_found = false;
          }

          //now that we have the keypoints try to get the rest of the dots
          if (keypoints_found) { //three keypoints were found

            //reorder the keypoints into an origin, xaxis, yaxis order
            reorder_keypoints(keypoints);

            //report the results
            t_stream_ << "    threshold: " << i_thresh << std::endl;
            t_stream_ << "    ordered keypoints " << std::endl;
            for (size_t i = 0; i < keypoints.size(); ++i) //save and display the keypoints
              t_stream_ << "      keypoint: " << keypoints[i].pt.x << " " << keypoints[i].pt.y << std::endl;
            send_tstream(extraction_stream_);

            //add the keypoints to the found positions 
            image_points_[i_cam][i_image][orig_loc_x_][orig_loc_y_].x = keypoints[0].pt.x;
            image_points_[i_cam][i_image][orig_loc_x_][orig_loc_y_].y = keypoints[0].pt.y;
            image_points_[i_cam][i_image][orig_loc_x_ + marker_points_x_ - 1][orig_loc_y_].x = keypoints[1].pt.x;
            image_points_[i_cam][i_image][orig_loc_x_ + marker_points_x_ - 1][orig_loc_y_].y = keypoints[1].pt.y;
            image_points_[i_cam][i_image][orig_loc_x_][orig_loc_y_ + marker_points_y_ - 1].x = keypoints[2].pt.x;
            image_points_[i_cam][i_image][orig_loc_x_][orig_loc_y_ + marker_points_y_ - 1].y = keypoints[2].pt.y;

            //if drawing an output image mark the keypoints with a alternate pattern
            Point cvpoint;
            if (draw_intersection_image_) { //if creating an output image
              //copy the image into the output image
              img.copyTo(out_img);
              for (int n = 0; n < 3; n++) {
                cvpoint.x = keypoints[n].pt.x;
                cvpoint.y = keypoints[n].pt.y;
                circle(out_img, cvpoint, 20, Scalar(0, 0, 0), -1);
                circle(out_img, cvpoint, 15, Scalar(255, 0, 255), -1);
                circle(out_img, cvpoint, 10, Scalar(0, 0, 0), -1);
                circle(out_img, cvpoint, 5, Scalar(255, 0, 255), -1);
              }
            }

            //from the keypoints calculate the image to grid and grid to image transforms (no keystoning)
            calc_trans_coeff(keypoints, marker_grid_locs);

            //determine a threshold from the gray levels between the keypoints
            int xstart, xend, ystart, yend;
            float maxgray, mingray;
            xstart = keypoints[0].pt.x;
            xend = keypoints[1].pt.x;
            if (xend < xstart) {
              xstart = keypoints[1].pt.x;
              xend = keypoints[0].pt.x;
            }
            ystart = keypoints[0].pt.y;
            yend = keypoints[1].pt.y;
            if (yend < ystart) {
              ystart = keypoints[1].pt.y;
              yend = keypoints[0].pt.y;
            }

            maxgray = img.at<uchar>(ystart, xstart);
            mingray = maxgray;
            int curgray;
            for (int ix = xstart; ix <= xend; ix++) {
              for (int iy = ystart; iy <= yend; iy++) {
                curgray = img.at<uchar>(iy, ix);
                if (maxgray < curgray) maxgray = curgray;
                if (mingray > curgray) mingray = curgray;
              }
            }
            i_thresh = (maxgray + mingray) / 2;
            t_stream_ << "  getting the rest of the dots" << std::endl;
            t_stream_ << "    threshold to get dots: " << i_thresh << std::endl;

            //get the rest of the dots
            std::vector<KeyPoint> dots;
            get_dot_markers(img, dots, i_thresh, false);
            t_stream_ << "    prospective grid points found: " << dots.size() << std::endl;
            send_tstream(extraction_stream_);

            //filter the dots
            std::vector<KeyPoint> img_points;
            std::vector<KeyPoint> grd_points;
            // filter dots based on avg size and whether the dots fall in the central box
            filter_dot_markers(i_cam, i_image, dots, img_points, grd_points, dot_tol, out_img, false);

            //initialize the process variables
            int filter_passes = 1;
            int old_dot_num = 3;
            int new_dot_num = img_points.size();
            int max_dots = num_points_x_ * num_points_x_ - 3;

            //if the number of dots has not changed
            while ((old_dot_num != new_dot_num && new_dot_num != max_dots && filter_passes < 20) || filter_passes < 3) {
              //update the old dot count
              old_dot_num = new_dot_num;
              //from the good points that were found improve the mapping parameters
              calc_trans_coeff(img_points, grd_points);
              // filter dots based on avg size and whether the dots fall in the central box with the new parameters
              // the transformation now includes keystoning
              filter_dot_markers(i_cam, i_image, dots, img_points, grd_points, dot_tol, out_img, false);
              filter_passes++;
              new_dot_num = img_points.size();
            }

            //if drawing the images run filter one more time and draw the intersection locations
            if (draw_intersection_image_) filter_dot_markers(i_cam, i_image, dots, img_points, grd_points, dot_tol, out_img, true);

            //save the information about the found dots
            t_stream_ << "    good dots identified: " << new_dot_num << std::endl;
            t_stream_ << "    filter passes: " << filter_passes << std::endl;
            send_tstream(extraction_stream_);

            //save the image points
            for (int n = 0; n < img_points.size(); n++) {
              image_points_[i_cam][i_image][grd_points[n].pt.x][grd_points[n].pt.y].x = img_points[n].pt.x;
              image_points_[i_cam][i_image][grd_points[n].pt.x][grd_points[n].pt.y].y = img_points[n].pt.y;
            }

            //write the intersection image file if requested
            if (draw_intersection_image_) imwrite(mod_filename(i_cam, i_image), out_img);
          }//end if keypoints_found
        }//end if not empty image
      }//end camera loop (i_cam)

      t_stream_ << "find common points in all images" << std::endl;
      send_tstream(extraction_stream_);

      //we have gridded data for each of the cameras use any point common to all the cameras
      vector<vector<Point2f>> temp_int_points;
      vector<Point3f> temp_obj_points;
      temp_int_points.resize(num_cams_);
      for (int m = 0; m < num_points_x_; m++) {
        for (int n = 0; n < num_points_y_; n++) {
          bool common_pt = true;
          //if the values for this location are non-zero for all cameras we have a common point
          for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
            if (image_points_[i_cam][i_image][m][n].x <= 0 || image_points_[i_cam][i_image][m][n].y <= 0)
              common_pt = false;  //do all the cameras have a value for this point?
          }
          if (common_pt) { //save into the image points and objective points for this image set
            num_common_pts_[i_image]++;  //increment the commont point counter for this image
          }
        }
      }
      if (num_common_pts_[i_image] < (num_points_x_*num_points_y_*include_image_set_tol)){
        //exclude the set
        include_set_[i_image] = false;
      }

      //save the summary for the image set
      headder_strm = stringstream();
      headder_strm << "Image Set " << i_image << " summary";
      t_stream_ << headder(headder_strm.str(), '*', headder_width_, 1) << std::endl;
      t_stream_ << "  images: ";
      for (int i = 0; i < num_cams_; i++)
        t_stream_ << image_list_[i][i_image] << "  ";
      t_stream_ << std::endl;
      t_stream_ << "  common points: " << num_common_pts_[i_image] << std::endl;
      if (include_set_[i_image])
        t_stream_ << "  included in calibration: true" << std::endl;
      else
        t_stream_ << "  included in calibration: false" << std::endl;
      t_stream_ << headder("End of Image Set", '*', headder_width_, 1) << std::endl << std::endl;
      send_tstream(extraction_stream_);

    }//end image loop
  }//end extract_dot_target_points

  //assembles the image/grid points into the object/intersection points for calibration
  void
    StereoCalib2::assemble_intersection_object_points() {
    t_stream_ << headder("Assembling intersection/object points", '*', headder_width_, 1) << std::endl << std::endl;
    clear_intersection_object_points();
    
    num_included_sets_ = 0;
    //step through by image set
    for (int i_image = 0; i_image < num_images_; i_image++) {
      //if the image set is included in the calibration
      //the function assumes that the include flag is set properly
      if (include_set_[i_image]) {
        int num_common = 0;
        //find common points in all cameras
        for (int i_x = 0; i_x < num_points_x_; i_x++) {
          for (int i_y = 0; i_y < num_points_y_; i_y++) {
            bool common_pt = true;
            for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
              //do all the cameras have a value for this point?
              if (image_points_[i_cam][i_image][i_x][i_y].x <= 0 || image_points_[i_cam][i_image][i_x][i_y].y <= 0) {
                common_pt = false;  
                break;
              }
            }
            if (common_pt) { //save into the image points and objective points for this image set
              num_common++;
              //fill the intersection and object points for the calibration
              for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
                intersection_points_[i_cam][num_included_sets_].push_back(image_points_[i_cam][i_image][i_x][i_y]);
              }
              object_points_[num_included_sets_].push_back(grid_points_[i_x][i_y]);
            }
          }
        }
        num_included_sets_++;
        t_stream_ << "Image set: " << i_image << "Common points: " << num_common << std::endl;
      }
    }

    send_tstream(do_calibration_stream_);
    //resize for the number of included sets
    for (int i_cam = 0; i_cam < num_cams_; i_cam++)
      intersection_points_[i_cam].resize(num_included_sets_);
    object_points_.resize(num_included_sets_);
  }



  //clears the intersection and object point arrays
  void
    StereoCalib2::clear_intersection_object_points() {
    object_points_.clear();
    object_points_.resize(num_images_);
    intersection_points_.clear();
    intersection_points_.resize(num_cams_);
    for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
      intersection_points_[i_cam].resize(num_images_);
    }
  }

  //initializes the grid image locations in world coordinates
  void
    StereoCalib2::init_grid_image_points() {
    //try to use a consistant coordinate grid system for all targets
    grid_points_.clear();
    grid_points_.resize(num_points_x_);
    for (int m = 0; m < num_points_x_; m++) {
      grid_points_[m].resize(num_points_y_);
      for (int n = 0; n < num_points_y_; n++) {
        grid_points_[m][n].x = (m - orig_loc_x_) * point_spacing_;
        grid_points_[m][n].y = (n - orig_loc_y_) * point_spacing_;
        grid_points_[m][n].z = 0;
      }
    }

    //initialize the image point array
    //initialize the temporary (0,0) image points
    cv::Point2f zero_point;
    zero_point.x = 0;
    zero_point.y = 0;
    //clear then reassemble the arrays filled with (0,0) values
    image_points_.clear();
    image_points_.resize(num_cams_);
    for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
      image_points_[i_cam].resize(num_images_);
      for (int i_image = 0; i_image < num_images_; i_image++) {
        image_points_[i_cam][i_image].resize(num_points_x_);
        for (int i_xpnt = 0; i_xpnt < num_points_x_; i_xpnt++) {
          image_points_[i_cam][i_image][i_xpnt].assign(num_points_y_, zero_point);
        }
      }
    }
  }

  //get all the possible dot markers
  void 
    StereoCalib2::get_dot_markers(cv::Mat img, std::vector<KeyPoint> & keypoints, int thresh, bool invert) {

    // setup the blob detector
    SimpleBlobDetector::Params params;
    params.maxArea = 10e4;
    params.minArea = 100;
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    //clear the points vector
    keypoints.clear();

    //create a temporary image
    Mat timg;
    img.copyTo(timg);
    // setup the binary image
    Mat bi_src(timg.size(), timg.type());
    // apply thresholding
    threshold(timg, bi_src, thresh, 255, cv::THRESH_BINARY);
    // invert the source image
    Mat not_src(bi_src.size(), bi_src.type());
    bitwise_not(bi_src, not_src);

    //detect dots on the appropriately inverted image
    if (target_type_ == DICe_StereoCalib2::BLACK_ON_WHITE_W_DONUT_DOTS) {
      if (!invert) detector->detect(bi_src, keypoints);
      if (invert) detector->detect(not_src, keypoints);
    }
    else {
      if (invert) detector->detect(bi_src, keypoints);
      if (!invert) detector->detect(not_src, keypoints);
    }
  }


  //filter the dot markers by size, bounding box and closeness to the expected grid location
  void
    StereoCalib2::filter_dot_markers(int i_cam, int i_img,
      std::vector<cv::KeyPoint>  dots,
      std::vector<cv::KeyPoint> & img_points, std::vector<cv::KeyPoint> & grd_points,
      float dot_tol, cv::Mat out_img, bool draw) {

    //bounding box values
    std::vector<float> box_x(5, 0.0);
    std::vector<float> box_y(5, 0.0);
    //returned grid values and the interger grid locations
    float grid_x, grid_y;
    long grid_ix, grid_iy;
    float img_x, img_y;
    float grid_x2, grid_y2;
    //single point and keypoint for drawing and storage
    Point cvpoint;
    KeyPoint cvkeypoint;
    //return value
    bool grid_dots_ok = true;
    
    //clear the grid and images points
    grd_points.clear();
    img_points.clear();

    //create the bounding box for the points
    create_bounding_box(box_x, box_y);
    if (draw) {
      for (int n = 0; n < 4; n++) {
        cvpoint.x = box_x[n];
        cvpoint.y = box_y[n];
        circle(out_img, cvpoint, 10, Scalar(128, 0, 128), -1);
        circle(out_img, cvpoint, 5, Scalar(0, 0, 0), -1);
      }
    }

    // compute the average dot size:
    float avg_dot_size = 0.0;
    assert(dots.size() > 0);
    for (size_t i = 0; i < dots.size(); ++i) {
      avg_dot_size += dots[i].size;
    }
    avg_dot_size /= dots.size();
    
    //filter the points 
    for (size_t n = 0; n < dots.size(); ++n) {
      //if requested draw all the found points
      if (draw) {
        //circle needs cvpoints not key points
        cvpoint.x = dots[n].pt.x;
        cvpoint.y = dots[n].pt.y;
        //draw the white (found) circle
        circle(out_img, cvpoint, 20, Scalar(255, 0, 255), -1);
      }

      //is the point in an acceptable size range
      if (dots[n].size < 0.8f*avg_dot_size || dots[n].size > 1.4f*avg_dot_size) continue;
      //draw the black (size ok) circle
      if (draw) circle(out_img, cvpoint, 16, Scalar(0, 0, 0), -1);

      //is the point in the bounding box 
      if (!is_in_quadrilateral(dots[n].pt.x, dots[n].pt.y, box_x, box_y)) continue;
      if (draw) circle(out_img, cvpoint, 12, Scalar(255, 0, 255), -1);

      //get the corresponding grid location from the image location
      image_to_grid((float)dots[n].pt.x, (float)dots[n].pt.y, grid_x, grid_y);
      //get the nearest integer dot location
      grid_ix = std::lround(grid_x);
      grid_iy = std::lround(grid_y);
      //is it within the acceptable distance from the expected location and in the desired grid area
      if (abs(grid_x - round(grid_x)) <= dot_tol && abs(grid_y - round(grid_y)) <= dot_tol &&
        grid_ix>=0 && grid_ix < num_points_x_ && grid_iy>=0 && grid_iy < num_points_y_) {
        //save the point
        cvkeypoint.pt.x = grid_ix;
        cvkeypoint.pt.y = grid_iy;
        img_points.push_back(dots[n]);
        grd_points.push_back(cvkeypoint);
        if(draw) circle(out_img, cvpoint, 8, Scalar(0, 0, 0), -1);
      }

    }//end dots loop

    
    //draw the expected locations
    if (draw) {
      float imgx, imgy, grdx, grdy;
      for (float i_x = 0; i_x < num_points_x_; i_x++) {
        for (float i_y = 0; i_y < num_points_y_; i_y++) {
          grid_to_image(i_x, i_y, imgx, imgy);
          cvpoint.x = imgx;
          cvpoint.y = imgy;
          circle(out_img, cvpoint, 4, Scalar(255, 0, 255), -1);
        }
      }
    }

  }


  //is a point contained within the quadrilateral
  bool
    StereoCalib2::is_in_quadrilateral(const float & x,
    const float & y,
    const std::vector<float> & box_x,
    const std::vector<float> & box_y) {
    //is the point within the box
    float angle = 0.0;
    assert(box_x.size() == 5);
    assert(box_y.size() == 5);

    for (int_t i = 0; i < 4; i++) {
      // get the two end points of the polygon side and construct
      // a vector from the point to each one:
      const float dx1 = box_x[i] - x;
      const float dy1 = box_y[i] - y;
      const float dx2 = box_x[i + 1] - x;
      const float dy2 = box_y[i + 1] - y;
      angle += angle_2d(dx1, dy1, dx2, dy2); //angle_2d from DICe_shape
    }
    // if the angle is greater than PI, the point is in the polygon
    if (std::abs(angle) >= DICE_PI) {
      return true;
    }
    return false;
  }


  //reorder the keypoints into origin, xaxis, yaxis order
  void
    StereoCalib2::reorder_keypoints(std::vector<KeyPoint> & keypoints) {
    vector<float> dist(3, 0.0); //holds the distances between the points
    vector<KeyPoint> temp_points; 
    vector<int> dist_order(3, 0); //index order of the distances max to min
    float cross; //cross product and indicies
    temp_points.clear();
    //save the distances between the points (note if dist(1,2) is max point 0 is the origin)
    dist[0] = dist2(keypoints[1], keypoints[2]);
    dist[1] = dist2(keypoints[0], keypoints[2]);
    dist[2] = dist2(keypoints[0], keypoints[1]);
    //order the distances
    order_dist3(dist, dist_order);

    //calaulate the cross product to determine the x and y axis
    int io = dist_order[0];
    int i1 = dist_order[1];
    int i2 = dist_order[2];
    
    //if the cross product is positive i1 was the y axis point because of inverted image coordinates
    cross = ((keypoints[i1].pt.x - keypoints[io].pt.x) * (keypoints[i2].pt.y - keypoints[io].pt.y)) -
      ((keypoints[i1].pt.y - keypoints[io].pt.y) * (keypoints[i2].pt.x - keypoints[io].pt.x));
    if (cross > 0.0) { //i2 is the x axis
      i2 = dist_order[1];
      i1 = dist_order[2];
    }
    //reorder the points and return
    temp_points.push_back(keypoints[io]);
    temp_points.push_back(keypoints[i1]);
    temp_points.push_back(keypoints[i2]);
    keypoints[0] = temp_points[0];
    keypoints[1] = temp_points[1];
    keypoints[2] = temp_points[2];
  }

  //distance between two points
  float
    StereoCalib2::dist2(KeyPoint pnt1, KeyPoint pnt2) {
    return (pnt1.pt.x - pnt2.pt.x)*(pnt1.pt.x - pnt2.pt.x) + (pnt1.pt.y - pnt2.pt.y)*(pnt1.pt.y - pnt2.pt.y);
  }

  //order three distances returns biggest to smallest
  void
  StereoCalib2::order_dist3(vector<float> & dist, vector<int> & dist_order) {
    dist_order[0] = 0;
    dist_order[2] = 0;
    for (int i = 1; i < 3; i++) {
      if (dist[dist_order[0]] < dist[i]) dist_order[0] = i;
      if (dist[dist_order[2]] > dist[i]) dist_order[2] = i;
    }
    if (dist_order[0] == dist_order[2]) assert(false);
    dist_order[1] = 3 - (dist_order[0] + dist_order[2]);
  }

  //create a box around the valid points
  void
    StereoCalib2::create_bounding_box(std::vector<float> & box_x, std::vector<float> & box_y) {
    float xgrid, ygrid;
    //xmin, ymin point
    xgrid = - 0.5;
    ygrid = - 0.5;
    grid_to_image(xgrid, ygrid, box_x[0], box_y[0]);
    //xmax, ymin point
    xgrid = num_points_x_ -1 + 0.5;
    ygrid =  - 0.5;
    grid_to_image(xgrid, ygrid, box_x[1], box_y[1]);
    //xmax, ymax point
    xgrid = num_points_x_  - 1 + 0.5;
    ygrid = num_points_y_  - 1 + 0.5;
    grid_to_image(xgrid, ygrid, box_x[2], box_y[2]);
    //xmin, ymax point
    xgrid =  - 0.5;
    ygrid = num_points_y_ - 1 + 0.5;
    grid_to_image(xgrid, ygrid, box_x[3], box_y[3]);
    //close the loop
    box_x[4] = box_x[0];
    box_y[4] = box_y[0];
  }


  //calculate the transformation coefficients
  void
    StereoCalib2::calc_trans_coeff(std::vector<cv::KeyPoint> & imgpoints, std::vector<cv::KeyPoint> & grdpoints) {
    //resize the coefficient vectors
    img_to_grdx_.resize(6);
    grd_to_imgx_.resize(6);
    img_to_grdy_.resize(6);
    grd_to_imgy_.resize(6);

    //if only three points are given the six parameter mapping will not include keystoning
    if (imgpoints.size() == 3) {
      Mat A = Mat_<double>(3, 3);
      Mat Ax = Mat_<double>(3, 1);
      Mat Ay = Mat_<double>(3, 1);
      Mat coeff_x = Mat_<double>(3, 1);
      Mat coeff_y = Mat_<double>(3, 1);

      //image to grid transform (not overdetermined so just solve the matrix equation)
      for (int i = 0; i < 3; i++) {
        A.at<double>(i, 0) = 1.0;
        A.at<double>(i, 1) = imgpoints[i].pt.x;
        A.at<double>(i, 2) = imgpoints[i].pt.y;
        Ax.at<double>(i, 0) = grdpoints[i].pt.x;
        Ay.at<double>(i, 0) = grdpoints[i].pt.y;
      }
      //solve for the coefficients
      coeff_x = A.inv()*Ax;
      coeff_y = A.inv()*Ay;
      //copy over the coefficients
      for (int i_coeff = 0; i_coeff < 3; i_coeff++) {
        img_to_grdx_[i_coeff] = coeff_x.at<double>(i_coeff, 0);
        img_to_grdy_[i_coeff] = coeff_y.at<double>(i_coeff, 0);
      }
      //set the higher order terms to 0
      img_to_grdx_[3] = 0.0;
      img_to_grdy_[3] = 0.0;
      img_to_grdx_[4] = 0.0;
      img_to_grdy_[4] = 0.0;      
      img_to_grdx_[5] = 0.0;
      img_to_grdy_[5] = 0.0;

      //grid to image transform (not overdetermined so just solve the matrix equation)
      for (int i = 0; i < 3; i++) {
        A.at<double>(i, 0) = 1.0;
        A.at<double>(i, 1) = grdpoints[i].pt.x;
        A.at<double>(i, 2) = grdpoints[i].pt.y;
        Ax.at<double>(i, 0) = imgpoints[i].pt.x;
        Ay.at<double>(i, 0) = imgpoints[i].pt.y;
      }
      //solve for the coefficients
      coeff_x = A.inv()*Ax;
      coeff_y = A.inv()*Ay;
      //copy over the coefficients
      for (int i_coeff = 0; i_coeff < 3; i_coeff++) {
        grd_to_imgx_[i_coeff] = coeff_x.at<double>(i_coeff, 0);
        grd_to_imgy_[i_coeff] = coeff_y.at<double>(i_coeff, 0);
      }
      //set the higher order terms to 0
      grd_to_imgx_[3] = 0.0;
      grd_to_imgy_[3] = 0.0;
      grd_to_imgx_[4] = 0.0;
      grd_to_imgy_[4] = 0.0;
      grd_to_imgx_[5] = 0.0;
      grd_to_imgy_[6] = 0.0;
    }

    //if more than three points are supplied used an 12 parameter mapping to reflect keystoning
    if (imgpoints.size() > 3) {
      
      Mat A = Mat_<double>(imgpoints.size(), 6);
      Mat AtA = Mat_<double>(6, 6);
      Mat bx = Mat_<double>(imgpoints.size(), 1);
      Mat by = Mat_<double>(imgpoints.size(), 1);
      Mat Atb = Mat_<double>(6, 1);
      Mat coeff_x = Mat_<double>(6, 1);
      Mat coeff_y = Mat_<double>(6, 1);

      //image to grid transform (least squares fit)
      for (int i = 0; i < imgpoints.size(); i++) {
        A.at<double>(i, 0) = 1.0;
        A.at<double>(i, 1) = imgpoints[i].pt.x;
        A.at<double>(i, 2) = imgpoints[i].pt.y;
        A.at<double>(i, 3) = imgpoints[i].pt.x * imgpoints[i].pt.y;
        A.at<double>(i, 4) = imgpoints[i].pt.x * imgpoints[i].pt.x;
        A.at<double>(i, 5) = imgpoints[i].pt.y * imgpoints[i].pt.y;
        bx.at<double>(i, 0) = grdpoints[i].pt.x;
        by.at<double>(i, 0) = grdpoints[i].pt.y;
      }
      //solve for the coefficients
      AtA = A.t()*A;
      Atb = A.t()*bx;
      coeff_x = AtA.inv()*Atb;
      Atb = A.t()*by;
      coeff_y = AtA.inv()*Atb;

      //copy over the coefficients
      for (int i_coeff = 0; i_coeff < 6; i_coeff++) {
        img_to_grdx_[i_coeff] = coeff_x.at<double>(i_coeff, 0);
        img_to_grdy_[i_coeff] = coeff_y.at<double>(i_coeff, 0);
      }

      //grid to image transform
      for (int i = 0; i < imgpoints.size(); i++) {
        A.at<double>(i, 0) = 1.0;
        A.at<double>(i, 1) = grdpoints[i].pt.x;
        A.at<double>(i, 2) = grdpoints[i].pt.y;
        A.at<double>(i, 3) = grdpoints[i].pt.x * grdpoints[i].pt.y;
        A.at<double>(i, 4) = grdpoints[i].pt.x * grdpoints[i].pt.x;
        A.at<double>(i, 5) = grdpoints[i].pt.y * grdpoints[i].pt.y;
        bx.at<double>(i, 0) = imgpoints[i].pt.x;
        by.at<double>(i, 0) = imgpoints[i].pt.y;
      }
      //solve for the coefficients
      AtA = A.t()*A;
      Atb = A.t()*bx;
      coeff_x = AtA.inv()*Atb;
      Atb = A.t()*by;
      coeff_y = AtA.inv()*Atb;

      //copy over the coefficients
      for (int i_coeff = 0; i_coeff < 6; i_coeff++) {
        grd_to_imgx_[i_coeff] = coeff_x.at<double>(i_coeff, 0);
        grd_to_imgy_[i_coeff] = coeff_y.at<double>(i_coeff, 0);
      }
    }
  }

  //convert grid locations to image locations
  void
    StereoCalib2::grid_to_image(const float grid_x, const float grid_y, float & img_x, float & img_y) {
    img_x = grd_to_imgx_[0] + grd_to_imgx_[1] * grid_x + grd_to_imgx_[2] * grid_y + grd_to_imgx_[3] * grid_x * grid_y + grd_to_imgx_[4] * grid_x * grid_x + grd_to_imgx_[5] * grid_y * grid_y;
    img_y = grd_to_imgy_[0] + grd_to_imgy_[1] * grid_x + grd_to_imgy_[2] * grid_y + grd_to_imgy_[3] * grid_x * grid_y + grd_to_imgy_[4] * grid_x * grid_x + grd_to_imgy_[5] * grid_y * grid_y;
  }

  //convert image locations to grid locations
  void
    StereoCalib2::image_to_grid(const float img_x, const float img_y, float & grid_x, float & grid_y) {
    grid_x = img_to_grdx_[0] + img_to_grdx_[1] * img_x + img_to_grdx_[2] * img_y + img_to_grdx_[3] * img_x * img_y + img_to_grdx_[4] * img_x * img_x + img_to_grdx_[5] * img_y * img_y;
    grid_y = img_to_grdy_[0] + img_to_grdy_[1] * img_x + img_to_grdy_[2] * img_y + img_to_grdy_[3] * img_x * img_y + img_to_grdy_[4] * img_x * img_x + img_to_grdy_[5] * img_y * img_y;
  }


  //write a file with the intersection information
  void
    StereoCalib2::write_intersection_file(string filename, string directory) {
    std::stringstream param_title;
    std::stringstream param_val;
    std::stringstream valid_fields;
    std::stringstream x_row;
    std::stringstream y_row;
    std::stringstream z_row;
    std::string out_file;
    std::string comment_text;
    std::string headder_text;

    intersetions_written_to_file_ = false;


    //clear the summary
    write_intersection_file_stream_ = stringstream();
    headder_text = headder("Writing intersection file", '*', headder_width_, 1);
    t_stream_ << headder_text << std::endl;
    //if no output directory is gien use the image directory
    t_stream_ << "filename: " << filename << std::endl;
    if (directory == "") {
      out_file = image_directory_ + filename;
      t_stream_ << "directory: " << image_directory_ << std::endl;
    }
    else {
      out_file = directory + filename;
      t_stream_ << "directory: " << directory << std::endl;
    }
    send_tstream(write_intersection_file_stream_);

    //initialize the output file
    initialize_xml_file(out_file);

    //write an identifier
    write_xml_comment(out_file, "DICe formatted intersection file");
    write_xml_comment(out_file, "intersection_file_ID must be in the file with a value of DICe_XML_Intersection_File");
    write_xml_string_param(out_file, "FILE_ID", DICe_StereoCalib2::intersection_file_ID_, false);

    //write the global parameters that define the intersection information
    write_xml_comment(out_file, "");
    write_xml_comment(out_file, "Base information on the intersection data");
    t_stream_ << std::endl << headder("Base information on the intersection data", '*', headder_width_, 1) << std::endl;
    param_val = stringstream();
    param_title = stringstream();
    comment_text = "Number of images";
    param_title << "NUM_IMAGE_SETS";
    param_val << num_images_;
    write_xml_size_param(out_file, param_title.str(), param_val.str(), false);
    t_stream_ << param_title.str() << ": " << param_val.str() << std::endl;

    param_val = stringstream();
    param_title = stringstream();
    comment_text = "Number of cameras";
    param_title << "NUM_CAMERAS";
    param_val << num_cams_;
    write_xml_size_param(out_file, param_title.str(), param_val.str(), false);
    t_stream_ << param_title.str() << ": " << param_val.str() << std::endl;

    param_val = stringstream();
    param_title = stringstream();
    comment_text = "Number of intersections/dots in the x direction";
    param_title << "NUM_X_INTERSECTIONS";
    param_val << num_points_x_;
    write_xml_size_param(out_file, param_title.str(), param_val.str(), false);
    t_stream_ << param_title.str() << ": " << param_val.str() << std::endl;

    param_val = stringstream();
    param_title = stringstream();
    comment_text = "Number of intersections/dots in the y direction";
    param_title << "NUM_Y_INTERSECTIONS";
    param_val << num_points_y_;
    write_xml_size_param(out_file, param_title.str(), param_val.str(), false);
    t_stream_ << param_title.str() << ": " << param_val.str() << std::endl;

    param_val = stringstream();
    param_title = stringstream();
    comment_text = "Image height";
    param_title << "IMAGE_HEIGHT";
    param_val << image_size_.height;
    write_xml_size_param(out_file, param_title.str(), param_val.str(), false);
    t_stream_ << param_title.str() << ": " << param_val.str() << std::endl;

    param_val = stringstream();
    param_title = stringstream();
    comment_text = "Image width";
    param_title << "IMAGE_WIDTH";
    param_val << image_size_.width;
    write_xml_size_param(out_file, param_title.str(), param_val.str(), false);
    t_stream_ << param_title.str() << ": " << param_val.str() << std::endl;

    param_val = stringstream();
    param_title = stringstream();
    comment_text = "Image directory";
    param_title << "IMAGE_DIRECTORY";
    param_val << image_directory_;
    write_xml_string_param(out_file, param_title.str(), param_val.str(), false);
    t_stream_ << param_title.str() << ": " << param_val.str() << std::endl;


    //write the include value and file names for each image set
    t_stream_ << std::endl << headder("Image sets", '*', headder_width_, 1) << std::endl;
    write_xml_comment(out_file, headder_text);

    write_xml_comment(out_file, "include flag and image information about each image set");
    for (int i_image = 0; i_image < num_images_; i_image++) {
      param_title = stringstream();
      param_title << "IMAGE_SET_" << i_image;
      t_stream_ << param_title.str() << std::endl;
      write_xml_comment(out_file, param_title.str());
      write_xml_param_list_open(out_file, param_title.str(), false);
      if (include_set_[i_image]) {
        write_xml_string_param(out_file, "INCLUDE", "true", false);
        t_stream_ << "Set included: ";
      }
      else { 
        write_xml_string_param(out_file, "INCLUDE", "false", false); 
        t_stream_ << "Set not included: ";
      }
      for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
        param_title = stringstream();
        param_title << "CAM_" << i_cam;
        write_xml_string_param(out_file, param_title.str(), image_list_[i_cam][i_image], false);
        t_stream_ << image_list_[i_cam][i_image] << " ";

      }
      write_xml_param_list_close(out_file, false);
      t_stream_ << std::endl;
    }

    //write the grid intersections
    write_xml_comment(out_file, "");
    write_xml_comment(out_file, "Grid intersection points");
    write_xml_comment(out_file, "  If you are using a flat grid all points will have Z=0");
    write_xml_param_list_open(out_file, "GRID_INTERSECTIONS", false);

    t_stream_ << std::endl << headder("grid intersections", '*', headder_width_, 1) << std::endl;

    t_stream_ << "writing grid intersections" << std::endl;

    for (int i_row = 0; i_row < num_points_y_; i_row++) {
      param_title = stringstream();
      param_title << "ROW_" << i_row;
      write_xml_comment(out_file, param_title.str());
      write_xml_param_list_open(out_file, param_title.str(), false);
      x_row = stringstream();
      y_row = stringstream();
      z_row = stringstream();
      x_row << "{ ";
      y_row << "{ ";
      z_row << "{ ";
      for (int i_x = 0; i_x < (num_points_x_ - 1); i_x++) {
        x_row << grid_points_[i_x][i_row].x << ",  ";
        y_row << grid_points_[i_x][i_row].y << ",  ";
        z_row << grid_points_[i_x][i_row].z << ",  ";
      }
      x_row << grid_points_[num_points_x_ - 1][i_row].x << "}";
      y_row << grid_points_[num_points_x_ - 1][i_row].y << "}";
      z_row << grid_points_[num_points_x_ - 1][i_row].z << "}";
      write_xml_string_param(out_file, "X", x_row.str(), false);
      write_xml_string_param(out_file, "Y", y_row.str(), false);
      write_xml_string_param(out_file, "Z", z_row.str(), false);

      write_xml_param_list_close(out_file, false);
    }
    write_xml_param_list_close(out_file, false);


    t_stream_ << std::endl << headder("image intersections", '*', headder_width_, 1) << std::endl;

    //write the images intersections
    write_xml_comment(out_file, "");
    write_xml_comment(out_file, "individual image intersection points");
    write_xml_comment(out_file, "the name of the sublist must match that of the images in the image set list");
    write_xml_comment(out_file, "unidentified points must be included in the list as points with a 0,0 value. They are not included in the calibration");
    write_xml_comment(out_file, "");
    for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
      for (int i_image = 0; i_image < num_images_; i_image++) {
        write_xml_comment(out_file, image_list_[i_cam][i_image]);
        write_xml_param_list_open(out_file, image_list_[i_cam][i_image], false);
        t_stream_ << "writing image intersections " << image_list_[i_cam][i_image] << std::endl;
        for (int i_row = 0; i_row < num_points_y_; i_row++) {
          param_title = stringstream();
          param_title << "ROW_" << i_row;
          write_xml_comment(out_file, param_title.str());
          write_xml_param_list_open(out_file, param_title.str(), false);
          x_row = stringstream();
          y_row = stringstream();
          x_row << "{ ";
          y_row << "{ ";
          for (int i_x = 0; i_x < (num_points_x_ - 1); i_x++) {
            x_row << image_points_[i_cam][i_image][i_x][i_row].x << ",  ";
            y_row << image_points_[i_cam][i_image][i_x][i_row].y << ",  ";
          }
          x_row << image_points_[i_cam][i_image][num_points_x_ - 1][i_row].x << "}";
          y_row << image_points_[i_cam][i_image][num_points_x_ - 1][i_row].y << "}";
          write_xml_string_param(out_file, "X", x_row.str(), false);
          write_xml_string_param(out_file, "Y", y_row.str(), false);
          write_xml_param_list_close(out_file, false);
        }
        write_xml_param_list_close(out_file, false);
        write_xml_comment(out_file, "");
      }
    }
    finalize_xml_file(out_file);
    t_stream_ << std::endl << headder("end writing intersection file", '*', headder_width_, 1) << std::endl;
    send_tstream(write_intersection_file_stream_);
    intersetions_written_to_file_ = true;
  }


  //read a file with intersection information
  void
    StereoCalib2::read_intersection_file(string filename, string directory) {

    //clear the summary string stream
    read_intersection_file_stream_ = stringstream();
    std::string headder_text;
    string out_file;
    int_t camera_index = 0;
    bool valid_xml = true;
    bool param_found = false;
    valid_intersection_file_ = true;

    std::string msg = "";
    std::string param_text = "";
    std::stringstream param_title;
    std::stringstream param_val;

    //use the image directory if no directory is given
    if (directory == "")
      out_file = image_directory_ + filename;
    else
      out_file = directory + filename;

    headder_text = headder("StereoCalib2::read_intersection_file", '*', headder_width_, 1);
    t_stream_ << headder_text << std::endl;

    t_stream_ << "load intersection file: " << out_file << std::endl;
    send_tstream(read_intersection_file_stream_);

    Teuchos::RCP<Teuchos::ParameterList> sys_parms = Teuchos::rcp(new Teuchos::ParameterList());
    Teuchos::Ptr<Teuchos::ParameterList> sys_parms_ptr(sys_parms.get());
    try {
      Teuchos::updateParametersFromXmlFile(out_file, sys_parms_ptr);
    }
    catch (std::exception & e) {
      t_stream_ << "Invalid XML file: " << std::endl;
      t_stream_ << e.what() << std::endl;
      send_tstream(read_intersection_file_stream_);
      valid_xml = false;
    }

    if (valid_xml) {  
      //valid XML file
      t_stream_ << "Valid XML file for Teuchos parser" << std::endl;
      param_found = sys_parms->isParameter("FILE_ID");
      if (param_found)	param_text = sys_parms->get<std::string>("FILE_ID");
      if (param_text != DICe_StereoCalib2::intersection_file_ID_ || !param_found) {
        t_stream_ << "StereoCalib2::read_intersection_file():XML calibration ID file not valid" << std::endl;
        valid_intersection_file_ = false;
      }
      send_tstream(read_intersection_file_stream_);

      //read the basic parameters from the file
      param_found = sys_parms->isParameter("NUM_CAMERAS");
      if (param_found) {
        num_cams_ = sys_parms->get<int_t>("NUM_CAMERAS");
        t_stream_ << "Number of cameras: " << num_cams_ << std::endl;
      }
      else t_stream_ << "Number of cameras {NUM_CAMERAS} missing from file" << std::endl;
      send_tstream(read_intersection_file_stream_);

      param_found = sys_parms->isParameter("NUM_IMAGE_SETS");
      if (param_found) {
        num_images_ = sys_parms->get<int_t>("NUM_IMAGE_SETS");
        t_stream_ << "Number of image sets: " << num_images_ << std::endl;
      }
      else t_stream_ << "Number of image sets {NUM_IMAGE_SETS} missing from file" << std::endl;
      send_tstream(read_intersection_file_stream_);

      param_found = sys_parms->isParameter("NUM_X_INTERSECTIONS");
      if (param_found) {
        num_points_x_ = sys_parms->get<int_t>("NUM_X_INTERSECTIONS");
        t_stream_ << "Number of intersections in the x direction: " << num_points_x_ << std::endl;
      }
      else t_stream_ << "Number of intersections in the x direction {NUM_X_INTERSECTIONS} missing from file" << std::endl;
      send_tstream(read_intersection_file_stream_);

      param_found = sys_parms->isParameter("NUM_Y_INTERSECTIONS");
      if (param_found) {
        num_points_y_ = sys_parms->get<int_t>("NUM_Y_INTERSECTIONS");
        t_stream_ << "Number of intersections in the y direction: " << num_points_y_ << std::endl;
      }
      else t_stream_ << "Number of intersections in the y direction {NUM_Y_INTERSECTIONS} missing from file" << std::endl;
      send_tstream(read_intersection_file_stream_);

      param_found = sys_parms->isParameter("IMAGE_HEIGHT");
      if (param_found) {
        image_size_.height = sys_parms->get<int_t>("IMAGE_HEIGHT");
        t_stream_ << "Image height: " << image_size_.height << std::endl;
      }
      else t_stream_ << "Image height {IMAGE_HEIGHT} missing from file" << std::endl;
      send_tstream(read_intersection_file_stream_);

      param_found = sys_parms->isParameter("IMAGE_WIDTH");
      if (param_found) {
        image_size_.width = sys_parms->get<int_t>("IMAGE_WIDTH");
        t_stream_ << "Image width: " << image_size_.width << std::endl;
      }
      else t_stream_ << "Image width {IMAGE_WIDTH} missing from file" << std::endl;
      t_stream_  << std::endl;
      send_tstream(read_intersection_file_stream_);

      //initially set the image list flag  if there is missing data the flag will be turned off
      image_list_set_ = true;
      param_found = sys_parms->isParameter("IMAGE_DIRECTORY");
      if (param_found) {
        image_directory_ = sys_parms->get<std::string>("IMAGE_DIRECTORY");
        t_stream_ << "Image directory: " << image_directory_ << std::endl;
      }
      else {
        t_stream_ << "Image directory {IMAGE_DIRECTORY} missing from file" << std::endl;
        image_list_set_ = false;
      }
      send_tstream(read_intersection_file_stream_);

      //initialize the grid and image arrays
      init_grid_image_points();
      //clear the include list
      include_set_.clear();
      include_set_.resize(num_images_);
      //clear the image list
      image_list_.clear();
      image_list_.resize(num_cams_);
      for (int i_cam = 0; i_cam < num_cams_; i_cam++) 
        image_list_[i_cam].resize(num_images_);

      //fill the image list and include list
      for (int i_image = 0; i_image < num_images_; i_image++) {
        param_title = stringstream();
        param_title << "IMAGE_SET_" << i_image;
        t_stream_ << "IMAGE_SET_" << i_image << std::endl;
        if (sys_parms->isSublist(param_title.str())) {
          Teuchos::ParameterList image_sets = sys_parms->sublist(param_title.str());
          param_found = image_sets.isParameter("INCLUDE");
          if (param_found) {
            param_text = image_sets.get<std::string>("INCLUDE");
            if (param_text == "true" || param_text == "True" || param_text == "TRUE") {
              include_set_[i_image] = true;
              t_stream_ << "  image set included" << std::endl;
            }
            else {
              include_set_[i_image] = false;
              t_stream_ << "  image set not included" << std::endl;
            }
          }
          else {
            include_set_[i_image] = false;
            t_stream_ << "Include boolean {INCLUDE} missing from file image set will not be included" << std::endl;
          }
          for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
            param_title = stringstream();
            param_title << "CAM_" << i_cam;
            param_found = image_sets.isParameter(param_title.str());
            if (param_found){
              param_text = image_sets.get<std::string>(param_title.str());
              image_list_[i_cam][i_image] = param_text;
              t_stream_ << "  " << param_title.str() << ": " << param_text << std::endl;
            }
            else {
              t_stream_ << "  " << param_title.str() << ": not found in file" << std::endl;
              image_list_set_ = false;
            }
          }
        }
        else {
          t_stream_ << param_title.str() << ": was not found in the file, image set marked as not included";
          include_set_[i_image] = false;
          image_list_set_ = false;
        }
        send_tstream(read_intersection_file_stream_);
      }

      //read the grid intersection from the file 
      t_stream_ << std::endl << "Reading grid intersections" << std::endl;
      if (sys_parms->isSublist("GRID_INTERSECTIONS")) {
        Teuchos::ParameterList grid_data = sys_parms->sublist("GRID_INTERSECTIONS");
        //read each of the rows
        for (int i_row = 0; i_row < num_points_y_; i_row++) {
          param_title = stringstream();
          param_title << "ROW_" << i_row;
          if (grid_data.isSublist(param_title.str())) {
            Teuchos::ParameterList row_data = grid_data.sublist(param_title.str());
            if (row_data.isParameter("X") && row_data.isParameter("Y") && row_data.isParameter("Z")) {
              Teuchos::Array<scalar_t> tArrayX;
              param_text = row_data.get<std::string>("X");
              tArrayX = Teuchos::fromStringToArray<scalar_t>(param_text);
              Teuchos::Array<scalar_t> tArrayY = Teuchos::fromStringToArray<scalar_t>(row_data.get<std::string>("Y"));
              Teuchos::Array<scalar_t> tArrayZ = Teuchos::fromStringToArray<scalar_t>(row_data.get<std::string>("Z"));

              if (tArrayX.size() == num_points_x_ && tArrayY.size() == num_points_x_ && tArrayZ.size() == num_points_x_) {
                for (int i_x = 0; i_x < num_points_x_; i_x++) {
                  grid_points_[i_x][i_row].x = tArrayX[i_x];
                  grid_points_[i_x][i_row].y = tArrayY[i_x];
                  grid_points_[i_x][i_row].z = tArrayZ[i_x];
                }
              }
              else {
                t_stream_ << param_title.str() << " the number of points in the X, Y or Z sublist did not match the expected value of " << num_points_x_ << std::endl;
                valid_xml = false;
              }
            }
          }
          else {
            t_stream_ << param_title.str() << ": X, Y or Z parameter of the row was not found in the file " << std::endl;
            valid_xml = false;
          }
        }
      }
      else {
        t_stream_ << "invalid intersection file: the grid intersections sublist {GRID_INTERSECTIONS} was not found" << std::endl;
        valid_xml = false;
      }

      send_tstream(read_intersection_file_stream_);

      points_extracted_ = false;

      //read the individual image intersections from the file
      t_stream_ << std::endl << "Reading intersection information for each camera/image set" << std::endl;
      for (int i_cam = 0; i_cam < num_cams_; i_cam++) {
        for (int i_image = 0; i_image < num_images_; i_image++) {

          t_stream_ << "Reading intersections for: " << image_list_[i_cam][i_image] << std::endl;
          //read the grid file infomation 
          if (sys_parms->isSublist(image_list_[i_cam][i_image])) {
            Teuchos::ParameterList grid_data = sys_parms->sublist(image_list_[i_cam][i_image]);
            //read each of the rows
            for (int i_row = 0; i_row < num_points_y_; i_row++) {
              param_title = stringstream();
              param_title << "ROW_" << i_row;
              if (grid_data.isSublist(param_title.str())) {
                Teuchos::ParameterList row_data = grid_data.sublist(param_title.str());
                if (row_data.isParameter("X") && row_data.isParameter("Y")) {
                  Teuchos::Array<scalar_t> tArrayX = Teuchos::fromStringToArray<scalar_t>(row_data.get<std::string>("X"));
                  Teuchos::Array<scalar_t> tArrayY = Teuchos::fromStringToArray<scalar_t>(row_data.get<std::string>("Y"));

                  if (tArrayX.size() == num_points_x_ && tArrayY.size() == num_points_x_ ) {
                    for (int i_x = 0; i_x < num_points_x_; i_x++) {
                      image_points_[i_cam][i_image][i_x][i_row].x = tArrayX[i_x];
                      image_points_[i_cam][i_image][i_x][i_row].y = tArrayY[i_x];
                    }
                  }
                  else {
                    t_stream_ << param_title.str() << " the number of points in the X or Y sublist did not match the expected value of " << num_points_x_ << std::endl;
                    if (include_set_[i_image]) t_stream_ << " image set " <<i_image << "will not be included in the calibration" << std::endl;
                    include_set_[i_image]=false;
                  }
                }
              }
              else {
                t_stream_ << param_title.str() << ": X or Y sublist of the row was not found in the file " << std::endl;
                if (include_set_[i_image]) t_stream_ << " image set " << i_image << "will not be included in the calibration" << std::endl;
                include_set_[i_image] = false;
              }
            }
          }
          else {
            t_stream_ << "the intersections sublist {"<< image_list_[i_cam][i_image] <<"} was not found in the file" << std::endl;
            if (include_set_[i_image]) t_stream_ << " image set " << i_image << "will not be included in the calibration" << std::endl;
            include_set_[i_image] = false;
          }
          send_tstream(read_intersection_file_stream_);
        }
      }
    }

    //set the points extracted flag
    points_extracted_ = true;
    intersections_read_from_file_ = true;

    //end the read intersections summary
    t_stream_ << "end reading intersection file" << std::endl;
    send_tstream(read_intersection_file_stream_);

  }

  //set the calibration options (all openCV options listed)
  //calling with no parameters will set the default options
  void
    StereoCalib2::set_Calib_Options(const bool fix_intrinsic,
      const bool use_intrinsic_guess,
      const bool use_extrinsic_guess,
      const bool fix_principal_point,
      const bool fix_focal_length,
      const bool fix_aspect_ratio,
      const bool same_focal_length,
      const bool zero_tangent_dist,
      const bool fix_k1,
      const bool fix_k2,
      const bool fix_k3,
      const bool fix_k4,
      const bool fix_k5,
      const bool fix_k6,
      const bool rational_model,
      const bool thin_prism_model,
      const bool fix_s1_s2_s3_s4,
      const bool tilted_model,
      const bool fix_taux_tauy) {

    //set the calibration options
    calib_options_ = (fix_intrinsic) ? CALIB_FIX_INTRINSIC : 0;
    calib_options_ += (use_intrinsic_guess) ? CALIB_USE_INTRINSIC_GUESS : 0;
    calib_options_ += (use_extrinsic_guess) ? CALIB_USE_EXTRINSIC_GUESS : 0;
    calib_options_ += (fix_principal_point) ? CALIB_FIX_PRINCIPAL_POINT : 0;
    calib_options_ += (fix_focal_length) ? CALIB_FIX_FOCAL_LENGTH : 0;
    calib_options_ += (fix_aspect_ratio) ? CALIB_FIX_ASPECT_RATIO : 0;
    calib_options_ += (same_focal_length) ? CALIB_SAME_FOCAL_LENGTH : 0;
    calib_options_ += (zero_tangent_dist) ? CALIB_ZERO_TANGENT_DIST : 0;
    calib_options_ += (fix_k1) ? CALIB_FIX_K1 : 0;
    calib_options_ += (fix_k2) ? CALIB_FIX_K2 : 0;
    calib_options_ += (fix_k3) ? CALIB_FIX_K3 : 0;
    calib_options_ += (fix_k4) ? CALIB_FIX_K4 : 0;
    calib_options_ += (fix_k5) ? CALIB_FIX_K5 : 0;
    calib_options_ += (fix_k6) ? CALIB_FIX_K6 : 0;
    calib_options_ += (rational_model) ? CALIB_RATIONAL_MODEL : 0;
    calib_options_ += (thin_prism_model) ? CALIB_THIN_PRISM_MODEL : 0;
    calib_options_ += (fix_s1_s2_s3_s4) ? CALIB_FIX_S1_S2_S3_S4 : 0;
    calib_options_ += (tilted_model) ? CALIB_TILTED_MODEL : 0;
    calib_options_ += (fix_taux_tauy) ? CALIB_FIX_TAUX_TAUY : 0;

    //save the options to a string array
    calib_options_text_.clear();
    if (fix_intrinsic) calib_options_text_.push_back(" CALIB_FIX_INTRINSIC");
    if (use_intrinsic_guess) calib_options_text_.push_back(" CALIB_USE_INTRINSIC_GUESS");
    if (use_extrinsic_guess) calib_options_text_.push_back(" CALIB_USE_EXTRINSIC_GUESS");
    if (fix_principal_point) calib_options_text_.push_back(" CALIB_FIX_PRINCIPAL_POINT");
    if (fix_focal_length) calib_options_text_.push_back(" CALIB_FIX_FOCAL_LENGTH");
    if (fix_aspect_ratio) calib_options_text_.push_back(" CALIB_FIX_ASPECT_RATIO");
    if (same_focal_length) calib_options_text_.push_back(" CALIB_SAME_FOCAL_LENGTH");
    if (zero_tangent_dist) calib_options_text_.push_back(" CALIB_ZERO_TANGENT_DIST");
    if (fix_k1) calib_options_text_.push_back(" CALIB_FIX_K1");
    if (fix_k2) calib_options_text_.push_back(" CALIB_FIX_K2");
    if (fix_k3) calib_options_text_.push_back(" CALIB_FIX_K3");
    if (fix_k4) calib_options_text_.push_back(" CALIB_FIX_K4");
    if (fix_k5) calib_options_text_.push_back(" CALIB_FIX_K5");
    if (fix_k6) calib_options_text_.push_back(" CALIB_FIX_K6");
    if (rational_model) calib_options_text_.push_back(" CALIB_RATIONAL_MODEL");
    if (thin_prism_model) calib_options_text_.push_back(" CALIB_THIN_PRISM_MODEL");
    if (fix_s1_s2_s3_s4) calib_options_text_.push_back(" CALIB_FIX_S1_S2_S3_S4");
    if (tilted_model) calib_options_text_.push_back(" CALIB_TILTED_MODEL");
    if (fix_taux_tauy) calib_options_text_.push_back(" CALIB_FIX_TAUX_TAUY");
  }


  //do the calibration using the openCV engine
  float
    StereoCalib2::do_openCV_calibration() {
    return do_openCV_calibration(int_camera_system_);
  }

  //do the calibration using the openCV engine
  float
    StereoCalib2::do_openCV_calibration(DICe::CamSystem & camera_system) {
    namespace CAM = DICe_Camera;

    TEUCHOS_TEST_FOR_EXCEPTION(!points_extracted_, std::runtime_error, "StereoCalib2::do_openCV_calibration: calibration cannot be run until the target points are extracted");

    //clear the camera system
    camera_system.clear_system();
    //clear the summary
    do_calibration_stream_ = stringstream();
    stringstream headder_text;

    t_stream_ << headder("Performing openCV calibration", '*', headder_width_, 1) << std::endl;
    send_tstream(do_calibration_stream_);

    //setup the vectors to hold the calibration parameters
    vector<scalar_t> intrinsics;
    vector<scalar_t> extrinsics;
    vector<vector<scalar_t>> rotation_3x3;
    intrinsics.assign(CAM::MAX_CAM_INTRINSIC_PARAMS, 0.0);
    extrinsics.assign(CAM::MAX_CAM_EXTRINSIC_PARAMS, 0.0);
    rotation_3x3.resize(3);
    rotation_3x3[0].assign(3, 0.0);
    rotation_3x3[1].assign(3, 0.0);
    rotation_3x3[2].assign(3, 0.0);
    rotation_3x3[0][0] = 1.0;
    rotation_3x3[1][1] = 1.0;
    rotation_3x3[2][2] = 1.0;

    //assemble the intersection and object points from the grid and image points 
    t_stream_ << "assemble the intersection and object points" << std::endl;
    send_tstream(do_calibration_stream_);
    assemble_intersection_object_points();

    //do the intrinsic calibration for the initial guess
    Mat cameraMatrix[2], distCoeffs[2];
    t_stream_ << "perform the intrinsic calibration to establish an initial guess" << std::endl;
    send_tstream(do_calibration_stream_);
    cameraMatrix[0] = initCameraMatrix2D(object_points_, intersection_points_[0], image_size_, 0);
    cameraMatrix[1] = initCameraMatrix2D(object_points_, intersection_points_[1], image_size_, 0);

    t_stream_ << "perform the stereo calibration with the following options" << std::endl;
    for (int i_opt = 0; i_opt < calib_options_text_.size(); i_opt++) {
      t_stream_ << "  " << calib_options_text_[i_opt] << std::endl;
    }
    send_tstream(do_calibration_stream_);

    //do the openCV calibration
    Mat R, T, E, F;
    double rms = stereoCalibrate(object_points_, intersection_points_[0], intersection_points_[1],
      cameraMatrix[0], distCoeffs[0],
      cameraMatrix[1], distCoeffs[1],
      image_size_, R, T, E, F,
      calib_options_,
      TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 1e-7));

    for (int i_cam = 0; i_cam < 2; i_cam++) {
      headder_text = stringstream();
      headder_text << "Camera " << i_cam << " intrinsic parameters";
      t_stream_ << std::endl << headder(headder_text.str(), '*', headder_width_, 1) << std::endl;
      //assign the intrinsic and extrinsic values for the first camera
      intrinsics[CAM::CX] = cameraMatrix[i_cam].at<double>(0, 2);
      intrinsics[CAM::CY] = cameraMatrix[i_cam].at<double>(1, 2);
      intrinsics[CAM::FX] = cameraMatrix[i_cam].at<double>(0, 0);
      intrinsics[CAM::FY] = cameraMatrix[i_cam].at<double>(1, 1);
      intrinsics[CAM::K1] = distCoeffs[i_cam].at<double>(0);
      intrinsics[CAM::K2] = distCoeffs[i_cam].at<double>(1);
      intrinsics[CAM::P1] = distCoeffs[i_cam].at<double>(2);
      intrinsics[CAM::P2] = distCoeffs[i_cam].at<double>(3);
      if (distCoeffs[0].cols > 4) intrinsics[CAM::K3] = distCoeffs[i_cam].at<double>(4);
      if (distCoeffs[0].cols > 5) intrinsics[CAM::K4] = distCoeffs[i_cam].at<double>(5);
      if (distCoeffs[0].cols > 6) intrinsics[CAM::K5] = distCoeffs[i_cam].at<double>(6);
      if (distCoeffs[0].cols > 7) intrinsics[CAM::K6] = distCoeffs[i_cam].at<double>(7);
      if (distCoeffs[0].cols > 8) intrinsics[CAM::S1] = distCoeffs[i_cam].at<double>(8);
      if (distCoeffs[0].cols > 9) intrinsics[CAM::S2] = distCoeffs[i_cam].at<double>(9);
      if (distCoeffs[0].cols > 10) intrinsics[CAM::S3] = distCoeffs[i_cam].at<double>(10);
      if (distCoeffs[0].cols > 11) intrinsics[CAM::S4] = distCoeffs[i_cam].at<double>(11);
      if (distCoeffs[0].cols > 12) intrinsics[CAM::T1] = distCoeffs[i_cam].at<double>(12);
      if (distCoeffs[0].cols > 13) intrinsics[CAM::T2] = distCoeffs[i_cam].at<double>(13);
      intrinsics[CAM::LD_MODEL] = CAM::OPENCV_DIS;

      for (int i_parm = 0; i_parm < CAM::MAX_CAM_INTRINSIC_PARAMS - 1; i_parm++) {
        t_stream_ << CAM::camIntrinsicParamsStrings[i_parm] << ": " << intrinsics[i_parm] << std::endl;
      }
      t_stream_ << CAM::camIntrinsicParamsStrings[CAM::LD_MODEL] << ": " << "OPENCV_DIS" << std::endl;

      headder_text = stringstream();
      headder_text << "Camera " << i_cam << " extrinsic parameters";
      t_stream_ << std::endl << headder(headder_text.str(), '*', headder_width_, 1) << std::endl;
      t_stream_ << CAM::camExtrinsicParamsStrings[CAM::TX] << ": " << extrinsics[CAM::TX] << std::endl;
      t_stream_ << CAM::camExtrinsicParamsStrings[CAM::TY] << ": " << extrinsics[CAM::TY] << std::endl;
      t_stream_ << CAM::camExtrinsicParamsStrings[CAM::TZ] << ": " << extrinsics[CAM::TZ] << std::endl;

      headder_text << "Camera " << i_cam << " rotation matrix";
      t_stream_ << std::endl << headder(headder_text.str(), '*', headder_width_, 1) << std::endl;
      t_stream_ << rotation_3x3[0][0] << ", " << rotation_3x3[1][0] << ", " << rotation_3x3[2][0]  << std::endl;
      t_stream_ << rotation_3x3[0][1] << ", " << rotation_3x3[1][1] << ", " << rotation_3x3[2][1] << std::endl;
      t_stream_ << rotation_3x3[0][2] << ", " << rotation_3x3[1][2] << ", " << rotation_3x3[2][2] << std::endl;


      //The first camera has zero value extrinsic values and an identity rotation matrix
      stringstream cam_id;
      cam_id << "Camera_" << i_cam;
      int cam_num;
      cam_num = camera_system.add_Camera(cam_id.str(), image_size_.width, image_size_.height, intrinsics, extrinsics, rotation_3x3);

      //fill the extrinsic values for the second camera
      extrinsics[CAM::TX] = T.at<double>(0);
      extrinsics[CAM::TY] = T.at<double>(1);
      extrinsics[CAM::TZ] = T.at<double>(2);
      for (int i_a = 0; i_a < 3; i_a++) {
        for (int i_b = 0; i_b < 3; i_b++) {
          rotation_3x3[i_a][i_b] = R.at<double>(i_a, i_b);
        }
      }
    }

    camera_system.set_System_Type(DICe_CamSystem::OPENCV);
    headder_text = stringstream();
    headder_text << "Calibration End RMS Value: " << rms;
    t_stream_ << std::endl << headder(headder_text.str(), '*', headder_width_, 1) << std::endl;
    send_tstream(do_calibration_stream_);
    calibration_run_ = true;
    return rms;
  }

  //turns on or off intersection image creation
  void
    StereoCalib2::Draw_Intersection_Image(bool draw_images, std::string dir, std::string modifier, std::string extension) {
    draw_intersection_image_ = draw_images;
    intersection_image_dir_ = dir;
    intersection_image_modifier_ = modifier;
    intersection_image_ext_ = extension;
  }

  //creates the modified file name to save intersection images
  std::string
    StereoCalib2::mod_filename(int i_cam, int i_image) {
    stringstream out_file_name;
    if (intersection_image_dir_ == "") {
      intersection_image_dir_ = image_directory_;
      //if we default to the image directory make sure there is a modifier so 
      //we don't overwrite the images
      if (intersection_image_modifier_ == "") {
        intersection_image_modifier_ = "_pnts";
      }
    }
    // create the output image name by modifying the image name
    size_t lastindex = image_list_[i_cam][i_image].find_last_of(".");
    string rawname = image_list_[i_cam][i_image].substr(0, lastindex);
    out_file_name = stringstream();
    //out_file_name << image_directory_ << rawname << "_markers.tif";
    out_file_name << intersection_image_dir_ << rawname << intersection_image_modifier_ << intersection_image_ext_;
    return out_file_name.str();
  }

  //makes a simple headder line
  std::string
    StereoCalib2::headder(std::string title, char headder_char, int len, int justification) {
    int title_len = title.length();
    std::string pre_string = "";
    std::string post_string = "";
    size_t length;
    switch (justification) {
    case 0: //left justified
      if (title_len < len) {
        length = len - title_len;
        post_string.assign(length, headder_char);
      }
      break;
    case 1: //center justified
      if (title_len < len - 1) {
        length = (len - title_len)/2;
        pre_string.assign(length, headder_char);
        length = len - title_len - length;
        post_string.assign(length, headder_char);
      }
      break;
    case 2: //right justified
      if (title_len < len) {
        length = len - title_len;
        pre_string.assign(length, headder_char);
      }
      break;
    }
    return pre_string + title + post_string;
  }

  //writes the calibration file  (one overload)
  void
    StereoCalib2::write_calibration_file(std::string filename, bool allfields, DICe::CamSystem & camsystem) {
    camsystem.write_calibration_file(filename, allfields);
  }
  void
    StereoCalib2::write_calibration_file(std::string filename, bool allfields) {
    int_camera_system_.write_calibration_file(filename, allfields);
  }

  //returns the calibration summary as a string
  void
    StereoCalib2::get_Calibration_Summary(std::string & summary) {
    t_stream_ = stringstream();
    if (!intersections_read_from_file_) {
      if (image_list_set_) t_stream_ << image_list_stream_.str();
      if (target_specified_) t_stream_ << target_info_stream_.str();
      if (points_extracted_) t_stream_ << extraction_stream_.str();
    }
    else{
      t_stream_ << read_intersection_file_stream_.str();
    }
    if (intersetions_written_to_file_) t_stream_ << write_intersection_file_stream_.str();
    if (calibration_run_) t_stream_ << do_calibration_stream_.str();
    summary = t_stream_.str();
  }


  //adds to area stringstreams and displays on the console if verbose
  void 
    StereoCalib2::send_tstream(stringstream & save_stream) {
      save_stream << t_stream_.str();
      if (verbose_)
        cout << t_stream_.str();
      t_stream_ = stringstream();
  }

} //end of DICe namespace