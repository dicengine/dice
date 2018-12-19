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

/*! \file  DICe_TrackingMovieMaker.cpp
    \brief Utility to produce a movie of the tracking results
*/

#include <DICe.h>
#include <DICe_Parser.h>
#include <DICe_ImageIO.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "opencv2/opencv.hpp"

#include <cassert>
#include <fstream>

using namespace cv;
using namespace DICe;

std::string results_file_name(const std::string & output_folder,
  const int_t roi,
  const int_t num_roi){

  // determine the number of digits to append:
  int_t num_digits_subset = 0;
  int_t num_digits_total = 0;
  int_t decrement_total = num_roi;
  int_t decrement_roi = roi;
  while (decrement_total){decrement_total /= 10; num_digits_total++;}
  if(roi==0) num_digits_subset = 1;
  else
    while (decrement_roi){decrement_roi /= 10; num_digits_subset++;}
  int_t num_zeros = num_digits_total - num_digits_subset;
  // determine the file name for this roi
  std::stringstream fName;
  fName << output_folder << "DICe_solution_";
  for(int_t i=0;i<num_zeros;++i)
    fName << "0";
  fName << roi;
  fName << ".txt";
  return fName.str();
}

int main(int argc, char *argv[]) {

  /// usage ./DICe_TrackingMovieMaker <path_to_input.xml>

  DICe::initialize(argc, argv);

  //Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
  *outStream << "DICe_TrackingMovieMaker(): creating avi movie from results" << std::endl;

  DEBUG_MSG("User specified " << argc << " arguments");
  for(int_t i=0;i<argc;++i){
    DEBUG_MSG(argv[i]);
  }

  int error_code = 0;

  // parse the input file
  std::string input_file = argv[1];
  Teuchos::RCP<Teuchos::ParameterList> inputParams = Teuchos::rcp( new Teuchos::ParameterList() );
  Teuchos::Ptr<Teuchos::ParameterList> inputParamsPtr(inputParams.get());
  Teuchos::updateParametersFromXmlFile(input_file, inputParamsPtr);
  if(inputParams==Teuchos::null){
    *outStream << "Could not read input file: " << input_file << std::endl;
    return -1;
  }
  if(inputParams->numParams()<=0){
    *outStream << "Input file load failed: " << input_file << std::endl;
  }
  // determine the results files location
  std::string output_folder;
  if(!inputParams->isParameter("output_folder")){
    *outStream << "Missing required output folder parameter" << std::endl;
    return -1;
  }
  output_folder = inputParams->get<std::string>("output_folder");
  DEBUG_MSG("Using results files from folder: " << output_folder);
  // parse the ROIs from the subset file
  std::string subset_file;
  if(!inputParams->isParameter("subset_file")){
    *outStream << "Missing required subset file parameter" << std::endl;
    return -1;
  }
  subset_file = inputParams->get<std::string>("subset_file");
  DEBUG_MSG("Using subset file: " << subset_file);
  // readin the ROI's and the centroids
  int_t ref_w = -1;
  int_t ref_h = -1;
  Teuchos::RCP<DICe::Subset_File_Info> subset_info = DICe::read_subset_file(subset_file,ref_w,ref_h);
  Teuchos::RCP<std::vector<scalar_t> > subset_centroids = subset_info->coordinates_vector;
  const int_t num_roi = subset_centroids->size()/2;
  DEBUG_MSG("Found " << num_roi << " ROIs");
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > conformal_area_defs = subset_info->conformal_area_defs;
  if((int_t)conformal_area_defs->size()!=num_roi){
    *outStream << "The number of conformal area defs " << conformal_area_defs->size() <<
        " does not match the number of ROIs " << num_roi << std::endl;
    return -1;
  }
  // determine the cine file to use as the
  if(!inputParams->isParameter("cine_file")){
    *outStream << "Error, cine file not defined in input file" << std::endl;
    return -1;
  }
  std::string cine_file = inputParams->get<std::string>("cine_file");
  DEBUG_MSG("Using cine_file: " << cine_file);

  // parse the header from the zeroth output file
  std::string zero_output_file = results_file_name(output_folder,0,num_roi);
  std::fstream zero_file(zero_output_file.c_str(), std::ios_base::in);
  if(!zero_file.good()){
    *outStream << "Error, could not read the results file for the zeroth roi" << std::endl;
    return -1;
  }
  // read the header from the results file:
  std::vector<std::string> fields = tokenize_line(zero_file," ,",true);
  // make sure the disp and rotation fields exist and get their ids
  int_t frame_col_id = -1;
  int_t disp_x_col_id = -1;
  int_t disp_y_col_id = -1;
  int_t rot_col_id = -1;
  int_t sigma_col_id = -1;
  for(int_t i=0;i<(int_t)fields.size();++i){
    DEBUG_MSG("field col: " << fields[i]);
    if(fields[i]=="FRAME")frame_col_id = i;
    if(fields[i]=="DISPLACEMENT_X")disp_x_col_id = i;
    if(fields[i]=="DISPLACEMENT_Y")disp_y_col_id = i;
    if(fields[i]=="ROTATION_Z")rot_col_id = i;
    if(fields[i]=="SIGMA")sigma_col_id = i;
  }
  if(frame_col_id<0||disp_x_col_id<0||disp_y_col_id<0||rot_col_id<0||sigma_col_id<0){
    *outStream << "Error, could not find all the required fields in the output file header" << std::endl;
    return -1;
  }
//  int_t num_frames = 0;
//  while(!zero_file.eof()){
//    std::vector<std::string> line = tokenize_line
//  }
  zero_file.close();

  // data structure to load all the results data in:
  // first index is the file
  // second index is the frame for each field
  std::vector<std::vector<int_t> > frames;
  std::vector<std::vector<scalar_t> > disp_x;
  std::vector<std::vector<scalar_t> > disp_y;
  std::vector<std::vector<scalar_t> > rot_z;
  std::vector<std::vector<scalar_t> > sigma;
  for(int_t i=0;i<num_roi;++i){
    frames.push_back(std::vector<int_t>());
    disp_x.push_back(std::vector<scalar_t>());
    disp_y.push_back(std::vector<scalar_t>());
    rot_z.push_back(std::vector<scalar_t>());
    sigma.push_back(std::vector<scalar_t>());
  }

  // read the results for each ROI
  for(int_t roi=0;roi<num_roi;++roi){
    const std::string fName = results_file_name(output_folder,roi,num_roi);
    DEBUG_MSG("results file " << fName);
    std::fstream results_file(fName.c_str(), std::ios_base::in);
    if(!results_file.good()){
      *outStream << "Error reading results file: " << fName << std::endl;
      return -1;
    }
    // skip the header line (assuming that it's only one line)
    tokenize_line(results_file);
    // for each line, read the frame and fields
    while(!results_file.eof()){
      std::vector<std::string> tokens = tokenize_line(results_file," ,",true);
      if(tokens.size()<9)continue; // there should be 9 columns in the standard output files
      frames[roi].push_back(std::atoi(tokens[frame_col_id].c_str()));
      disp_x[roi].push_back(std::strtod(tokens[disp_x_col_id].c_str(),NULL));
      disp_y[roi].push_back(std::strtod(tokens[disp_y_col_id].c_str(),NULL));
      rot_z[roi].push_back(std::strtod(tokens[rot_col_id].c_str(),NULL));
      sigma[roi].push_back(std::strtod(tokens[sigma_col_id].c_str(),NULL));
    }
    results_file.close();
  } // roi loop

  // check that all the vecs are the same size and that the frame numbers match
  int_t num_frames = frames[0].size();
  for(int_t roi=0;roi<num_roi;++roi){
    if((int_t)frames[roi].size()!=num_frames||(int_t)disp_x[roi].size()!=num_frames||
        (int_t)disp_y[roi].size()!=num_frames||(int_t)rot_z[roi].size()!=num_frames||(int_t)sigma[roi].size()!=num_frames){
      *outStream << "Error, not all of the results files have the same number of frames" << std::endl;
      return -1;
    }
    for(int_t i=0;i<num_frames;++i){
      if(frames[roi][i]!=frames[0][i]){
        *outStream << "Error, the frame numbers for each resuls file do not match" << std::endl;
      }
    }
  }
  // checked above that the frame numbers are all the same for each file so using the zeroeth
  // iterate the cine frames
  const std::string ext(".cine");
  std::string trimmed_cine_name = cine_file;
  if(trimmed_cine_name.size() > ext.size() && trimmed_cine_name.substr(trimmed_cine_name.size() - ext.size()) == ".cine" )
  {
     trimmed_cine_name = trimmed_cine_name.substr(0, trimmed_cine_name.size() - ext.size());
  }else{
    *outStream << "Error, invalid cine file: " << cine_file << std::endl;
    return -1;
  }
  std::string banner1 = "Digital Image Correlation Engine (DICe)";
  std::stringstream banner2;
  banner2 << GITSHA1;
  // generate a video
  std::stringstream movie_name;
  movie_name << output_folder << "DICe_results.avi";
  // get the image size:
  int_t img_w = 0;
  int_t img_h = 0;
  // cine file name needs to be decorated with a frame number:
  std::stringstream cine_ss_temp;
  cine_ss_temp << trimmed_cine_name << "_0.cine";
  DICe::utils::read_image_dimensions(cine_ss_temp.str().c_str(),img_w,img_h);
  VideoWriter video(movie_name.str(),-1,10, Size(img_w,img_h));
  //VideoWriter video(movie_name.str(),CV_FOURCC('M','J','P','G'),10, Size(img_w,img_h));
  for(int_t i=0;i<num_frames;++i){
    const int_t frame = frames[0][i];
    // load the cine image for each frame
    std::stringstream cine_ss;
    cine_ss << trimmed_cine_name << "_" << frame << ".cine";
    DEBUG_MSG("Loading cine image " << cine_ss.str());
    DICe::Image img(cine_ss.str().c_str());
    // convert the image to an opencv mat:
    Mat mat_img(img.height(),img.width(), CV_8UC1);
    for(int_t y=0;y<img.height();++y){
      for(int_t x=0;x<img.width();++x){
        mat_img.at<uchar>(y,x) = std::floor(img(x,y));
      }
    }
    Mat out_img(mat_img.size(), CV_8UC3);
    cvtColor(mat_img, out_img, cv::COLOR_GRAY2RGB);

    // add some text to the image
    std::stringstream banner3;
    banner3 << "frame " << frame;
    putText(out_img, banner1, Point(30,30),
      FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 1, cv::LINE_AA);
    putText(out_img, banner2.str(), Point(30,50),
      FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 1, cv::LINE_AA);
    putText(out_img, banner3.str(), Point(30,70),
      FONT_HERSHEY_DUPLEX, 0.5, Scalar(255,255,255), 1, cv::LINE_AA);

    for(int_t roi=0;roi<num_roi;++roi){
      const scalar_t u = disp_x[roi][i];
      const scalar_t v = disp_y[roi][i];
      const scalar_t sig = sigma[roi][i];
      const scalar_t cost = std::cos(rot_z[roi][i]);
      const scalar_t sint = std::sin(rot_z[roi][i]);
      const scalar_t cx = (*subset_centroids)[roi*2+0];
      const scalar_t cy = (*subset_centroids)[roi*2+1];
      Scalar roi_color = sig<0.0?Scalar(0,0,255):Scalar(0,255,255);
      //DEBUG_MSG("centroid subset " << roi << " " << cx << " " << cy);
      TEUCHOS_TEST_FOR_EXCEPTION(conformal_area_defs->find(roi)==conformal_area_defs->end(),std::runtime_error,"");
      // assume there is only one boundary polygon
      // compute the deformed vertices
      Teuchos::RCP<DICe::Polygon> boundary_polygon = Teuchos::rcp_dynamic_cast<DICe::Polygon>((*conformal_area_defs->find(roi)->second.boundary())[0]);
      TEUCHOS_TEST_FOR_EXCEPTION(boundary_polygon==Teuchos::null,std::runtime_error,"Error, failed cast to polygon.");
      const int_t num_boundary_vertices = boundary_polygon->num_vertices();
      std::vector<Point> contour;
      for(int_t vert=0;vert<num_boundary_vertices;++vert){
        const scalar_t dx = (*boundary_polygon->vertex_coordinates_x())[vert] - cx;
        const scalar_t dy = (*boundary_polygon->vertex_coordinates_y())[vert] - cy;
        const scalar_t vx = cost*dx - sint*dy + u + cx;
        const scalar_t vy = sint*dx + cost*dy + v + cy;
        contour.push_back(Point(vx,vy));
      }
      const Point *pts = (const Point*) Mat(contour).data;
      // draw the roi
      polylines(out_img, &pts,&num_boundary_vertices, 1,
                true,       // draw closed contour (i.e. joint end to start)
                roi_color,// colour BGR ordering
                1,            // line thickness
                cv::LINE_AA, 0);
      // draw any excluded regions as well
      int_t num_excluded = 0;
      if(conformal_area_defs->find(roi)->second.has_excluded_area())
        num_excluded = (*conformal_area_defs->find(roi)->second.excluded_area()).size();
      for(int_t ex=0;ex<num_excluded;++ex){
        Teuchos::RCP<DICe::Polygon> excluded_polygon = Teuchos::rcp_dynamic_cast<DICe::Polygon>((*conformal_area_defs->find(roi)->second.excluded_area())[ex]);
        TEUCHOS_TEST_FOR_EXCEPTION(excluded_polygon==Teuchos::null,std::runtime_error,"Error, failed cast to polygon.");
        const int_t num_ex_vertices = excluded_polygon->num_vertices();
        std::vector<Point> ex_contour;
        for(int_t vert=0;vert<num_ex_vertices;++vert){
          const scalar_t dx = (*excluded_polygon->vertex_coordinates_x())[vert] - cx;
          const scalar_t dy = (*excluded_polygon->vertex_coordinates_y())[vert] - cy;
          const scalar_t vx = cost*dx - sint*dy + u + cx;
          const scalar_t vy = sint*dx + cost*dy + v + cy;
          ex_contour.push_back(Point(vx,vy));
        }
        const Point *ex_pts = (const Point*) Mat(ex_contour).data;
        // draw the roi
        polylines(out_img, &ex_pts,&num_ex_vertices, 1,
          true,       // draw closed contour (i.e. joint end to start)
          Scalar(0,0,255),// colour RGB ordering
          1,            // line thickness
          cv::LINE_AA, 0);
      }// end excluded shapes
      // draw centroids on the ROIs as a small circle
      circle(out_img,Point(cx+u,cy+v),2,Scalar(0,255,0),-1);
      std::stringstream pt_text;
      pt_text << roi;
      // denote the ROI ids
      putText(out_img, pt_text.str(), Point(cx+u+5,cy+v+5),FONT_HERSHEY_DUPLEX,0.5,Scalar(0,255,0), 1, cv::LINE_AA);
    }
    video.write(out_img);
  }
  video.release();

  *outStream << "DICe_TrackingMovieMaker(): avi movie made successfully!" << std::endl;

  DICe::finalize();

  return error_code;
}

