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

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_OpenCVServerUtils.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint = argc - 1;
  int_t error_flag = 0;
  work_t error_tol = 1.0E-4;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  std::vector<std::string> arguments = {"./DICe_OpenCVServer",
    "../images/CalB-sys2-0001_0.jpeg",
    "./cal_0.png",
    "../images/CalB-sys2-0001_1.jpeg",
    "./cal_1.png",
    "filter:adaptive_threshold",
    "filter_mode","1",
    "threshold_mode","0",
    "block_size","75",
    "binary_constant","25.0",
    "filter:binary_threshold",
    "filter_mode","1"};
  std::vector<char*> tmp_argv;
  for (const auto& arg : arguments)
      tmp_argv.push_back((char*)arg.data());
  tmp_argv.push_back(nullptr);

  Teuchos::ParameterList opencv_params = parse_filter_string(tmp_argv.size()-1,tmp_argv.data());
  Teuchos::ParameterList io_files = opencv_params.get<Teuchos::ParameterList>(DICe::opencv_server_io_files,Teuchos::ParameterList());
  if(io_files.numParams()!=2){
    *outStream << "error, file name reading incorrect, wrong number of files: " << io_files.numParams() << " should be 2" << std::endl;
    error_flag++;
  }
  if(io_files.get<std::string>(arguments[1],"")!=arguments[2]){
    *outStream << "error, file name parsing incorrect" << std::endl;
    error_flag++;
  }
  if(io_files.get<std::string>(arguments[3],"")!=arguments[4]){
    *outStream << "error, file name parsing incorrect" << std::endl;
    error_flag++;
  }
  Teuchos::ParameterList filters = opencv_params.get<Teuchos::ParameterList>(DICe::opencv_server_filters,Teuchos::ParameterList());
  if(filters.numParams()!=2){
    *outStream << "error, filter reading incorrect, wrong number of filters: " << filters.numParams() << " should be 2" << std::endl;
    error_flag++;
  }
  // iterate the filters
  for(Teuchos::ParameterList::ConstIterator it=filters.begin();it!=filters.end();++it){
    Teuchos::ParameterList filter_params = filters.get<Teuchos::ParameterList>(it->first,Teuchos::ParameterList());
    if(it==filters.begin()){
      if(it->first!="adaptive_threshold"){
        *outStream << "error, first filter name incorrect" << std::endl;
        error_flag++;
      }
      if(filter_params.get<int_t>(arguments[6],0)!=std::atoi(arguments[7].c_str())){
        *outStream << "error, first filter, first option is incorrect" << std::endl;
        error_flag++;
      }
      if(filter_params.get<int_t>(arguments[8],0)!=std::atoi(arguments[9].c_str())){
        *outStream << "error, first filter, second option is incorrect" << std::endl;
        error_flag++;
      }
      if(filter_params.get<int_t>(arguments[10],0)!=std::atoi(arguments[11].c_str())){
        *outStream << "error, first filter, third option is incorrect" << std::endl;
        error_flag++;
      }
      if(filter_params.get<double>(arguments[12],0.0)!=std::strtod(arguments[13].c_str(),NULL)){
        *outStream << "error, first filter, fourth option is incorrect" << std::endl;
        error_flag++;
      }
    } // end filter a
    if(it==filters.end()){
      if(it->first!="binary_threshold"){
        *outStream << "error, second filter name incorrect" << std::endl;
        error_flag++;
      }
      if(filter_params.get<int_t>(arguments[15],0)!=std::atoi(arguments[16].c_str())){
        *outStream << "error, second filter, fist option is incorrect" << std::endl;
        error_flag++;
      }
    } // end filter a
  }

  // find the checkerboard corners in an image
  *outStream << "finding the checkerboard corners in an image" << std::endl;
  std::vector<std::string> cb_args = {"./DICe_OpenCVServer",
    "../images/left03.jpg",
    "./cb_out.png",
    "filter:checkerboard_targets",
    "num_cal_fiducials_x","9",
    "num_cal_fiducials_y","6"};
  std::vector<char*> tmp_cb_argv;
  for (const auto& arg : cb_args)
    tmp_cb_argv.push_back((char*)arg.data());
  tmp_cb_argv.push_back(nullptr);
  int_t error_code = opencv_server(tmp_cb_argv.size()-1,tmp_cb_argv.data());
  if(error_code!=0){
    *outStream << "error, running the opencv server failed" << std::endl;
    error_flag++;
  }
  // compare to gold image
  Teuchos::RCP<DICe::Image> cb_image = Teuchos::rcp(new DICe::Image("cb_out.png"));
  Teuchos::RCP<DICe::Image> cb_image_gold = Teuchos::rcp(new DICe::Image("../images/cb_out.png"));
  work_t cb_diff = cb_image->diff(cb_image_gold);
  *outStream << "checkerboard image diff: " << cb_diff << std::endl;
  if(cb_diff>error_tol){
    *outStream << "error, running the opencv server failed for the checkerboard example" << std::endl;
    error_flag++;
  }

  *outStream << "testing to make sure bad parameters for the checkerboard case don't work" << std::endl;
  std::vector<std::string> bad_cb_args = {"./DICe_OpenCVServer",
    "../images/left03.jpg",
    "./cb_out.png",
    "filter:checkerboard_targets",
    "num_cal_fiducials_x","9"};
  std::vector<char*> tmp_bad_cb_argv;
  for (const auto& arg : bad_cb_args)
    tmp_bad_cb_argv.push_back((char*)arg.data());
  tmp_bad_cb_argv.push_back(nullptr);
  *outStream << "this next call is supposed to error" << std::endl;
  error_code = opencv_server(tmp_bad_cb_argv.size()-1,tmp_bad_cb_argv.data());
  if(error_code!=2){
    *outStream << "error, opencv server should have failed with error code 2, but instead returned code " << error_code << std::endl;
    error_flag++;
  }

  *outStream << "testing a dot grid calibration target dot extraction using opencv" << std::endl;

  std::vector<std::string> dot_args = {"./DICe_OpenCVServer",
    "../images/CalB-sys2-0001_0.jpeg",
    "./dot_out.png",
    "filter:dot_targets",
    "threshold_start","20",
    "threshold_end","250",
    "threshold_step","5",
    "num_cal_fiducials_x","10",
    "num_cal_fiducials_y","8",
    "cal_origin_x","2",
    "cal_origin_y","2",
    "num_cal_fiducials_origin_to_x_marker","6",
    "num_cal_fiducials_origin_to_y_marker","4",
    "cal_target_type","BLACK_ON_WHITE_W_DONUT_DOTS",
    "dot_tol","0.26"};
  std::vector<char*> tmp_dot_argv;
  for (const auto& arg : dot_args)
    tmp_dot_argv.push_back((char*)arg.data());
  tmp_dot_argv.push_back(nullptr);
  error_code = opencv_server(tmp_dot_argv.size()-1,tmp_dot_argv.data());
  if(error_code!=0){
    *outStream << "error, running the opencv server for dot marker example failed" << std::endl;
    error_flag++;
  }
  // compare to gold image
  Teuchos::RCP<DICe::Image> dot_image = Teuchos::rcp(new DICe::Image("dot_out.png"));
  Teuchos::RCP<DICe::Image> dot_image_gold = Teuchos::rcp(new DICe::Image("../images/dot_out.png"));
  work_t dot_diff = dot_image->diff(dot_image_gold);
  *outStream << "dot marker image diff: " << dot_diff << std::endl;
  if(dot_diff>error_tol){
    *outStream << "error, running the opencv server failed for the dot marker example" << std::endl;
    error_flag++;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (error_flag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return error_flag;

}


