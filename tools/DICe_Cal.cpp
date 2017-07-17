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

#include <DICe.h>
#include <DICe_Parser.h>
#include <DICe_StereoCalib.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <cassert>

using namespace std;
using namespace DICe;

/// Initializes the cross correlation between two images from the left and right camera, respectively

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // read the parameters from the input file:
  if(argc!=2 && argc!=3){ // executable and input file
    std::cout << "Usage: DICe_Cal <input_file.xml> [output_file=cal.txt]" << std::endl;
    exit(1);
  }

  std::string input_file = argv[1];
  std::string output_file = "cal.txt";
  if(argc==3)
    output_file = argv[2];
  DEBUG_MSG("output will be written to " << output_file);
  DEBUG_MSG("parsing input file: " << input_file);

  Teuchos::RCP<Teuchos::ParameterList> inputParams = Teuchos::rcp( new Teuchos::ParameterList() );
  Teuchos::Ptr<Teuchos::ParameterList> inputParamsPtr(inputParams.get());
  Teuchos::updateParametersFromXmlFile(input_file, inputParamsPtr);
  TEUCHOS_TEST_FOR_EXCEPTION(inputParams==Teuchos::null,std::runtime_error,"");
#ifdef DICE_DEBUG_MSG
  inputParams->print(std::cout);
#endif

  // test that all the params that are needed are there
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::image_folder),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::reference_image_index),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::start_image_index),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::end_image_index),std::runtime_error,"");
  //TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::cal_target_width),std::runtime_error,"");
  //TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::cal_target_height),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::cal_target_spacing_size),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::cal_target_has_adaptive),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::cal_target_is_inverted),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::cal_target_block_size),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::cal_target_binary_constant),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::cal_mode),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::skip_image_index),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::stereo_left_suffix),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::stereo_right_suffix),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::image_file_prefix),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::image_file_extension),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(!inputParams->isParameter(DICe::num_file_suffix_digits),std::runtime_error,"");

  std::vector<std::string> image_files;
  std::vector<std::string> stereo_image_files;
  DICe::decipher_image_file_names(inputParams,image_files,stereo_image_files);
  TEUCHOS_TEST_FOR_EXCEPTION(stereo_image_files.size() <= 0|| image_files.size()<=0,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(stereo_image_files.size() != image_files.size(),std::runtime_error,"");
  const int_t num_images = image_files.size() -1;
  std::vector<std::string> images(num_images*2,"");
  for(size_t i=0;i<image_files.size()-1;++i){
    images[i*2+0] = image_files[i+1];
    images[i*2+1] = stereo_image_files[i+1];
    DEBUG_MSG("cal file left: " << images[i*2+0] << " right: " << images[i*2+1]);
  }
//  const int_t mode = inputParams->get<int_t>(DICe::cal_mode);
//  const int_t target_width = inputParams->get<int_t>(DICe::cal_target_width);
//  const int_t target_height = inputParams->get<int_t>(DICe::cal_target_height);
//  const double spacing_size = inputParams->get<double>(DICe::cal_target_spacing_size);
//  const int_t threshold = inputParams->get<int_t>(DICe::cal_binary_threshold,30);

  const float rms = StereoCalibDotTarget(inputParams,images,output_file);
//  const float rms = StereoCalib(mode, images, target_width, target_height, spacing_size, threshold, true, false, output_file);

  DICe::finalize();

  if(rms < 0.0){
    std::cout << "Error, stereo calibration failed" << std::endl;
    return -1;
  }
  if(rms > 1.0){
    std::cout << "Warning, RMS error is large: " << rms << ". Should be under 0.5." << std::endl;
  }
  return 0;
}

