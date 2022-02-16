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

#include <DICe.h>
#include <DICe_Triangulation.h>
#include <DICe_Camera.h>

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
  if(argc!=5){
    std::cout << "Error. Usage: DICe_ImageDistorter <cal_file> <camera_id> <image_file_in> <image_file_out>" << std::endl;
    exit(1);
  }

  const std::string cal_file = argv[1];
  const int cam_id = std::atoi(argv[2]);
  const std::string image_file_in = argv[3];
  const std::string image_file_out = argv[4];

  std::cout << "DICe_ImageDistorter: distorting image " << image_file_in << " using cal file " << cal_file << " camera id " << cam_id << " output image " << image_file_out << std::endl;

  TEUCHOS_TEST_FOR_EXCEPTION(cam_id!=0&&cam_id!=1,std::runtime_error,"");

  Teuchos::RCP<DICe::Triangulation> tri = Teuchos::rcp(new DICe::Triangulation(cal_file));
  const int w = tri->image_width();
  const int h = tri->image_height();

  DICe::Image img_in(image_file_in.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION(w!=img_in.width(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(h!=img_in.height(),std::runtime_error,"");
  DICe::Image img_out(img_in.width(),img_in.height(),0.0);

  Teuchos::RCP<Camera> camera = tri->camera_system()->camera(cam_id);
  const scalar_t fx = (*camera->intrinsics())[DICe::Camera::FX];
  const scalar_t fy = (*camera->intrinsics())[DICe::Camera::FY];
  const scalar_t fs = (*camera->intrinsics())[DICe::Camera::FS];
  const scalar_t cx = (*camera->intrinsics())[DICe::Camera::CX];
  const scalar_t cy = (*camera->intrinsics())[DICe::Camera::CY];
  std::vector<scalar_t> dist_coords_x(w*h,0.0);
  std::vector<scalar_t> dist_coords_y(w*h,0.0);
  std::vector<scalar_t> sens_coords_x(w*h,0.0);
  std::vector<scalar_t> sens_coords_y(w*h,0.0);
  for(int_t j=0;j<h;++j){
    for(int_t i=0;i<w;++i){
      dist_coords_x[j*w+i] = i;
      dist_coords_y[j*w+i] = j;
    }
  }
  camera->image_to_sensor(dist_coords_x,dist_coords_y,sens_coords_x,sens_coords_y);
  for(int_t j=0;j<h;++j){
    for(int_t i=0;i<w;++i){
      // convert the sensor coords back to image coords using no distortion
      scalar_t undist_x = sens_coords_x[j*w+i]*fx + sens_coords_y[j*w+i]*fs + cx;
      scalar_t undist_y = sens_coords_y[j*w+i]*fy + cy;
      img_out.intensities()[j*w+i] = img_in.interpolate_keys_fourth(undist_x,undist_y);
    }
  }
  img_out.write(image_file_out);

  DICe::finalize();
  return 0;
}

