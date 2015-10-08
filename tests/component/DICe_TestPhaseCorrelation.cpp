// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Shape.h>
#include <DICe_FFT.h>

#include <iostream>
#include <fstream>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  // initialize kokkos
  Kokkos::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "creating the mask" << std::endl;
  std::vector<int_t> coords_x(7);
  std::vector<int_t> coords_y(7);
  coords_x[0] = 185;  coords_y[0] = 235;
  coords_x[1] = 178;  coords_y[1] = 249;
  coords_x[2] = 198;  coords_y[2] = 289;
  coords_x[3] = 219;  coords_y[3] = 286;
  coords_x[4] = 228;  coords_y[4] = 255;
  coords_x[5] = 305;  coords_y[5] = 220;
  coords_x[6] = 290;  coords_y[6] = 187;
  Teuchos::RCP<DICe::Polygon> polygon = Teuchos::rcp(new DICe::Polygon(coords_x,coords_y));
  DICe::multi_shape boundary;
  boundary.push_back(polygon);
  DICe::Conformal_Area_Def area_def(boundary);

  // read the reference image
  Teuchos::RCP<Image> ref = Teuchos::rcp(new Image("./images/pc_ref.tif"));
  ref->apply_mask(area_def);
  ref->write_tiff("pc_ref_mask.tif");
  *outStream << "fft of reference image" << std::endl;
  Teuchos::RCP<Image> ref_fft = image_fft(ref,true,true,100.0,true,true,false);
  ref_fft->write_tiff("ref_fft.tif");
  *outStream << "polar transform of reference fft image" << std::endl;
  Teuchos::RCP<Image> ref_pol = polar_transform(ref_fft,true,false);
  ref_pol->write_tiff("ref_polar.tif");
  int_t h_2 = ref_pol->height()/2;
  int_t h_4 = ref_pol->height()/4;

  // TODO create a vector with the correct solution

  // cycle through the pc_ images
  for(int_t i=0;i<12;++i){ // 12
    std::stringstream name;
    name << "./images/pc_0";
    if(i<10)
      name << "0" << i << ".tif";
    else
      name << i << ".tif";
    *outStream << "processing image: " << name.str() << std::endl;
    Teuchos::RCP<Image> img = Teuchos::rcp(new Image(name.str().c_str()));
    //*outStream << "  fft of image" << std::endl;
    Teuchos::RCP<Image> img_fft = image_fft(img,true,true,100.0,true,true,false);
    std::stringstream fftName;
    fftName << "img_fft_" << i << ".tif";
    img_fft->write_tiff(fftName.str());
    //*outStream << "  polar transform of fft" << std::endl;
    Teuchos::RCP<Image> img_pol = polar_transform(img_fft,true,false);
    std::stringstream polar_name_ss;
    polar_name_ss << "_" << i;
    std::string polar_name = polar_name_ss.str();
    img_pol->set_file_name(polar_name);
    polar_name_ss << ".tif";
    img_pol->write_tiff(polar_name_ss.str());
    //*outStream << "  phase correlating angle" << std::endl;

    // phase correlate
    scalar_t theta = 0.0;
    phase_correlate_row(img_pol,ref_pol,h_2,theta,true);
    scalar_t theta_orig = theta;
    theta *= -180.0/DICE_PI;
    if (theta > 180) theta = -360 + theta;
    if (theta < -180) theta = 360 + theta;
    scalar_t angle2 = (theta < 0) ? theta + 180 : theta - 180;
    *outStream << "***orig: " << theta_orig <<  " theta " << theta << " " << angle2 << std::endl;

    // fft correlate the two polar fft images:
    //scalar_t r = 0.0;
    //scalar_t t = 0.0;
    //phase_correlate_x_y(img_pol,ref_pol,t,r,true);
    //*outStream << "  radius: " << r << " theta: " << t << std::endl;
  }

  *outStream << "--- End test ---" << std::endl;

  // finalize kokkos
  Kokkos::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

