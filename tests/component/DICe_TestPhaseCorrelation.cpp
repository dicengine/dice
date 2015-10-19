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
  //ref->write_tiff("pc_ref_mask.tif");
  *outStream << "fft of reference image" << std::endl;
  Teuchos::RCP<Image> ref_fft = image_fft(ref,true,true,100.0,true,true);
  //ref_fft->write_tiff("ref_fft.tif");
  *outStream << "polar transform of reference fft image" << std::endl;
  Teuchos::RCP<Image> ref_pol = polar_transform(ref_fft,true);
  //ref_pol->write_tiff("ref_polar.tif");
  int_t h_2 = ref_pol->height()/2;
  int_t h_4 = ref_pol->height()/4;
  int_t w_2 = ref_pol->width()/2;

  // vector with the correct solution for rotations
  std::vector<scalar_t> exact_rot(12);
  exact_rot[0]  =  0.0;
  exact_rot[1]  =  30.0;
  exact_rot[2]  =  60.0;
  exact_rot[3]  =  90.0;
  exact_rot[4]  =  120.0;
  exact_rot[5]  =  150.0;
  exact_rot[6]  = -180.0;
  exact_rot[7]  = -150.0;
  exact_rot[8]  = -120.0;
  exact_rot[9]  = -90.0;
  exact_rot[10] = -60.0;
  exact_rot[11] = -30.0;
  std::vector<scalar_t> exact_ux(12);
  std::vector<scalar_t> exact_uy(12);
  exact_ux[0]  =   9; exact_uy[0]  = -12;
  exact_ux[1]  =   2; exact_uy[1]  =  -2;
  exact_ux[2]  =  12; exact_uy[2]  = -10;
  exact_ux[3]  =  10; exact_uy[3]  = -12;
  exact_ux[4]  =   1; exact_uy[4]  = -39;
  exact_ux[5]  =  -8; exact_uy[5]  = -38;
  exact_ux[6]  = -17; exact_uy[6]  = -34;
  exact_ux[7]  = -33; exact_uy[7]  = -11;
  exact_ux[8]  = -21; exact_uy[8]  =  -5;
  exact_ux[9]  = -13; exact_uy[9]  =  -1;
  exact_ux[10] = -11; exact_uy[10] =  10;
  exact_ux[11] =   8; exact_uy[11] =  14;

  // cycle through the pc_ images
  for(int_t i=0;i<12;++i){
    std::stringstream name;
    name << "./images/pc_0";
    if(i<10)
      name << "0" << i << ".tif";
    else
      name << i << ".tif";
    *outStream << "processing image: " << name.str() << std::endl;
    Teuchos::RCP<Image> img = Teuchos::rcp(new Image(name.str().c_str()));
    //img->write_tiff("img0.tif");
    //*outStream << "  fft of image" << std::endl;
    Teuchos::RCP<Image> img_fft = image_fft(img,true,true,100.0,true,true);
    //std::stringstream fftName;
    //fftName << "img_fft_" << i << ".tif";
    //img_fft->write_tiff(fftName.str());
    //*outStream << "  polar transform of fft" << std::endl;
    Teuchos::RCP<Image> img_pol = polar_transform(img_fft,true);
    //std::stringstream polar_name_ss;
    //polar_name_ss << "_" << i;
    //std::string polar_name = polar_name_ss.str();
    //img_pol->set_file_name(polar_name);
    //polar_name_ss << ".tif";
    //img_pol->write_tiff(polar_name_ss.str());
    //*outStream << "  phase correlating angle" << std::endl;

    // phase correlate
    scalar_t theta = 0.0;
    phase_correlate_row(img_pol,ref_pol,h_2,theta,true);
    //scalar_t theta_orig = theta;
    theta *= -1.0;
    if (theta > DICE_PI) theta = -DICE_TWOPI + theta;
    if (theta < -DICE_PI) theta = DICE_TWOPI + theta;
    const scalar_t theta_180 = (theta < 0) ? theta + DICE_PI : theta - DICE_PI;
    //*outStream << "***orig: " << theta_orig <<  " theta: " << theta*180.0/DICE_PI << " theta_180: " << theta_180*180.0/DICE_PI << std::endl;

    // transform the image by the computed rotation angle
    Teuchos::RCP<std::vector<scalar_t> > def = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
    (*def)[ROTATION_Z] = theta;
    Teuchos::RCP<std::vector<scalar_t> > def180 = Teuchos::rcp(new std::vector<scalar_t>(DICE_DEFORMATION_SIZE,0.0));
    (*def180)[ROTATION_Z] = theta_180;
    Teuchos::RCP<Image> rot_ref_0 = ref->apply_transformation(def);
    Teuchos::RCP<Image> rot_ref_180 = ref->apply_transformation(def180);
    //std::stringstream transName0;
    //transName0 << "trans0_" << i << ".tif";
    //std::stringstream transName180;
    //transName180 << "trans180_" << i << ".tif";
    //rot_ref_0->write_tiff(transName0.str());
    //rot_ref_180->write_tiff(transName180.str());

    // phase correlate the rotated images
    scalar_t u_x0 = 0.0;
    scalar_t u_y0 = 0.0;
    const scalar_t test_0 = phase_correlate_x_y(rot_ref_0,img,u_x0,u_y0);
    scalar_t u_x180 = 0.0;
    scalar_t u_y180 = 0.0;
    const scalar_t test_180 = phase_correlate_x_y(rot_ref_180,img,u_x180,u_y180);
    scalar_t final_theta = theta;
    scalar_t final_ux = u_x0;
    scalar_t final_uy = u_y0;
    if(test_180>test_0){
      final_theta = theta_180;
      final_ux = u_x180;
      final_uy = u_y180;
    }
    *outStream << "theta: " << final_theta*180.0/DICE_PI << " ux: " << final_ux << " uy: " << final_uy << std::endl;
    scalar_t angle_tol = 1.0; // 1 degree of error for the angle is accepted
    if(std::abs(final_theta*180/DICE_PI - exact_rot[i]) > angle_tol){
      *outStream << "Error, the angle for phase correlation of image " << i << " is not correct" << std::endl;
      errorFlag++;
    }
    scalar_t disp_tol = 1.0; // 1 degree of error for the angle is accepted
    if(std::abs(final_ux - exact_ux[i]) > disp_tol){
      *outStream << "Error, the x-displacement for phase correlation of image " << i << " is not correct" << std::endl;
      errorFlag++;
    }
    if(std::abs(final_uy - exact_uy[i]) > disp_tol){
      *outStream << "Error, the y-displacement for phase correlation of image " << i << " is not correct" << std::endl;
      errorFlag++;
    }
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

