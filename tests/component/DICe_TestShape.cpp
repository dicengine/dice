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

/*! \file  DICe_TestShape.cpp
    \brief Testing of shape class
*/

#include <DICe_Shape.h>
#include <DICe_Image.h>
#include <DICe.h>

#include <Teuchos_oblackholestream.hpp>

#include <iostream>
#include <cassert>

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

  scalar_t errtol  = 1.0E-6;
  scalar_t errtolSoft  = 1.0E-2;

  try {
    *outStream << "creating a polygon" << std::endl;
    int_t imgW = 200;
    int_t cx = 100;
    int_t cy = 100;
    Teuchos::ArrayRCP<scalar_t> ref_intensities(imgW*imgW,0.0);
    std::vector<int_t> shape_coords_x(4);
    std::vector<int_t> shape_coords_y(4);
    shape_coords_x[0] = 80;
    shape_coords_x[1] = 90;
    shape_coords_x[2] = 120;
    shape_coords_x[3] = 110;
    shape_coords_y[0] = 100;
    shape_coords_y[1] = 80;
    shape_coords_y[2] = 100;
    shape_coords_y[3] = 120;
    Teuchos::RCP<DICe::Polygon> poly1 = Teuchos::rcp(new DICe::Polygon(shape_coords_x,shape_coords_y));
    *outStream << "collecting the reference owned pixels" << std::endl;
    //*outStream << "    Included pixels: " << std::endl;
    std::set<std::pair<int_t,int_t> > ref_owned_pixels = poly1->get_owned_pixels();
    std::set<std::pair<int_t,int_t> >::iterator ref_set_it = ref_owned_pixels.begin();
    for(;ref_set_it!=ref_owned_pixels.end();++ref_set_it){
      //*outStream << ref_set_it->first << " " << ref_set_it->second << std::endl;
      ref_intensities[ref_set_it->first*imgW + ref_set_it->second] = 255;
    }
    *outStream << "creating the reference output image" << std::endl;
    DICe::Image ref_image(imgW,imgW,ref_intensities);
    ref_image.write_tiff("shape_ref.tif");
    *outStream << "creating a deformation map" << std::endl;
    Teuchos::RCP<Def_Map> def = Teuchos::rcp(new Def_Map());
    def->u_ = 25;
    def->v_ = -30;
    std::set<std::pair<int_t,int_t> > def_owned_pixels = poly1->get_owned_pixels(def,cx,cy);
    std::set<std::pair<int_t,int_t> >::iterator def_set_it = def_owned_pixels.begin();
    for(ref_set_it = ref_owned_pixels.begin();ref_set_it!=ref_owned_pixels.end();++ref_set_it){
      if(def_owned_pixels.find(std::pair<int_t,int_t>(ref_set_it->first+def->v_,ref_set_it->second+def->u_))==def_owned_pixels.end()){
        *outStream << "Error, owned pixels are not right for the deformed image" << std::endl;
        *outStream << "    Point was not found (ref) " << ref_set_it->second << " " << ref_set_it->first <<
            " (def) " << ref_set_it->second + def->u_ << " " << ref_set_it->first + def->v_ << std::endl;
        errorFlag++;
      }
    }
    Teuchos::ArrayRCP<scalar_t> def_intensities(imgW*imgW,0.0);
    *outStream << "the deformed shape has " << def_owned_pixels.size() << " pixels" << std::endl;
    if(def_owned_pixels.size()!=ref_owned_pixels.size()){
      *outStream << "Error, def owned pixels is not the right size" << std::endl;
      errorFlag++;
    }
    for(def_set_it = def_owned_pixels.begin();def_set_it!=def_owned_pixels.end();++def_set_it){
      //*outStream << "DEF: " << def_set_it->first << " " << def_set_it->second << std::endl;
      def_intensities[def_set_it->first*imgW + def_set_it->second] = 255;
    }
    *outStream << "creating deformed output image" << std::endl;
    DICe::Image def_image(imgW,imgW,def_intensities);
    def_image.write_tiff("shape_def.tif");

    *outStream << "testing deformed shape with larger skin" << std::endl;
    const scalar_t large_skin_factor = 1.5;
    std::set<std::pair<int_t,int_t> > large_skin_owned_pixels = poly1->get_owned_pixels(def,cx,cy,large_skin_factor);
    Teuchos::ArrayRCP<scalar_t> large_skin_intensities(imgW*imgW,0.0);
    *outStream << "large skin shape has " << large_skin_owned_pixels.size() << " pixels" << std::endl;
    if(large_skin_owned_pixels.size() <= ref_owned_pixels.size()){
      *outStream << "Error, large skin has too few pixels" << std::endl;
      errorFlag++;
    }
    std::set<std::pair<int_t,int_t> >::iterator large_skin_set_it = large_skin_owned_pixels.begin();
    for(;large_skin_set_it!=large_skin_owned_pixels.end();++large_skin_set_it){
      large_skin_intensities[large_skin_set_it->first*imgW + large_skin_set_it->second] = 255;
    }
    for(int_t i=0;i<shape_coords_x.size();++i){
      if(large_skin_owned_pixels.find(std::pair<int_t,int_t>((int_t)((shape_coords_y[i]-cy)*large_skin_factor*0.9 + cy) + def->v_,
        (int_t)((shape_coords_x[i]-cx)*large_skin_factor*0.9 + cx) + def->u_))==large_skin_owned_pixels.end()){
        *outStream << "Error, large skin owned pixels are not right" << std::endl;
        *outStream << "    Point was not found (ref) " << shape_coords_x[i] << " " << shape_coords_y[i] <<
            " (def) " << (int_t)((shape_coords_x[i]-cx)*large_skin_factor*0.9 + cx) + def->u_ <<
            " " << (int_t)((shape_coords_y[i]-cy)*large_skin_factor*0.9 + cy) + def->v_ << std::endl;
        errorFlag++;
      }
    }
    DICe::Image large_skin_image(imgW,imgW,large_skin_intensities);
    large_skin_image.write_tiff("shape_large_skin.tif");

    *outStream << "testing deformed shape with smaller skin" << std::endl;
    const scalar_t small_skin_factor = 0.75;
    std::set<std::pair<int_t,int_t> > small_skin_owned_pixels = poly1->get_owned_pixels(def,cx,cy,small_skin_factor);
    Teuchos::ArrayRCP<scalar_t> small_skin_intensities(imgW*imgW,0.0);
    *outStream << "the small skin shape has " << small_skin_owned_pixels.size() << " pixels" << std::endl;
    if(small_skin_owned_pixels.size() >= ref_owned_pixels.size()){
      *outStream << "Error, the small skin has too many pixels" << std::endl;
      errorFlag++;
    }
    std::set<std::pair<int_t,int_t> >::iterator small_skin_set_it = small_skin_owned_pixels.begin();
    for(;small_skin_set_it!=small_skin_owned_pixels.end();++small_skin_set_it){
      small_skin_intensities[small_skin_set_it->first*imgW + small_skin_set_it->second] = 255;
    }
    DICe::Image small_skin_image(imgW,imgW,small_skin_intensities);
    small_skin_image.write_tiff("shape_small_skin.tif");

  }
  catch (std::logic_error err) {
    *outStream << err.what() << "\n";
    errorFlag = -1000;
  }; // end try

  *outStream << "--- End test ---" << std::endl;


  // finalize kokkos
  Kokkos::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

