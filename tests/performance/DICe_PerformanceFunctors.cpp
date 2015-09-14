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
#include <DICe_Kokkos.h>
#include <DICe_Tiff.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <boost/timer/timer.hpp>

#include <iostream>

using namespace DICe;
using namespace boost::timer;

int main(int argc, char *argv[]) {

  // initialize kokkos
  Kokkos::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  size_t iprint     = argc - 1;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin performance test ---" << std::endl;

  // create a vector of image sizes to use
  const size_t num_img_sizes = 5;
  size_t width = 0;
  size_t height = 0;
  std::vector<size_t> widths(num_img_sizes,0);
  std::vector<size_t> heights(num_img_sizes,0);
  const std::string file_name = "./images/UberImage.tif";

  // read the image dimensions from file:
  read_tiff_image_dimensions(file_name,width,height);
  *outStream << "master image dims:     " << width << " x " << height << std::endl;
  *outStream << "number of image sizes: " << num_img_sizes << std::endl;
  const size_t w_step = width/num_img_sizes;
  const size_t h_step = height/num_img_sizes;
  for(size_t i=0;i<num_img_sizes;++i){
    widths[i] = (i+1)*w_step;
    heights[i] = (i+1)*h_step;
  }

  // set up the parameters
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::compute_image_gradients,true);

  // image size loop:
  for(size_t size_it=0;size_it<num_img_sizes;++size_it){
    const size_t w_it = widths[size_it];
    const size_t h_it = heights[size_it];
    *outStream << "image size: " << w_it << " x " << h_it << std::endl;

    *outStream << "reading the image" << std::endl;
    {
      cpu_timer read_timer;
      Image img(file_name,0,0,w_it,h_it,params);
      *outStream << read_timer.format() << std::endl;
    }

    // TODO image gradient


  } // end size loop






  // TODO image filter/convolution

  // TODO subset mean value

  // TODO subset initialization(mapping + interpolation)

  // TODO compute correlation value

  *outStream << "--- End performance test ---" << std::endl;

  // finalize kokkos
  Kokkos::finalize();

  return 0;
}

