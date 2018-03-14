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
#include <DICe_Triangulation.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <cassert>

using namespace std;
using namespace DICe;

/// Initializes the cross correlation between two images from the left and right camera, respectively

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&bhs, false);

  // read the parameters from the input file:
  if(argc!=4){ // executable and input file
    std::cout << "Usage: DICe_CrossInit <left_image> <right_image> <1=use_nonlinear_with_points_file>" << std::endl;
    exit(1);
  }
  std::string left_name = argv[1];
  DEBUG_MSG("left image: " << left_name);
  std::string right_name = argv[2];
  DEBUG_MSG("right image: " << right_name);
  const int_t use_nonlinear = std::strtol(argv[3],NULL,0);
  const bool use_nonlin = use_nonlinear==1;
  DEBUG_MSG("CrossInit(): use_nonlinear is " << use_nonlin);

  Teuchos::RCP<Image> left_img = Teuchos::rcp(new Image(left_name.c_str()));
  Teuchos::RCP<Image> right_img = Teuchos::rcp(new Image(right_name.c_str()));

  // create a triangulation
  Teuchos::RCP<Triangulation> triang = Teuchos::rcp(new Triangulation());
  const int_t ret_val = triang->estimate_projective_transform(left_img,right_img,true,use_nonlin);
  DEBUG_MSG("CrossInit(): estimate_projective_transform() return value " << ret_val);


  DICe::finalize();

  return ret_val;

}

