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
#include <DICe_Subset.h>
#include <DICe_LocalShapeFunction.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  scalar_t errorTol = 1.0E-3;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  // create a subset

  Teuchos::RCP<Image> image = Teuchos::rcp(new Image("./images/shapeFuncTestAngle.tif"));
  const int_t cx = 249;
  const int_t cy = 204;
  const int_t subset_size = 301;
  Teuchos::RCP<Subset> subset = Teuchos::rcp(new Subset(cx,cy,subset_size,subset_size));
  subset->initialize(image);
  subset->write_tiff("quadTestSubset.tif");

  // create a quadratic shape function
  Teuchos::RCP<Local_Shape_Function> shape_func = Teuchos::rcp(new Quadratic_Shape_Function());
  subset->initialize(image,REF_INTENSITIES,shape_func);
  subset->write_tiff("quadTestSubsetNoMap.tif");
  Teuchos::RCP<Image> quad_img = Teuchos::rcp(new Image("./quadTestSubsetNoMap.tif"));
  Teuchos::RCP<Image> quad_img_gold = Teuchos::rcp(new Image("./images/quadTestSubsetNoMap.tif"));
  scalar_t no_map_diff = quad_img->diff(quad_img_gold);
  *outStream << "no map diff: " << no_map_diff << std::endl;
  if(no_map_diff > errorTol){
    *outStream << "Error, no mapping subset incorrect" << std::endl;
    errorFlag++;
  }

  const scalar_t u = 63.0;
  const scalar_t v = -63.0;
  shape_func->insert_motion(u,v);
  subset->initialize(image,REF_INTENSITIES,shape_func);
  subset->write_tiff("quadTestSubsetTranslate.tif");
  Teuchos::RCP<Image> quad_trans_img = Teuchos::rcp(new Image("./quadTestSubsetTranslate.tif"));
  Teuchos::RCP<Image> quad_trans_img_gold = Teuchos::rcp(new Image("./images/quadTestSubsetTranslate.tif"));
  scalar_t trans_diff = quad_trans_img->diff(quad_trans_img_gold);
  *outStream << "translate diff: " << trans_diff << std::endl;
  if(trans_diff > errorTol){
    *outStream << "Error, translation incorrect" << std::endl;
    errorFlag++;
  }

  scalar_t rot = 0.785398;
  shape_func->insert_motion(0.0,0.0,rot);
  subset->initialize(image,REF_INTENSITIES,shape_func);
  subset->write_tiff("quadTestSubsetRot.tif");
  Teuchos::RCP<Image> quad_rot_img = Teuchos::rcp(new Image("./quadTestSubsetRot.tif"));
  Teuchos::RCP<Image> quad_rot_img_gold = Teuchos::rcp(new Image("./images/quadTestSubsetRot.tif"));
  scalar_t rot_diff = quad_rot_img->diff(quad_rot_img_gold);
  *outStream << "rotation diff: " << rot_diff << std::endl;
  if(rot_diff > errorTol){
    *outStream << "Error, rotation incorrect" << std::endl;
    errorFlag++;
  }

  // test the map to u,v,theta
  shape_func->insert_motion(u,v,rot);
  scalar_t out_u=0.0,out_v=0.0,out_t=0.0;
  shape_func->map_to_u_v_theta(cx,cy,out_u,out_v,out_t);
  if(std::abs(out_u - u)>errorTol || std::abs(out_v - v)>errorTol || std::abs(out_t - rot)>errorTol){
    *outStream << "Error, map_to_u_v_theta failed" << std::endl;
    errorFlag++;
  }

  // cycle through several rotations:
  rot = 0.0;
  while(rot < 6.3){
    shape_func->insert_motion(0.0,0.0,rot);
    shape_func->map_to_u_v_theta(cx,cy,out_u,out_v,out_t);
    *outStream << "testing rotation input " << rot << " output " << out_t << std::endl;
    if(std::abs(out_u)>errorTol || std::abs(out_v)>errorTol || std::abs(out_t - rot)>errorTol){
      *outStream << "Error, map_to_u_v_theta failed" << std::endl;
      errorFlag++;
    }
    rot += 0.785398;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

