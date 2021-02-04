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

/*! \file  DICe_TestObjective.cpp
    \brief Testing of objective class construction and gamma methods
    actual correlation methods are tested in a separate test, not here
*/

#include <DICe_Schema.h>
#include <DICe_Objective.h>
#include <DICe.h>

#include <Teuchos_oblackholestream.hpp>

#include <iostream>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  int_t errorFlag  = 0;
  work_t errtol  = 5.0E-2;

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "creating stripes images " << std::endl;
  // create an image with black and white stripes:
  const int_t img_width = 199;
  const int_t img_height = 199;
  const int_t num_stripes = 10;
  const int_t stripe_width = 20;//img_width/num_stripes;
  Teuchos::ArrayRCP<storage_t> intensities(img_width*img_height,0);
  for(int_t y=0;y<img_height;++y){
    for(int_t stripe=0;stripe<=num_stripes/2;++stripe){
      for(int_t x=stripe*(2*stripe_width);x<stripe*(2*stripe_width)+stripe_width;++x){
        if(x>=img_width)continue;
        intensities[y*img_width+x] = 255;
      }
    }
  }
  *outStream << "reference striped image created successfuly\n";

  // test moving the centroid of a quadratic shape function:
  Teuchos::RCP<Local_Shape_Function> move_shape_func = Teuchos::rcp(new Quadratic_Shape_Function());
  assert(move_shape_func->parameters()->size()==12);
  //(*move_shape_func)(0)  = 1.0; // A (should already by 1.0 from constructor)
  (*move_shape_func)(1)  = 0.0002; // B
  (*move_shape_func)(2)  = 0.0003; // C
  (*move_shape_func)(3)  = 0.0004; // D
  (*move_shape_func)(4)  = 0.0005; // E
  (*move_shape_func)(5)  = 1.352; // F
  (*move_shape_func)(6)  = 0.0006; // G
  //(*move_shape_func)(7)  = 1.0; // H (should already by 1.0 from constructor)
  (*move_shape_func)(8)  = 0.0007; // I
  (*move_shape_func)(9)  = 0.0008; // J
  (*move_shape_func)(10) = 0.0009; // K
  (*move_shape_func)(11) = -0.897; // L
  const work_t cx_orig = 345.98;
  const work_t cy_orig = 12.39;
  const work_t pt_x = 45.289;
  const work_t pt_y = -20.94;
  work_t x_prime = 0.0,y_prime=0.0;
  move_shape_func->map(pt_x,pt_y,cx_orig,cy_orig,x_prime,y_prime);
  *outStream << " x_prime " << x_prime << " y_prime " << y_prime << std::endl;
  const work_t cx_new = 56.98;
  const work_t cy_new = -899.0;
  const work_t delta_x = cx_new - cx_orig;
  const work_t delta_y = cy_new - cy_orig;
  move_shape_func->update_params_for_centroid_change(delta_x,delta_y);
  work_t x_prime_new = 0.0,y_prime_new=0.0;
  move_shape_func->map(pt_x,pt_y,cx_new,cy_new,x_prime_new,y_prime_new);
  *outStream << " x_prime new " << x_prime_new << " y_prime new " << y_prime_new << std::endl;
  if(std::abs(x_prime_new-x_prime) > 1.0E-4||std::abs(y_prime_new-y_prime) > 1.0E-4){
    *outStream << "Error, change of centroid for shape function not correct" << std::endl;
    errorFlag++;
  }

  // testing objective gamma:

  // dummy shape function to pass deformation map parameters to objective
  Teuchos::RCP<Local_Shape_Function> shape_function = shape_function_factory();
  Teuchos::RCP<Local_Shape_Function> quadratic_shape_function = Teuchos::rcp(new Quadratic_Shape_Function());

  *outStream << "testing ZNSSD correlation" << std::endl;

  // track the correlation gamma for various pixel shifts:
  // no shift should result in gamma = 0.0 and when the images are opposite gamma should = 4.0
  for(int_t shift=0;shift<11;shift++){
    *outStream << "processing shift: " << shift*2 << "\n";
    Teuchos::ArrayRCP<storage_t> intensitiesShift(img_width*img_height,0);
    for(int_t y=0;y<img_height;++y){
      for(int_t stripe=0;stripe<=num_stripes/2;++stripe){
        for(int_t x=stripe*(2*stripe_width) - shift*2;x<stripe*(2*stripe_width)+stripe_width - shift*2;++x){
          if(x>=img_width)continue;
          if(x<0)continue;
          intensitiesShift[y*img_width+x] = 255;
        }
      }
    }
    // create a temp schema:
    Teuchos::ArrayRCP<work_t> coords_x(1,100);
    Teuchos::ArrayRCP<work_t> coords_y(1,100);
    DICe::Schema * schema = new DICe::Schema(coords_x,coords_y,99);
    schema->set_ref_image(img_width,img_height,intensities);
    schema->set_def_image(img_width,img_height,intensitiesShift);
    //schema->sync_fields_all_to_dist(); // distribute the fields across processors if necessary
    // create an objective:
    Teuchos::RCP<DICe::Objective_ZNSSD> obj = Teuchos::rcp(new DICe::Objective_ZNSSD(schema,0));
    // evaluate the correlation value:
    const work_t gamma =  obj->gamma(shape_function);
    *outStream << "gamma value: " << gamma << std::endl;
    if(std::abs(gamma - shift*0.4)>errtol){
      *outStream << "Error, gamma is not " << shift*0.4 << " value=" << gamma << "\n";
      errorFlag++;
    }
    Teuchos::RCP<DICe::Objective_ZNSSD> quadratic_obj = Teuchos::rcp(new DICe::Objective_ZNSSD(schema,0));
    const work_t quadratic_gamma =  quadratic_obj->gamma(quadratic_shape_function);
    *outStream << "quadratic gamma value: " << quadratic_gamma << std::endl;
    if(std::abs(quadratic_gamma - shift*0.4)>errtol){
      *outStream << "Error, gamma (for quadratic shape function) is not " << shift*0.4 << " value=" << quadratic_gamma << "\n";
      errorFlag++;
    }
    delete schema;
  }

  // dummy deformation vector to pass to objective
  *outStream << "testing quadratic deformation with ZNSSD correlation" << std::endl;
  Teuchos::RCP<Local_Shape_Function> quad_shape_func_exact = Teuchos::rcp(new Quadratic_Shape_Function());
  Teuchos::RCP<Local_Shape_Function> quad_shape_func = Teuchos::rcp(new Quadratic_Shape_Function());
  assert(quad_shape_func_exact->parameters()->size()==12);
  //(*quad_shape_func_exact)(0)  = 1.0; // A (should already by 1.0 from constructor)
  (*quad_shape_func_exact)(1)  = 0.0002; // B
  (*quad_shape_func_exact)(2)  = 0.0003; // C
  (*quad_shape_func_exact)(3)  = 0.0004; // D
  (*quad_shape_func_exact)(4)  = 0.0005; // E
  (*quad_shape_func_exact)(5)  = 1.352; // F
  (*quad_shape_func_exact)(6)  = 0.0006; // G
  //(*quad_shape_func_exact)(7)  = 1.0; // H (should already by 1.0 from constructor)
  (*quad_shape_func_exact)(8)  = 0.0007; // I
  (*quad_shape_func_exact)(9)  = 0.0008; // J
  (*quad_shape_func_exact)(10) = 0.0009; // K
  (*quad_shape_func_exact)(11) = -0.897; // L

  // initialize the solution to be close to the exact sol
  (*quad_shape_func)(1)  = 0.0002; // B
  (*quad_shape_func)(2)  = 0.0003; // C
  (*quad_shape_func)(3)  = 0.0004; // D
  (*quad_shape_func)(4)  = 0.0005; // E
  (*quad_shape_func)(5)  = 1.0; // F
  (*quad_shape_func)(6)  = 0.0006; // G
  (*quad_shape_func)(8)  = 0.0007; // I
  (*quad_shape_func)(9)  = 0.0008; // J
  (*quad_shape_func)(10) = 0.0009; // K
  (*quad_shape_func)(11) = -1.0; // L

  // test the mapping function of the quadratic shape function
  const work_t test_x=105, test_y=27;
  const work_t cx = 250.0;
  const work_t cy = 250.0;
  work_t test_map_x=0.0, test_map_y=0.0;
  quad_shape_func_exact->map(test_x,test_y,cx,cy,test_map_x,test_map_y);
  const work_t exact_map_x = 149.282;
  const work_t exact_map_y = 110.227;
  if(std::abs(exact_map_x - test_map_x) > 1.0E-3 || std::abs(exact_map_y - test_map_y) > 1.0E-3){
    *outStream << "Error, incorrect mapped location for quadratic shape function" << std::endl;
    errorFlag++;
  }

  Teuchos::RCP<DICe::Image> affineRef = Teuchos::rcp(new DICe::Image("./images/refSpeckled.tif"));
  const int_t affine_w = affineRef->width();
  const int_t affine_h = affineRef->height();
  Teuchos::ArrayRCP<work_t> intensitiesMod(affine_w*affine_h,0.0);
  work_t mapped_x=0.0,mapped_y=0.0;
  for(int_t y=0;y<affine_h;++y){
    for(int_t x=0;x<affine_w;++x){
      quad_shape_func_exact->map(x,y,cx,cy,mapped_x,mapped_y);
      if(mapped_x>4.0&&mapped_x<affine_w-4.0&&mapped_y>4.0&&mapped_y<affine_h-4.0){
        intensitiesMod[y*affine_w+x] = affineRef->interpolate_keys_fourth(mapped_x,mapped_y);
      }
    } // end x pixel
  } // end y pixel
  Teuchos::RCP<DICe::Scalar_Image> affineDef = Teuchos::rcp(new DICe::Scalar_Image(affine_w,affine_h,intensitiesMod));
  affineRef->write("affineDefImage.tiff");
  affineDef->write("affineRefImage.tiff");
  Teuchos::ArrayRCP<work_t> coords_x(1,250);
  Teuchos::ArrayRCP<work_t> coords_y(1,250);
  DICe::Schema * schema = new DICe::Schema(coords_x,coords_y,99);
  schema->set_ref_image(affineDef); // these are switched on purpose
  schema->set_def_image(affineRef);
  //schema->sync_fields_all_to_dist(); // distribute the fields across processors if necessary
  // create an objective:
  Teuchos::RCP<DICe::Objective_ZNSSD> obj = Teuchos::rcp(new DICe::Objective_ZNSSD(schema,0));

  int_t num_iterations = 0;
  obj->computeUpdateFast(quad_shape_func,num_iterations);
  // check the values of def guess and gamma
  const work_t gamma =  obj->gamma(quad_shape_func);
  *outStream << "gradient-based gamma value: " << gamma << std::endl;
  if(std::abs(gamma)>1.0E-4){
    *outStream << "Error, gamma is too large\n";
    errorFlag++;
  }
  bool grad_converged = quad_shape_func->test_for_convergence(*quad_shape_func_exact->parameters(),1.0E-3);
  if(!grad_converged){
    *outStream << "Error, gradient optimized solution is not correct" << std::endl;
    *outStream << "Quad shape function parameter values" << std::endl;
    quad_shape_func->print_parameters();
    *outStream << "Exact values" << std::endl;
    quad_shape_func_exact->print_parameters();
    errorFlag++;
  }

  // initialize the solution to be close to the exact sol
  quad_shape_func->clear();
  (*quad_shape_func)(1)  = 0.0002; // B
  (*quad_shape_func)(2)  = 0.0003; // C
  (*quad_shape_func)(3)  = 0.0004; // D
  (*quad_shape_func)(4)  = 0.0005; // E
  (*quad_shape_func)(5)  = 1.0; // F
  (*quad_shape_func)(6)  = 0.0006; // G
  (*quad_shape_func)(8)  = 0.0007; // I
  (*quad_shape_func)(9)  = 0.0008; // J
  (*quad_shape_func)(10) = 0.0009; // K
  (*quad_shape_func)(11) = -1.0; // L

  num_iterations = 0;
  obj->computeUpdateRobust(quad_shape_func,num_iterations);
  bool simplex_converged = quad_shape_func->test_for_convergence(*quad_shape_func_exact->parameters(),1.0E-3);
  if(!simplex_converged){
    *outStream << "Error, simplex optimized solution is not correct" << std::endl;
    *outStream << "Quad shape function parameter values" << std::endl;
    quad_shape_func->print_parameters();
    *outStream << "Exact values" << std::endl;
    quad_shape_func_exact->print_parameters();
    errorFlag++;
  }
  const work_t gamma_simplex =  obj->gamma(quad_shape_func);
  *outStream << "simplex gamma value: " << gamma_simplex << std::endl;
  if(std::abs(gamma_simplex)>1.0E-4){
    *outStream << "Error, gamma is too large\n";
    errorFlag++;
  }


  delete schema;

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

