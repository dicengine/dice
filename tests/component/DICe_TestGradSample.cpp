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

/*! \file  DICe_TestLargeStretch.cpp
    \brief Test of large stretch applied to a subset at initialization
*/

// note: the commented out code below was left in this test because it's useful for plotting the objective function

#include <DICe.h>
#include <DICe_Schema.h>
#include <DICe_Objective.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>
#include <cstdio>

#include <cassert>

using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  int_t iprint     = argc - 1;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);
  int_t errorFlag  = 0;

  *outStream << "--- Begin test ---" << std::endl;

  const scalar_t error_tol = 1.0;

  // read in the images
  Teuchos::RCP<Teuchos::ParameterList> img_params = Teuchos::rcp(new Teuchos::ParameterList());
  img_params->set(DICe::gauss_filter_images,true);
  img_params->set(DICe::gauss_filter_mask_size,7);
  Teuchos::RCP<DICe::Image> ref_image = Teuchos::rcp(new Image("./images/LargeStretchRef.png",img_params));
  Teuchos::RCP<DICe::Image> def_image = Teuchos::rcp(new Image("./images/LargeStretchDef.png",img_params));

  // create a schema to work with
  Teuchos::RCP<Teuchos::ParameterList> schema_params = rcp(new Teuchos::ParameterList());
  schema_params->set(DICe::compute_def_gradients,true);
  //schema_params->set(DICe::max_solver_iterations_fast,500);
  //schema_params->set(DICe::momentum_factor,0.90);
  //schema_params->set(DICe::fast_solver_tolerance,1.0E-8);
  //schema_params->set(DICe::levenberg_marquardt_regularization_factor,0.1);

  const int num_subsets = 10;
  Teuchos::ArrayRCP<scalar_t> coords_x(num_subsets,60.0);
  Teuchos::ArrayRCP<scalar_t> coords_y(num_subsets,0.0);
  for(size_t i=0;i<num_subsets;++i)
    coords_y[i] = 35 + i*5;
//  const int num_subsets = 1;
//  Teuchos::ArrayRCP<scalar_t> coords_x(num_subsets,60.0);
//  Teuchos::ArrayRCP<scalar_t> coords_y(num_subsets,80.0);

  const int_t subset_size = 25;
  Teuchos::RCP<DICe::Schema> schema = Teuchos::rcp(new DICe::Schema(coords_x,coords_y,subset_size,Teuchos::null,Teuchos::null,schema_params));
  schema->set_ref_image(ref_image);
  schema->set_def_image(def_image);

//  std::vector<int_t> neigh_i = {-1,0,1,1,1,0,-1,-1};
//  std::vector<int_t> neigh_j = {-1,-1,-1,0,1,1,1,0};
//
//  // start with initial guess for u
//  const scalar_t init_u = -30.0;
//  const scalar_t init_e = 0.0;
//
//  // set the grid spacing
//  const scalar_t step_u = 0.5; // 0.1
//  const scalar_t step_e = 0.1; // .05
//  const scalar_t window_size_u = 10.0;
//  const scalar_t window_size_e = 1.6;
//
//  const int_t num_steps_u = (int_t)(window_size_u / step_u);
//  const int_t num_steps_e = (int_t)(window_size_e / step_e);
//  const int_t num_grid = num_steps_u * num_steps_e;
//
//  std::vector<scalar_t> grads(num_grid,0.0);

//  std::ofstream output;
//  output.open ("stationary_points_.txt");

  // correlate the subsets
  for(int_t local_id=0;local_id<num_subsets;++local_id){

    Teuchos::RCP<Local_Shape_Function> sf = Teuchos::rcp(new Affine_Shape_Function(true,true,true));
    Teuchos::RCP<Objective_ZNSSD> obj = Teuchos::rcp(new Objective_ZNSSD(schema.get(),schema->subset_global_id(local_id)));
    obj->gradSampleMin2D(sf,0,3,-30.0,0.0,0.5,0.1,10.0,1.6);
    *outStream << "Subset " << local_id << " initial guess: " << (*sf)(0) << "  " << (*sf)(3) << std::endl;
    if(std::abs((*sf)(3) + 0.5)>error_tol){
      *outStream << "Error, initial guess for e fails error tol" << std::endl;
      errorFlag++;
    }
    if(std::abs((*sf)(0) + 30.0)>error_tol){
      *outStream << "Error, initial guess for u fails error tol" << std::endl;
      errorFlag++;
    }

//    //std::ofstream output_grid;
//    //output_grid.open ("grid_points.txt");
//    // evaluate the gradient on the grid
//    Teuchos::RCP<Local_Shape_Function> sf = Teuchos::rcp(new Affine_Shape_Function(true,true,true));
//    std::vector<scalar_t> params(sf->num_params(),0.0);
//    Teuchos::RCP<Objective_ZNSSD> obj = Teuchos::rcp(new Objective_ZNSSD(schema.get(),schema->subset_global_id(local_id)));
//    for(int_t i_u = 0;i_u<num_steps_u;++i_u){
//      for(int_t i_e = 0; i_e<num_steps_e;++i_e){
//        params[0] = init_u - window_size_u/2.0 + i_u*step_u;
//        params[3] = init_e - window_size_e/2.0 + i_e*step_e;
//        sf->insert(params);
//        grads[i_e*num_steps_u+i_u] = obj->gradient_norm(sf);
//        //output_grid << params[0] << " " << params[3] << " " << grads[i_e*num_steps_u+i_u] << std::endl;
//      }
//    }
//    //output_grid.close();
//
//    //std::ofstream output_mins;
//    //output_mins.open ("min_points.txt");
//    // find the local minima (not on the boundary)
//    std::vector<std::pair<size_t,scalar_t> > mins;
//    mins.reserve(num_grid/10);
//    for(int_t i_u = 1;i_u<num_steps_u-1;++i_u){
//      for(int_t i_e = 1; i_e<num_steps_e-1;++i_e){
//        bool is_min = true;
//        for(size_t neigh=0;neigh<neigh_i.size();++neigh){
//          if(grads[i_e*num_steps_u+i_u] > grads[(i_e+neigh_j[neigh])*num_steps_u+i_u+neigh_i[neigh]]){
//            is_min = false;
//            break;
//          }
//        }
//        if(is_min){
//          //output_mins << init_u - window_size_u/2.0 + i_u*step_u << " " << init_e - window_size_e/2.0 + i_e*step_e << std::endl;
//          // evaluate the objective for this min:
//          params[0] = init_u - window_size_u/2.0 + i_u*step_u;
//          params[3] = init_e - window_size_e/2.0 + i_e*step_e;
//          sf->insert(params);
//          const scalar_t gamma = obj->gamma(sf);
//          mins.push_back(std::pair<size_t,scalar_t>(i_e*num_steps_u+i_u,gamma));
//        }
//      }
//    }
//    //output_mins.close();
//
//    // find the smallest values of the mins
//    std::pair<size_t,scalar_t> min_of_mins = *std::min_element(mins.cbegin(), mins.cend(), [](const std::pair<size_t,scalar_t>& lhs, const std::pair<size_t,scalar_t>& rhs) {
//      return lhs.second < rhs.second;
//    });
//    const int_t min_index = min_of_mins.first;
//    const int_t min_j = min_index / num_steps_u;
//    const int_t min_i = min_index - min_j*num_steps_u;
//    const scalar_t min_e = init_e - window_size_e/2.0 + min_j*step_e;
//    const scalar_t min_u = init_u - window_size_u/2.0 + min_i*step_u;
//    *outStream << "Min element is at " << min_index << " i " << min_i << " j " << min_j << " " << min_of_mins.second << " " << min_u << " " << min_e << std::endl;
//    if(std::abs(min_e + 0.5)>error_tol){
//      *outStream << "Error, initial guess for u fails error tol" << std::endl;
//      errorFlag++;
//    }
//    if(std::abs(min_u + 30.0)>error_tol){
//      *outStream << "Error, initial guess for e fails error tol" << std::endl;
//      errorFlag++;
//    }

//    const scalar_t x = schema->local_field_value(local_id,DICe::field_enums::SUBSET_COORDINATES_X_FS);
//    const scalar_t y = schema->local_field_value(local_id,DICe::field_enums::SUBSET_COORDINATES_Y_FS); // FIXME unused variable if not in debug mode

//    Teuchos::RCP<Local_Shape_Function> shape_function = Teuchos::rcp(new Affine_Shape_Function(true,true,true));
//    std::vector<scalar_t> update(shape_function->num_params(),0.0);
//    update[0] = -30.0;
//    //update[3] = -0.5;
//    shape_function->update(update);
//    DEBUG_MSG("Evaluating subset " << local_id << " at (" << x << "," << y << ") initial u " << update[0]);
//    Teuchos::RCP<Objective> obj = Teuchos::rcp(new Objective_ZNSSD(schema.get(),schema->subset_global_id(local_id)));
//    int_t num_iterations = -1;
//    Status_Flag corr_status = obj->computeUpdateFast(shape_function,num_iterations);
//    //Status_Flag corr_status = obj->computeUpdateRobust(shape_function,num_iterations);
//    const scalar_t gamma = obj->gamma(shape_function);
//    DEBUG_MSG("Similarity measure gamma: " << gamma);
//    scalar_t noise_std_dev = 0.0;
//    const scalar_t sigma = obj->sigma(shape_function, noise_std_dev);

//    std::vector<scalar_t> update(num_params,0.0);
//    for(scalar_t u = -35.0; u < -25.0; u+= 0.05){
//      for(scalar_t ex = -0.8; ex < -0.2; ex+= 0.005){
//        update[0] = u;
//        update[3] = ex;
//        Teuchos::RCP<Local_Shape_Function> shape_function = Teuchos::rcp(new Affine_Shape_Function(true,true,true));
//        shape_function->update(update);
//        Teuchos::RCP<Objective_ZNSSD> obj = Teuchos::rcp(new Objective_ZNSSD(schema.get(),schema->subset_global_id(local_id)));
//        scalar_t grad_q = obj->gradient_norm(shape_function);
//        const scalar_t gamma = obj->gamma(shape_function);
//        for(int_t r=0;r<6;++r)
//          output << (*shape_function)(r) << " ";
//        output << grad_q << " " << gamma << std::endl;
//      }
//      std::cout << "u " << u << std::endl;
//    }
//    output.close();

  } // end loop over subsets

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

