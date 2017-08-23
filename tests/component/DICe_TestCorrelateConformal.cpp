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

/*! \file  DICe_TestCorrelateParamsConformal.cpp
    \brief Test of various parameters that can be set for a correlation.
    We try to hit all combinations below for a mutli_shape (conformal) subset
*/

#include <DICe.h>
#include <DICe_Schema.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>
#include <iomanip>
#include <cstdio>

#include <cassert>

using namespace DICe;
using namespace DICe::mesh::field_enums;

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
  scalar_t errtol  = 5.0E-2;
  scalar_t errtolSoft  = 1.0;

  *outStream << "--- Begin test ---" << std::endl;

  // initialization

  std::string fileRef("./images/TestSubsetConstructionRef.tif");
  std::string fileDef("./images/TestSubsetConstructionNormalRotDisp.tif");

  const scalar_t ex_exact = 0.121;
  const scalar_t ey_exact = 0.121;
  const scalar_t t_exact = 0.262;
  const scalar_t u_exact = 9.8;
  const scalar_t v_exact = -7.62;

  *outStream << "testing conformal subset param combinations" << std::endl;

  // create the conformal subset, four sided shape at odd angles
  std::vector<int_t> shape_coords_x(4);
  std::vector<int_t> shape_coords_y(4);
  shape_coords_x[0] = 25;
  shape_coords_x[1] = 35;
  shape_coords_x[2] = 39;
  shape_coords_x[3] = 20;
  shape_coords_y[0] = 22;
  shape_coords_y[1] = 20;
  shape_coords_y[2] = 40;
  shape_coords_y[3] = 35;
  Teuchos::RCP<DICe::Polygon> poly1 = Teuchos::rcp(new DICe::Polygon(shape_coords_x,shape_coords_y));

  DICe::multi_shape multiShape;
  multiShape.push_back(poly1);
  DICe::Conformal_Area_Def subset_def(multiShape);
  // there's only one subset defined in this example, but it still needs to get passed as a map
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > subset_defs = Teuchos::rcp(new std::map<int_t,DICe::Conformal_Area_Def>);
  subset_defs->insert(std::pair<int_t,DICe::Conformal_Area_Def>(0,subset_def));

  // solution parameter
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  // enable all the shape functions (even if not needed)
  params->set(DICe::enable_translation,true);
  params->set(DICe::enable_rotation,true);
  params->set(DICe::enable_normal_strain,true);
  params->set(DICe::enable_shear_strain,true);
  params->set(DICe::robust_solver_tolerance,1.0E-4);
  Teuchos::ArrayRCP<scalar_t> coords_x(1,30);
  Teuchos::ArrayRCP<scalar_t> coords_y(1,30);
  Teuchos::RCP<DICe::Schema> schema = Teuchos::rcp(new DICe::Schema(coords_x,coords_y,-1,subset_defs,Teuchos::null,params));
  schema->set_ref_image(fileRef);
  schema->set_def_image(fileDef);

  // create the column titles
  std::vector<std::string> colTitles;
  colTitles.push_back("Status");
  colTitles.push_back("Rout");
  colTitles.push_back("Interp");
  colTitles.push_back("Init");
  colTitles.push_back("Opt");
  colTitles.push_back("Ux");
  colTitles.push_back("Uy");
  colTitles.push_back("Rz");
  colTitles.push_back("Nx");
  colTitles.push_back("Ny");
  colTitles.push_back("Sxy");
  colTitles.push_back("Sigma");
  colTitles.push_back("Gamma");
  colTitles.push_back("Flag");
  for(size_t title=0;title<colTitles.size();++title){
    if((title>10&&title<14)||title>14)
      *outStream << std::setw(12) << colTitles[title];
    else
      *outStream << std::setw(7) << colTitles[title];
  }
  *outStream << std::endl;

  std::vector<DICe::Optimization_Method> opt_methods;
  opt_methods.push_back(DICe::SIMPLEX);
  opt_methods.push_back(DICe::SIMPLEX_THEN_GRADIENT_BASED);
  opt_methods.push_back(DICe::GRADIENT_BASED);
  opt_methods.push_back(DICe::GRADIENT_BASED_THEN_SIMPLEX);

  std::vector<DICe::Interpolation_Method> interp_methods;
  interp_methods.push_back(DICe::BILINEAR);
  interp_methods.push_back(DICe::KEYS_FOURTH);

  // BEGIN PARAMS LOOP
  for(size_t opt_method=0;opt_method<opt_methods.size();++opt_method){
    for(size_t interp_method=0;interp_method<interp_methods.size();++interp_method){
      // set the parameters
      // change the default values
      params->set(DICe::correlation_routine,DICe::GENERIC_ROUTINE);
      params->set(DICe::interpolation_method,interp_methods[interp_method]);
      params->set(DICe::initialization_method,DICe::USE_FIELD_VALUES);
      params->set(DICe::optimization_method,opt_methods[opt_method]);
      schema->set_params(params);

      // reset the initial guess
      schema->local_field_value(0,SUBSET_COORDINATES_X_FS) = 30;
      schema->local_field_value(0,SUBSET_COORDINATES_Y_FS) = 30;
      schema->local_field_value(0,SIGMA_FS) = 0.0;
      schema->local_field_value(0,GAMMA_FS) = 0.0;
      schema->local_field_value(0,STATUS_FLAG_FS) = 0;
      schema->local_field_value(0,SUBSET_DISPLACEMENT_X_FS) = u_exact - 0.25;
      schema->local_field_value(0,SUBSET_DISPLACEMENT_Y_FS) = v_exact - 0.25;
      schema->local_field_value(0,ROTATION_Z_FS)     = t_exact     - 0.05;
      schema->local_field_value(0,NORMAL_STRETCH_XX_FS) = 0.0;
      schema->local_field_value(0,NORMAL_STRETCH_YY_FS) = 0.0;
      schema->local_field_value(0,SHEAR_STRETCH_XY_FS) = 0.0;

      schema->execute_correlation();

      // check the solution to ensure the error is small:
      bool valueError = false;
      if(std::abs(schema->local_field_value(0,SUBSET_DISPLACEMENT_X_FS) - u_exact) > errtol) valueError = true;
      if(std::abs(schema->local_field_value(0,SUBSET_DISPLACEMENT_Y_FS) - v_exact) > errtol) valueError = true;
      if(std::abs(schema->local_field_value(0,ROTATION_Z_FS) - t_exact) > errtol) valueError = true;
      if(std::abs(schema->local_field_value(0,NORMAL_STRETCH_XX_FS) - ex_exact) > errtol) valueError = true;
      if(std::abs(schema->local_field_value(0,NORMAL_STRETCH_YY_FS) - ey_exact) > errtol) valueError = true;
      if(std::abs(schema->local_field_value(0,SHEAR_STRETCH_XY_FS)) > errtol) valueError = true;
      if(std::abs(schema->local_field_value(0,STATUS_FLAG_FS) - DICe::INITIALIZE_USING_PREVIOUS_FRAME_SUCCESSFUL) > errtol) valueError = true;
      if(std::abs(schema->local_field_value(0,SIGMA_FS) + 1) < errtol) valueError = true; // check that sigma != -1

      std::string statusStr = valueError ? "FAIL": "PASS";
      if(valueError) errorFlag++;

      // print the results in easy to read table format
      std::ios::fmtflags f( outStream->flags() ); // get the state of the cout flags
      *outStream << std::setw(7)  << statusStr;
      *outStream << std::setw(7)  << schema->correlation_routine();
      *outStream << std::setw(7)  << schema->interpolation_method();
      *outStream << std::setw(7)  << schema->initialization_method();
      *outStream << std::setw(7)  << schema->optimization_method();
      *outStream << std::setw(7)  << std::setprecision(3) << schema->local_field_value(0,SUBSET_DISPLACEMENT_X_FS);
      *outStream << std::setw(7)  << std::setprecision(3) << schema->local_field_value(0,SUBSET_DISPLACEMENT_Y_FS);
      *outStream << std::setw(7)  << std::setprecision(3) << schema->local_field_value(0,ROTATION_Z_FS);
      *outStream << std::setw(7)  << std::setprecision(3) << schema->local_field_value(0,NORMAL_STRETCH_XX_FS);
      *outStream << std::setw(7)  << std::setprecision(3) << schema->local_field_value(0,NORMAL_STRETCH_YY_FS);
      *outStream << std::setw(12) << std::setprecision(2) << schema->local_field_value(0,SHEAR_STRETCH_XY_FS);
      *outStream << std::setw(12) << std::setprecision(3) << std::setiosflags(std::ios::scientific) <<  schema->local_field_value(0,SIGMA_FS);
      *outStream << std::setw(12) <<  schema->local_field_value(0,GAMMA_FS);
      *outStream << std::setw(7)  << std::setiosflags(std::ios_base::fixed) << schema->local_field_value(0,STATUS_FLAG_FS);
      *outStream << std::endl;
      outStream->flags(f); // reset the cout flags to the original state
    }
  }
  // END PARAMS LOOP

  *outStream << "testing multiple conformal subsets" << std::endl;

  // test mutliple conformal subsets
  // create the conformal subset, four sided shape at odd angles
  shape_coords_x[0] = 27;
  shape_coords_x[1] = 37;
  shape_coords_x[2] = 41;
  shape_coords_x[3] = 22;
  shape_coords_y[0] = 22;
  shape_coords_y[1] = 20;
  shape_coords_y[2] = 40;
  shape_coords_y[3] = 35;
  Teuchos::RCP<DICe::Polygon> poly2 = Teuchos::rcp(new DICe::Polygon(shape_coords_x,shape_coords_y));

  DICe::multi_shape multiShape2;
  multiShape2.push_back(poly1);
  DICe::Conformal_Area_Def subset_def2(multiShape2);
  // add this subset to the argument vector
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> >  subset_defs2 = Teuchos::rcp(new std::map<int_t,DICe::Conformal_Area_Def>);
  subset_defs2->insert(std::pair<int_t,DICe::Conformal_Area_Def>(0,subset_def));
  subset_defs2->insert(std::pair<int_t,DICe::Conformal_Area_Def>(1,subset_def2));

  Teuchos::ArrayRCP<scalar_t> coords_x2(2,0.0);
  Teuchos::ArrayRCP<scalar_t> coords_y2(2,0.0);
  coords_x2[0] = 30; coords_y2[0] = 30;
  coords_x2[1] = 32; coords_y2[1] = 32;
  Teuchos::RCP<DICe::Schema> schema2 = Teuchos::rcp(new DICe::Schema(coords_x2,coords_y2,-1,subset_defs2,Teuchos::null,params));
  schema2->set_ref_image(fileRef);
  schema2->set_def_image(fileDef);

  // centroid 1
  schema2->local_field_value(0,SUBSET_COORDINATES_X_FS) = 30;
  schema2->local_field_value(0,SUBSET_COORDINATES_Y_FS) = 30;
  // centroid 2
  schema2->local_field_value(1,SUBSET_COORDINATES_X_FS) = 32;
  schema2->local_field_value(1,SUBSET_COORDINATES_Y_FS) = 32;

  for(int_t i=0;i<2;++i){
    // reset the initial guess
    schema2->local_field_value(i,SIGMA_FS) = 0.0;
    schema2->local_field_value(i,GAMMA_FS) = 0.0;
    schema2->local_field_value(i,STATUS_FLAG_FS) = 0;
    schema2->local_field_value(i,SUBSET_DISPLACEMENT_X_FS) = u_exact - 0.25;
    schema2->local_field_value(i,SUBSET_DISPLACEMENT_Y_FS) = v_exact - 0.25;
    schema2->local_field_value(i,ROTATION_Z_FS)            = t_exact - 0.05;
    schema2->local_field_value(i,NORMAL_STRETCH_XX_FS) = 0.0;
    schema2->local_field_value(i,NORMAL_STRETCH_YY_FS) = 0.0;
    schema2->local_field_value(i,SHEAR_STRETCH_XY_FS) = 0.0;
  }

  schema2->execute_correlation();

  // check the solution to ensure the error is small:
  bool valueError = false;
  for(int_t i=0;i<2;++i){
    *outStream << " x[" << i << "] " << schema2->local_field_value(i,SUBSET_COORDINATES_X_FS);
    *outStream << " y[" << i << "] " << schema2->local_field_value(i,SUBSET_COORDINATES_Y_FS);
    *outStream << " u[" << i << "] " << schema2->local_field_value(i,SUBSET_DISPLACEMENT_X_FS);
    *outStream << " v[" << i << "] " << schema2->local_field_value(i,SUBSET_DISPLACEMENT_Y_FS);
    *outStream << " Rz[" << i << "] " << schema2->local_field_value(i,ROTATION_Z_FS);
    *outStream << " Nx[" << i << "] " << schema2->local_field_value(i,NORMAL_STRETCH_XX_FS);
    *outStream << " Ny[" << i << "] " << schema2->local_field_value(i,NORMAL_STRETCH_YY_FS);
    *outStream << " Gxy[" << i << "] " << schema2->local_field_value(i,SHEAR_STRETCH_XY_FS);
    *outStream << " Sigma[" << i << "] " <<  schema2->local_field_value(i,SIGMA_FS);
    *outStream << " Gamma[" << i << "] " <<  schema2->local_field_value(i,GAMMA_FS);
    *outStream << " Status[" << i << "] " << schema2->local_field_value(i,STATUS_FLAG_FS) << std::endl;

    if(std::abs(schema2->local_field_value(i,SUBSET_DISPLACEMENT_X_FS) - u_exact) > errtolSoft) valueError = true;
    if(std::abs(schema2->local_field_value(i,SUBSET_DISPLACEMENT_Y_FS) - v_exact) > errtolSoft) valueError = true;
    if(std::abs(schema2->local_field_value(i,ROTATION_Z_FS) - t_exact) > errtolSoft) valueError = true;
    if(std::abs(schema2->local_field_value(i,NORMAL_STRETCH_XX_FS) - ex_exact) > errtol) valueError = true;
    if(std::abs(schema2->local_field_value(i,NORMAL_STRETCH_YY_FS) - ey_exact) > errtol) valueError = true;
    if(std::abs(schema2->local_field_value(i,SHEAR_STRETCH_XY_FS)) > errtol) valueError = true;
    if(std::abs(schema2->local_field_value(i,STATUS_FLAG_FS) - DICe::INITIALIZE_USING_PREVIOUS_FRAME_SUCCESSFUL) > errtol) valueError = true;
    if(std::abs(schema2->local_field_value(i,SIGMA_FS) + 1) < errtol) valueError = true; // check that sigma != -1
  }
  if(valueError){
    *outStream << "---> POSSIBLE ERROR ABOVE! Solution values not correct for multiple subsets " << std::endl;
    errorFlag++;
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

