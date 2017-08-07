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

/*! \file  DICe_TestInitializer.cpp
    \brief Testing of Path_Initializer class
*/

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Subset.h>
#include <DICe_Initializer.h>
#include <DICe_Schema.h>
#include <DICe_ParameterUtilities.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <iostream>
#include <fstream>
#include <cassert>

using namespace DICe;
using namespace DICe::mesh::field_enums;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  const scalar_t errorTol = 1.0E-4;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  *outStream << "writing the path file" << std::endl;
  std::ofstream outfile("sample.path");
  if(outfile.is_open()){
    for(int_t i=0;i<10;++i){
      outfile << "100.25 234.01 456.098\n";
      outfile << "456.12 15.89 0.0001\n";
      outfile << "52.01 45.988 0.25\n";
    }
    outfile.close();
  }
  else
    errorFlag++;

  *outStream << "creating a Path_Initializer" << std::endl;
  *outStream << "tyring a non-existent file on purpose" << std::endl;

  Teuchos::RCP<Image> image = Teuchos::null;
  Teuchos::RCP<Subset> subset = Teuchos::null;

  bool exception_thrown = false;
  try{
    Path_Initializer bad_path(NULL,subset,"sample.txt");
  }
  catch(std::exception &e){
    *outStream << "an exception was thrown as it should have been" << std::endl;
    exception_thrown = true;
  }
  if(!exception_thrown){
    *outStream << "Error, an exception should have been thrown here" << std::endl;
    errorFlag++;
  }
  *outStream << "trying a valid file" << std::endl;
  const size_t num_neighbors = 2;
  Path_Initializer path(NULL,subset,"sample.path",2);
  *outStream << "the path has " << path.num_triads() << " unique points" << std::endl;
  if(path.num_triads()!=3){
    *outStream << "Error, wrong number of triads." << std::endl;
    errorFlag++;
  }
  if(path.num_neighbors()!=num_neighbors){
    *outStream << "Error, wrong number of neighbors" << std::endl;
    errorFlag++;
  }
  *outStream << "testing the values of the triads" << std::endl;
  std::vector<scalar_t> exact_u(3,0.0);
  std::vector<scalar_t> exact_v(3,0.0);
  std::vector<scalar_t> exact_t(3,0.0);
  exact_u[0] = 52;   exact_u[1] = 100.5; exact_u[2] = 456;
  exact_v[0] = 46;   exact_v[1] = 234;   exact_v[2] = 16;
  exact_t[0] = 0.25; exact_t[1] = 456.1; exact_t[2] = 0;
  int_t index = 0;
  std::set<def_triad>::iterator it = path.triads()->begin();
  for(;it!=path.triads()->end();++it){
    if(index >= 3) {errorFlag++; continue;}
    def_triad t = *it;
    *outStream << "   " << t.u_ << " " << t.v_ << " " << t.t_ << std::endl;
    if(t.u_!=exact_u[index]){
      *outStream << "Error, wrong u value " << t.u_ << " should be " << exact_u[index] << std::endl;
      errorFlag++;
    }
    if(t.v_!=exact_v[index]){
      *outStream << "Error, wrong v value " << t.v_ << " should be " << exact_v[index] << std::endl;
      errorFlag++;
    }
    if(t.t_!=exact_t[index]){
      *outStream << "Error, wrong t value " << t.t_ << " should be " << exact_t[index] << std::endl;
      errorFlag++;
    }
    index++;
  }
  *outStream << "testing the neighbor list" << std::endl;
  // all ids should have all neighbors
  size_t exact_neigh[][3] = {{0,2},{1,0},{2,0}};
  for(size_t i=0;i<path.num_triads();++i){
    for(size_t j=0;j<path.num_neighbors();++j){
      if(path.neighbor(i,j)!=exact_neigh[i][j]){
        *outStream << "neighbor error" << std::endl;
        errorFlag++;
      }
    }
  }
  *outStream << "testing the closest point function" << std::endl;
  const scalar_t up = 10.0;
  const scalar_t vp = 12.0;
  const scalar_t tp = 0.01;
  scalar_t dist = 0.0;
  size_t id = 0;
  path.closest_triad(up,vp,tp,id,dist);
  if(id!=0){
    *outStream << "Error, the closest triad is wrong" << std::endl;
    errorFlag++;
  }
  if(std::abs(dist - 2920.06) > 1.0E-2){
    *outStream << "Error, the distance to this point is wrong" << std::endl;
    errorFlag++;
  }

  *outStream << "testing the initializers in a schema" << std::endl;
  const int_t num_subsets = 4;
  const int_t subset_size = 27;
  Teuchos::RCP<std::vector<int_t> > neighbor_ids = Teuchos::rcp(new std::vector<int_t>(4,0));
  (*neighbor_ids)[0] = -1; // subsets 1 2 and 3 all have 0 as their seed neighbor, but 0 has -1 because it is the seed
  Teuchos::ArrayRCP<scalar_t> coords_x(num_subsets,0);
  Teuchos::ArrayRCP<scalar_t> coords_y(num_subsets,0);
  coords_x[0] = 151; coords_y[0] = 277;
  coords_x[1] = 209; coords_y[1] = 269;
  coords_x[2] = 153; coords_y[2] = 328;
  coords_x[3] = 211; coords_y[3] = 336;
  const scalar_t u_exact = 138.0;
  const scalar_t v_exact = -138.0;
  // write a simple path file to be used in the examples below:
  std::ofstream path_file;
  path_file.open("InitializerTest.path");
  path_file << u_exact << " " << v_exact << " 0.0\n";
  path_file.close();
  Teuchos::RCP<std::map<int_t,std::string> > path_file_names = Teuchos::rcp(new std::map<int_t,std::string>());
  path_file_names->insert(std::pair<int_t,std::string>(0,"InitializerTest.path")); // use a path file for the first subset
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  std::vector<Initialization_Method> init_methods;
  init_methods.push_back(USE_PHASE_CORRELATION);
  init_methods.push_back(USE_FIELD_VALUES);
  init_methods.push_back(INITIALIZATION_METHOD_NOT_APPLICABLE); // use this one for path file test
  std::vector<Correlation_Routine> corr_routines;
  corr_routines.push_back(TRACKING_ROUTINE);
  corr_routines.push_back(GENERIC_ROUTINE);

  for(size_t init_i = 0;init_i<init_methods.size();++init_i){
    for(size_t corr_i = 0;corr_i<corr_routines.size();++corr_i){
      *outStream << "testing initializer for combination " << init_i << " " << corr_i << std::endl;
      // if the init method was set to N/A, then use neighbor values automatically
      params->set(DICe::initialization_method,init_methods[init_i]==INITIALIZATION_METHOD_NOT_APPLICABLE?USE_NEIGHBOR_VALUES:init_methods[init_i]);
      params->set(DICe::correlation_routine,corr_routines[corr_i]);
      params->set(DICe::disp_jump_tol,500.0);
      params->set(DICe::theta_jump_tol,100.0);
      Teuchos::RCP<DICe::Schema> schema = Teuchos::rcp(new DICe::Schema(coords_x,coords_y,subset_size,Teuchos::null,neighbor_ids,params));
      schema->set_ref_image("./images/InitRef.tif");
      schema->set_def_image("./images/InitDef.tif");
      if(init_methods[init_i]==INITIALIZATION_METHOD_NOT_APPLICABLE)
        schema->set_path_file_names(path_file_names);
      for(int_t i=0;i<num_subsets;++i){
        schema->local_field_value(i,SUBSET_COORDINATES_X_FS) = coords_x[i];
        schema->local_field_value(i,SUBSET_COORDINATES_Y_FS) = coords_y[i];
        if(init_methods[init_i] == USE_FIELD_VALUES){
          schema->local_field_value(i,SUBSET_DISPLACEMENT_X_FS) = u_exact;
          schema->local_field_value(i,SUBSET_DISPLACEMENT_Y_FS) = v_exact;
        }
      }
      bool exception_thrown = false;
      try{
         schema->execute_correlation();
      }
      catch(std::exception &e){
        exception_thrown = true;
      }
      // an error should have thrown for path files used with generic routine
      if(init_methods[init_i]==INITIALIZATION_METHOD_NOT_APPLICABLE && corr_routines[corr_i]==GENERIC_ROUTINE){
        if(!exception_thrown){
          *outStream << "Error, an exception should have thrown here" << std::endl;
          errorFlag++;
        }
        else{
          *outStream << "exception thrown as expected" << std::endl;
        }
        continue;
      }
      //schema->print_fields();
      for(int_t i=0;i<schema->local_num_subsets();++i){
        const scalar_t u = schema->local_field_value(i,SUBSET_DISPLACEMENT_X_FS);
        const scalar_t v = schema->local_field_value(i,SUBSET_DISPLACEMENT_Y_FS);
        *outStream << i << " u: " << u << " v: " << v << std::endl;
        if(std::abs(u-u_exact) > errorTol || std::abs(v-v_exact) > errorTol){
          *outStream << "Error, the displacement values are not correct" << std::endl;
          errorFlag++;
        }
      }
      *outStream << "output values have been tested for combination " << init_i << " " << corr_i << std::endl;
    }
  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

