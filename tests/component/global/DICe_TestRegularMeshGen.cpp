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
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>
#include <DICe_MeshEnums.h>
#include <DICe_TriangleUtils.h>

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
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  const scalar_t begin_x = -0.5;
  const scalar_t end_x = 3.5;
  const scalar_t begin_y = 1.5;
  const scalar_t end_y = 5.5;
  const scalar_t h = 1.0;
  std::vector<int_t> dirichlet_sides;
  std::vector<int_t> neumann_sides;
  for(int_t i=0;i<4;++i)
    dirichlet_sides.push_back(i);

  Teuchos::RCP<DICe::mesh::Mesh> mesh =
      DICe::generate_regular_tri_mesh(DICe::mesh::TRI6,begin_x,end_x,begin_y,end_y,h,dirichlet_sides,neumann_sides,"regular_tri_mesh.e");
  *outStream << "creating the output meshes" << std::endl;
  DICe::mesh::create_output_exodus_file(mesh,"./");
  DICe::mesh::create_exodus_output_variable_names(mesh);
  *outStream << "writing an output step" << std::endl;
  scalar_t time = 0.0;
  DICe::mesh::exodus_output_dump(mesh,1,time);
  *outStream << "closing the exodus output files" << std::endl;
  DICe::mesh::close_exodus_output(mesh);

  // test the coordinate locations in the mesh

  const scalar_t width = end_x - begin_x;
  TEUCHOS_TEST_FOR_EXCEPTION(width<=0,std::runtime_error,"Error, invalid width");
  const scalar_t height = end_y - begin_y;
  TEUCHOS_TEST_FOR_EXCEPTION(height<=0,std::runtime_error,"Error, invalid height");
  std::vector<scalar_t> x_ticks;
  for(scalar_t x=begin_x;x<=end_x;x+=h){
    x_ticks.push_back(x);
  }
  std::vector<scalar_t> y_ticks;
  for(scalar_t y=begin_y;y<=end_y;y+=h){
    y_ticks.push_back(y);
  }
  const int_t num_nodes_x = x_ticks.size();
  const int_t num_nodes_y = y_ticks.size();
  // set up the corner nodes
  std::vector<scalar_t> x_coords;
  std::vector<scalar_t> y_coords;
  for(int_t j=0;j<num_nodes_y;++j){
    for(int_t i=0;i<num_nodes_x;++i){
      x_coords.push_back(x_ticks[i]);
      y_coords.push_back(y_ticks[j]);
    }
  }
  // set up the mid nodes in y
  for(int_t j=0;j<num_nodes_y-1;++j){
    for(int_t i=0;i<2*num_nodes_x-1;++i){
      x_coords.push_back(begin_x + i*0.5*h);
      y_coords.push_back(begin_y + j*h + 0.5*h);
    }
  }
  // set up the mid nodes in x
  for(int_t j=0;j<num_nodes_y;++j){
    for(int_t i=0;i<num_nodes_x-1;++i){
      x_coords.push_back(begin_x + i*h + 0.5*h);
      y_coords.push_back(begin_y + j*h);
    }
  }
  const int_t num_nodes = x_coords.size();
  const int_t num_elem = (num_nodes_x-1)*(num_nodes_y-1)*2;

  *outStream << "testing the output mesh for correctness" << std::endl;
  if((int_t)mesh->num_nodes()!=num_nodes){
    *outStream << "Error, wrong number of nodes in the mesh" << std::endl;
    errorFlag++;
  }
  if((int_t)mesh->num_elem()!=num_elem){
    *outStream << "Error, wrong number of elements" << std::endl;
    errorFlag++;
  }
  if(errorFlag==0){
    Teuchos::RCP<MultiField> coords_x = mesh->get_field(DICe::mesh::field_enums::INITIAL_COORDINATES_X_FS);
    Teuchos::RCP<MultiField> coords_y = mesh->get_field(DICe::mesh::field_enums::INITIAL_COORDINATES_Y_FS);
    const scalar_t tol = 1.0E-5;
    for(int_t i=0;i<num_nodes;++i){
      scalar_t error_x = std::abs(coords_x->local_value(i) - x_coords[i]);
      scalar_t error_y = std::abs(coords_y->local_value(i) - y_coords[i]);
      if(error_x>tol||error_y>tol){
        *outStream << "Error, coordinates are not correct" << std::endl;
        errorFlag++;
      }
    }
  }
  *outStream << "mesh has been tested for correctness" << std::endl;

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

