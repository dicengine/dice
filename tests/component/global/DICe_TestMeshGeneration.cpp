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

  // define the boundary points
  Teuchos::ArrayRCP<scalar_t> points_x(4);
  Teuchos::ArrayRCP<scalar_t> points_y(4);
  const scalar_t max_size_constraint = 0.1;
  points_x[0] = 0.0; points_y[0] = 0.0;
  points_x[1] = 5.0; points_y[1] = 0.0;
  points_x[2] = 5.0; points_y[2] = 10.0;
  points_x[3] = 0.0; points_y[3] = 10.0;

  Teuchos::RCP<DICe::mesh::Mesh> mesh = DICe::generate_tri_mesh(DICe::mesh::TRI6,points_x,points_y,max_size_constraint,"scratch_mesh.e");
  *outStream << "creating some fields on the mesh" << std::endl;
  //mesh->create_field(mesh::field_enums::CVFEM_AD_PHI_FS);
  //mesh->create_field(mesh::field_enums::CVFEM_AD_IMAGE_PHI_FS);
  //mesh->create_field(mesh::field_enums::CVFEM_AD_LAMBDA_FS);
  *outStream << "populating values for phi field" << std::endl;
  //MultiField & phi = *mesh->get_field(mesh::field_enums::CVFEM_AD_PHI_FS);
  MultiField & coords = *mesh->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  Teuchos::ArrayRCP<const scalar_t> coords_values = coords.get_1d_view();
  //const int_t spa_dim = mesh->spatial_dimension();
  //for(size_t i=0;i<mesh->num_nodes();++i){
  //  phi.local_value(i) = coords_values[i*spa_dim]*10.0;
  //}
  *outStream << "fields have been created" << std::endl;
  *outStream << "creating the output meshes" << std::endl;
  DICe::mesh::create_output_exodus_file(mesh,"./");
  DICe::mesh::create_exodus_output_variable_names(mesh);
  *outStream << "writing an output step" << std::endl;
  scalar_t time = 0.0;
  DICe::mesh::exodus_output_dump(mesh,1,time);
  *outStream << "closing the exodus output files" << std::endl;
  DICe::mesh::close_exodus_output(mesh);
  *outStream << "checking the output file for correct mesh properties" << std::endl;
  Teuchos::RCP<DICe::mesh::Mesh> mesh_out = DICe::mesh::read_exodus_mesh("scratch_mesh.e","no_file.e");
  *outStream << "checking the basic properties of the output mesh" << std::endl;
  if(mesh_out->num_nodes()!=1662){
    *outStream << "Error, the number of nodes read from the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  if(mesh_out->num_elem()!=789){
    *outStream << "Error, the number of elements read from the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  if(mesh_out->num_blocks()!=1){
    *outStream << "Error, the number of blocks in the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "output mesh properties have been checked" << std::endl;

  *outStream << "generating a tri3 mesh from a tri6 mesh" << std::endl;

  Teuchos::RCP<DICe::mesh::Mesh> mesh_tri3 = DICe::mesh::create_tri3_mesh_from_tri6(mesh,"scratch_mesh_tri3.e");
  *outStream << "creating a linear tri3 output mesh" << std::endl;
  DICe::mesh::create_output_exodus_file(mesh_tri3,"./");
  DICe::mesh::create_exodus_output_variable_names(mesh_tri3);
  *outStream << "writing an output step" << std::endl;
  DICe::mesh::exodus_output_dump(mesh_tri3,1,time);
  *outStream << "closing the linear tri3 exodus output file" << std::endl;
  DICe::mesh::close_exodus_output(mesh_tri3);
  *outStream << "checking the output file for correct mesh properties" << std::endl;
  Teuchos::RCP<DICe::mesh::Mesh> mesh_out_tri3 = DICe::mesh::read_exodus_mesh("scratch_mesh_tri3.e","no_file.e");
  *outStream << "checking the basic properties of the output mesh" << std::endl;
  if(mesh_out_tri3->num_nodes()!=437){
    *outStream << "Error, the number of nodes read from the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  if(mesh_out_tri3->num_elem()!=789){
    *outStream << "Error, the number of elements read from the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  if(mesh_out_tri3->num_blocks()!=1){
    *outStream << "Error, the number of blocks in the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "output mesh properties have been checked" << std::endl;

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

