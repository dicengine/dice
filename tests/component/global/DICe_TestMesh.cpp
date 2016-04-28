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

  *outStream << "creating an exodus mesh object" << std::endl;
  const std::string mesh_file_name = "./meshes/mesh_test_in.g";
  const std::string mesh_output_file_name = "mesh_test_out.e";
  Teuchos::RCP<DICe::mesh::Mesh> mesh = DICe::mesh::read_exodus_mesh(mesh_file_name,mesh_output_file_name);
  *outStream << "checking the basic properties of the input mesh" << std::endl;
  if(mesh->num_nodes()!=257){
    *outStream << "Error, the number of nodes read from the input mesh is not correct" << std::endl;
    errorFlag++;
  }
  if(mesh->num_elem()!=330){
    *outStream << "Error, the number of elements read from the input mesh is not correct" << std::endl;
    errorFlag++;
  }
  if(mesh->num_blocks()!=2){
    *outStream << "Error, the number of blocks in the input mesh is not correct" << std::endl;
    errorFlag++;
  }
  // mesh should start out with 0 internal faces, etc. these are initialized later
  if(mesh->num_internal_faces()!=0){
    *outStream << "Error, the number of internal faces is not correct, is " << mesh->num_internal_faces() << " should be 0 at this point" << std::endl;
    errorFlag++;
  }
  if(mesh->num_subelem()!=0){
    *outStream << "Error, the number of sub elements is not correct, is " << mesh->num_subelem() << " should be 0 at this point" << std::endl;
    errorFlag++;
  }
  // iterate the blocks and check the element type
  DICe::mesh::block_type_map::iterator block_map_it = mesh->get_block_type_map()->begin();
  DICe::mesh::block_type_map::iterator block_map_end = mesh->get_block_type_map()->end();
  int_t block_index = 0;
  for(block_map_it = mesh->get_block_type_map()->begin();block_map_it!=block_map_end;++block_map_it)
  {
    const DICe::mesh::Base_Element_Type elem_type_enum = block_map_it->second;
    *outStream << "mesh has block of element type " << tostring(elem_type_enum) << std::endl;
    if(block_index==0){
      if(elem_type_enum!=DICe::mesh::QUAD4){
        *outStream << "Error, the first block does not have the right element type" << std::endl;
        errorFlag++;
      }
    }
    else{
      if(elem_type_enum!=DICe::mesh::TRI3){
        *outStream << "Error, the second block does not have the right element type" << std::endl;
        errorFlag++;
      }
    }
    block_index++;
  }
  *outStream << "the basic properties of the input mesh have been checked" << std::endl;

  *outStream << "checking that the coordinates fields have been created " << std::endl;
  if(!mesh->field_exists("INITIAL_COORDINATES")){
    *outStream << "Error, the initial coordinates fields do not exist, but should" << std::endl;
    errorFlag++;
  }
  *outStream << "coordinate fields have been checked" << std::endl;

  *outStream << "checking the boundary conditions on the input mesh" << std::endl;
  DICe::mesh::side_set_info & ss_info = *mesh->get_side_set_info();
  int_t num_side_sets = ss_info.ids.size();
  if(num_side_sets!=2){
    *outStream << "error, the number of side sets is not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "the boundary conditions have been checked" << std::endl;

  *outStream << "creating the internal faces and cells used for CVFEM type methods" << std::endl;
  DICe::mesh::create_cell_size_and_radius(mesh);
  DICe::mesh::initialize_control_volumes(mesh);
  *outStream << "number of internal faces " << mesh->num_internal_faces() << std::endl;
  *outStream << "number of sub elements " << mesh->num_subelem() << std::endl;
  if(mesh->num_internal_faces()!=1370){
    *outStream << "Error, the number of internal faces is not correct, is " << mesh->num_internal_faces() << " should be 1370" << std::endl;
    errorFlag++;
  }
  if(mesh->num_subelem()!=430){
    *outStream << "Error, the number of sub elements is not correct, is " << mesh->num_subelem() << " should be 430" << std::endl;
    errorFlag++;
  }
  *outStream << "internal faces and cells have been checked" << std::endl;

  *outStream << "creating some fields on the mesh" << std::endl;
  mesh->create_field(mesh::field_enums::CVFEM_AD_PHI_FS);
  mesh->create_field(mesh::field_enums::CVFEM_AD_IMAGE_PHI_FS);
  mesh->create_field(mesh::field_enums::CVFEM_AD_LAMBDA_FS);
  *outStream << "populating values for phi field" << std::endl;
  MultiField & phi = *mesh->get_field(mesh::field_enums::CVFEM_AD_PHI_FS);
  MultiField & coords = *mesh->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  Teuchos::ArrayRCP<const scalar_t> coords_values = coords.get_1d_view();
  const int_t spa_dim = mesh->spatial_dimension();
  for(size_t i=0;i<mesh->num_nodes();++i){
    phi.local_value(i) = coords_values[i*spa_dim]*10.0;
  }
  *outStream << "fields have been created" << std::endl;

  *outStream << "creating the output meshes" << std::endl;
  DICe::mesh::create_output_exodus_file(mesh,"./");
  DICe::mesh::create_face_edge_output_exodus_file(mesh,"./");
  DICe::mesh::create_exodus_output_variable_names(mesh);
  DICe::mesh::create_face_edge_output_variable_names(mesh);

  *outStream << "writing an output step" << std::endl;
  scalar_t time = 0.0;
  DICe::mesh::exodus_output_dump(mesh,1,time);
  DICe::mesh::exodus_face_edge_output_dump(mesh,1,time);

  *outStream << "closing the exodus output files" << std::endl;
  DICe::mesh::close_exodus_output(mesh);
  DICe::mesh::close_face_edge_exodus_output(mesh);

  *outStream << "output files have been written" << std::endl;

  *outStream << "checking the output file for correct mesh properties" << std::endl;
  Teuchos::RCP<DICe::mesh::Mesh> mesh_out = DICe::mesh::read_exodus_mesh(mesh_output_file_name,"no_file.e");
  *outStream << "checking the basic properties of the output mesh" << std::endl;
  if(mesh_out->num_nodes()!=257){
    *outStream << "Error, the number of nodes read from the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  if(mesh_out->num_elem()!=330){
    *outStream << "Error, the number of elements read from the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  if(mesh_out->num_blocks()!=2){
    *outStream << "Error, the number of blocks in the output mesh is not correct" << std::endl;
    errorFlag++;
  }
  *outStream << "output mesh properties have been checked" << std::endl;
// TODO test the field values in the output mesh
//  *outStream << "checking the output file for correct fields" << std::endl;
//  MultiField & phi_out = *mesh_out->get_field(mesh::field_enums::CVFEM_AD_PHI_FS);
//  bool field_value_error = false;
//  for(int_t i=0;i<mesh_out->num_nodes();++i){
//    if(std::abs(phi_out.local_value(i)-coords_values[i*spa_dim]*10.0)>1.0E-3){
//      field_value_error=true;
//    }
//  }
//  if(field_value_error){
//    *outStream << "error, the fields in the output mesh are not correct" << std::endl;
//    errorFlag++;
//  }
//  *outStream << "output fields have been checked" << std::endl;

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

