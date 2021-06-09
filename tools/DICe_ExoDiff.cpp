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

/*! \file  DICe_ExoDiff.cpp
    \brief Utility for diffing exodus files
*/

#include <DICe.h>
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>
#include <DICe_FieldEnums.h>
#include <DICe_MeshEnums.h>
#include <DICe_ParameterUtilities.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>

#include <fstream>
#include <cassert>

using namespace DICe;

int main(int argc, char *argv[]) {

  // config_file format: 2 columns field_name tolerance, one row for each field to check (must be a valid field), no commas, just a space separator

  DICe::initialize(argc, argv);

  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
  int_t errorFlag = 0;

  if(argc!=4){
    *outStream << "Invalid number of command line parameters. Usage: ./DICe_ExoDiff <exodus_file_A> <exodus_file_B> <config_file>" << std::endl;
    std::cout << "End Result: TEST FAILED\n";
    return 1;
  }

  std::string file_a_name = argv[1];
  *outStream << "File A: " << file_a_name << std::endl;

  std::string file_b_name = argv[2];
  *outStream << "File B: " << file_b_name << std::endl;

  std::string config_file_name = argv[3];
  *outStream << "Configuration file: " << config_file_name << std::endl;

  // read the configuration file
  std::fstream config_file(config_file_name,std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!config_file.good(),std::runtime_error,"Error, failed to open config file");

  int_t num_lines = 0;
  std::string line;
  // get the number of lines in the file:
  while (std::getline(config_file, line)){
    ++num_lines;
  }
  DEBUG_MSG("number of fields to compare: " << num_lines);
  config_file.clear();
  config_file.seekg(0,std::ios::beg);

  // there are 2 columns of data
  // field_name tolerance
  std::vector<std::string> field_names(num_lines);
  std::vector<scalar_t> tols(num_lines);
  *outStream << std::left << std::setw(25) << "Field";
  *outStream << std::left << std::setw(15) << "Tolerance" << std::endl;
  for(int_t line=0;line<num_lines;++line){
    config_file >> field_names[line] >> tols[line];
    stringToUpper(field_names[line]);
    *outStream << std::left << std::setw(25) << field_names[line];
    *outStream << std::left << std::setw(15) << tols[line] << std::endl;
  }
  config_file.close();
  TEUCHOS_TEST_FOR_EXCEPTION(tols.size()!=field_names.size(),std::runtime_error,"Error, tols and field_names must be the same size");

  for(size_t i=0;i<field_names.size();++i){
  }

  // Read the first mesh
  *outStream << "reading " << file_a_name << std::endl;
  Teuchos::RCP<DICe::mesh::Mesh> mesh_a = DICe::mesh::read_exodus_mesh(file_a_name,"dummy_output_name.e");
  std::vector<std::string> mesh_a_fields = DICe::mesh::read_exodus_field_names(mesh_a);
  *outStream << "fields in " << file_a_name << std::endl;
  for(size_t i=0;i<mesh_a_fields.size();++i){
    *outStream << mesh_a_fields[i] << std::endl;
  }
  int_t num_steps_a = DICe::mesh::read_exodus_num_steps(mesh_a);
  *outStream << "number of time steps " << num_steps_a << std::endl;
  *outStream << "reading " << file_b_name << std::endl;
  Teuchos::RCP<DICe::mesh::Mesh> mesh_b = DICe::mesh::read_exodus_mesh(file_b_name,"dummy_output_name.e");
  std::vector<std::string> mesh_b_fields = DICe::mesh::read_exodus_field_names(mesh_b);
  *outStream << "fields in " << file_b_name << std::endl;
  for(size_t i=0;i<mesh_b_fields.size();++i){
    *outStream << mesh_b_fields[i] << std::endl;
  }
  int_t num_steps_b = DICe::mesh::read_exodus_num_steps(mesh_b);
  *outStream << "number of time steps " << num_steps_b << std::endl;

  // check that the meshes are compatible
  int_t num_nodes_a = mesh_a->num_nodes();
  int_t num_nodes_b = mesh_b->num_nodes();
  TEUCHOS_TEST_FOR_EXCEPTION(num_nodes_a!=num_nodes_b,std::runtime_error,"Error, meshes not compatible. Num_nodes differs");

  TEUCHOS_TEST_FOR_EXCEPTION(num_steps_a!=num_steps_b,std::runtime_error,"Error, files not compatible. Number of steps in each is different");

  *outStream << "--------------------------------------------------------------" << std::endl;
  *outStream << std::left << std::setw(25) << "Field";
  *outStream << std::left << std::setw(5) <<"Step";
  *outStream << std::left << std::setw(15) <<"Error" << std::endl;

  // check that the fields are valid for both meshes
  for(size_t i=0;i<field_names.size();++i){
    bool field_found_in_a = false;
    bool field_found_in_b = false;
    int_t var_index_a = 0;
    int_t var_index_b = 0;
    for(size_t j=0;j<mesh_a_fields.size();++j){
      if(mesh_a_fields[j]==field_names[i]){
        field_found_in_a = true;
        var_index_a = j+1;
      }
    }
    for(size_t j=0;j<mesh_b_fields.size();++j){
      if(mesh_b_fields[j]==field_names[i]){
        field_found_in_b = true;
        var_index_b = j+1;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(!field_found_in_a,std::runtime_error,"Error, field not found in mesh_a: " << field_names[i]);
    TEUCHOS_TEST_FOR_EXCEPTION(!field_found_in_b,std::runtime_error,"Error, field not found in mesh_b: " << field_names[i]);

    // iterate all steps in the file
    for(int_t time_step=1;time_step<=num_steps_a;++time_step){
      std::vector<scalar_t> a_field = DICe::mesh::read_exodus_field(mesh_a,var_index_a,time_step);
      std::vector<scalar_t> b_field = DICe::mesh::read_exodus_field(mesh_b,var_index_b,time_step);

      scalar_t error = 0.0;
      for(int_t node=0;node<num_nodes_a;++node){
        error += (a_field[node] - b_field[node])*(a_field[node] - b_field[node]);
      }
      error = std::sqrt(error);
      *outStream << std::left << std::setw(25) << field_names[i];
      *outStream << std::left << std::setw(5) << time_step;
      *outStream << std::left << std::setw(15) << error << std::endl;
      if(error > tols[i]){
        *outStream << "Error, difference is greater than the tolerance for field: " << field_names[i] << " step " << time_step << " error " << error << std::endl;
        errorFlag++;
      }
    }
  }
  *outStream << "--------------------------------------------------------------" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;
}

