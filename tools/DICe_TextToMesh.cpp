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
#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <DICe_TriangleUtils.h>
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cassert>

using namespace std;
using namespace DICe;

int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  Teuchos::oblackholestream bhs; // outputs nothing
  Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&bhs, false);

  // read the parameters from the input file:
  if(argc!=4){ // executable and input file
    std::cout << "Usage: DICe_TextTomesh <intput_text_file.txt> <other_dim_is_y(1 or 0)> <other_dimension>" << std::endl;
    exit(1);
  }
  /// first dimension is taken from the data
  /// the second dimension is from the command line
  std::string data_file_name = argv[1];
  DEBUG_MSG("data file name: " << data_file_name);
  const bool other_dim_is_y = std::stoi(argv[2])==1;
  DEBUG_MSG("other dim is y: " << other_dim_is_y);
  const scalar_t other_dim = std::stoi(argv[3]);
  DEBUG_MSG("other dim: " << other_dim);

  // read the text input file:
  // should be in the format coordinate_value solution_value, one pair for each point along a line
  // read in the solution file:
  std::string line;
  std::fstream data_file(data_file_name,std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!data_file.is_open(),std::runtime_error,"Error, unable to load data file.");
  int_t num_points = 0;
  int_t num_cols = 0;
  // get the number of lines in the file:
  while (std::getline(data_file, line)){
    if(num_points==0){
      // determine the number of values
      std::string token;
      size_t pos = 0;
      std::string delimiter = ",";
      while ((pos = line.find(delimiter)) != std::string::npos) {
          token = line.substr(0, pos);
          line.erase(0, pos + delimiter.length());
          num_cols++;
      }
      if(!line.empty())
        num_cols++;
    }
    ++num_points;
  }
  DEBUG_MSG("number of points in the input path file: " << num_points);
  DEBUG_MSG("number of columns in the input path file: " << num_cols);
  data_file.clear();
  data_file.seekg(0,std::ios::beg);

  TEUCHOS_TEST_FOR_EXCEPTION(num_cols > 11,std::runtime_error,"Too many columns in the data file (needs to be 11 or less)");

  std::vector<scalar_t> coords(num_points);
  scalar_t min_coord = std::numeric_limits<scalar_t>::max();
  scalar_t max_coord = 0.0;
  scalar_t avg_dist = 0.0;
  std::vector<std::vector<scalar_t> > values(num_points);
  for(int_t i=0;i<num_points;++i){
    values[i].resize(num_cols-1);
  }

  int_t pt_index = 0;
  while (std::getline(data_file, line)){
    //std::cout << " read line " << line << std::endl;
    // determine the number of values
    std::string token;
    size_t pos = 0;
    std::string delimiter = ",";
    int_t col = 0;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      token = line.substr(0, pos);
      //std::cout << " token " << token << std::endl;
      if(col==0){
        coords[pt_index] = std::stod(token);
        if(coords[pt_index] < min_coord) min_coord = coords[pt_index];
        if(coords[pt_index] > max_coord) max_coord = coords[pt_index];
        if(pt_index>0)
          avg_dist += coords[pt_index] - coords[pt_index-1];
      }
      else{
        values[pt_index][col-1] = std::stod(token);
      }
      col++;
      line.erase(0, pos + delimiter.length());
    }
    if(!line.empty())
      values[pt_index][num_cols-2] = std::stod(line);
    pt_index++;
  }
  data_file.close();
  avg_dist = avg_dist/(num_points-1);
  DEBUG_MSG("Min coord: " << min_coord << " max coord: " << max_coord << " average dist: " << avg_dist);

  //  for(int_t i=0;i<num_points;++i){
  //    std::cout << "coord " << coords[i] << " values ";
  //    for(int_t j=0;j<num_cols-1;++j){
  //      std::cout << " " << values[i][j];
  //    }
  //    std::cout << std::endl;
  //  }

  const scalar_t size_constraint = avg_dist*avg_dist;

  // create an exodus mesh:
  // determine the number of points in the opposite dimension
  const int_t other_dim_num_pts = int_t(other_dim / avg_dist) + 1;
  const int_t mesh_num_points = other_dim_num_pts * num_points;
  DEBUG_MSG("Mesh has " << mesh_num_points << " points");
  Teuchos::ArrayRCP<scalar_t> points_x(mesh_num_points);
  Teuchos::ArrayRCP<scalar_t> points_y(mesh_num_points);
  for(int_t dim_0=0;dim_0<num_points;++dim_0){
    for(int_t dim_1=0;dim_1<other_dim_num_pts;++dim_1){
      if(other_dim_is_y){
        points_x[dim_0*other_dim_num_pts + dim_1] = coords[dim_0];
        points_y[dim_0*other_dim_num_pts + dim_1] = dim_1*avg_dist;
      }
      else{
        points_y[dim_0*other_dim_num_pts + dim_1] = coords[dim_0];
        points_x[dim_0*other_dim_num_pts + dim_1] = dim_1*avg_dist;
      }
    }
  }

  Teuchos::RCP<DICe::mesh::Mesh> mesh = generate_tri6_mesh(points_x,points_y,size_constraint,"DICe_TextToMesh_out.e");

  DICe::mesh::create_output_exodus_file(mesh,"");

  std::vector<DICe::field_enums::Field_Spec> field_specs;
  field_specs.push_back(DICe::field_enums::FIELD_1_FS);
  field_specs.push_back(DICe::field_enums::FIELD_2_FS);
  field_specs.push_back(DICe::field_enums::FIELD_3_FS);
  field_specs.push_back(DICe::field_enums::FIELD_4_FS);
  field_specs.push_back(DICe::field_enums::FIELD_5_FS);
  field_specs.push_back(DICe::field_enums::FIELD_6_FS);
  field_specs.push_back(DICe::field_enums::FIELD_7_FS);
  field_specs.push_back(DICe::field_enums::FIELD_8_FS);
  field_specs.push_back(DICe::field_enums::FIELD_9_FS);
  field_specs.push_back(DICe::field_enums::FIELD_10_FS);

  // populate the various fields
  // create a field for each column
  for(int_t col=0;col<num_cols-1;++col){
    mesh->create_field(field_specs[col]);
    MultiField & field = *mesh->get_field(field_specs[col]);
    for(int_t i=0;i<num_points;++i){
      for(int_t j=0;j<other_dim_num_pts;++j){
        int_t local_id = i*other_dim_num_pts + j;
        field.local_value(local_id) = values[i][col];
      }
    }
  }
  DICe::mesh::create_exodus_output_variable_names(mesh);
  DICe::mesh::exodus_output_dump(mesh,1,1.0);

  DICe::finalize();

  return 0;

}

