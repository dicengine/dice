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
#include <DICe_TriangleUtils.h>
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>
//#include <DICe_Parser.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>
#include <fstream>

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

  *outStream << "creating an ROI file " << std::endl;

  std::ofstream roi_file;
  std::string roi_file_name = "roi.txt";
  roi_file.open(roi_file_name);
  roi_file << "begin region_of_interest\n";
  roi_file << "  begin boundary\n";
  roi_file << "    begin polygon\n";
  roi_file << "      begin vertices\n";
  roi_file << "        0 0\n";
  roi_file << "        100 0\n";
  roi_file << "        100 50\n";
  roi_file << "        0 50\n";
  roi_file << "      end vertices\n";
  roi_file << "    end polygon\n";
  roi_file << "  end boundary\n";
  roi_file << "  begin excluded\n";
  roi_file << "    begin polygon\n";
  roi_file << "      begin vertices\n";
  roi_file << "        40 15\n";
  roi_file << "        60 15\n";
  roi_file << "        60 35\n";
  roi_file << "        40 35\n";
  roi_file << "      end vertices\n";
  roi_file << "    end polygon\n";
  roi_file << "    begin polygon\n";
  roi_file << "      begin vertices\n";
  roi_file << "        5 40\n";
  roi_file << "        10 40\n";
  roi_file << "        10 45\n";
  roi_file << "        5 45\n";
  roi_file << "      end vertices\n";
  roi_file << "    end polygon\n";
  roi_file << "  end excluded\n";
  roi_file << "  dirichlet_bc boundary 0 0 3\n";
  roi_file << "  dirichlet_bc boundary 0 1 2\n";
  roi_file << "end region_of_interest\n";
  roi_file.close();

//  std::vector<scalar_t> pts_x;
//  std::vector<scalar_t> pts_y;
//  const Teuchos::RCP<Subset_File_Info> subset_file_info = DICe::read_subset_file(roi_file_name);
//  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > roi_defs = subset_file_info->conformal_area_defs;
//  DEBUG_MSG("Number of conformal area defs: " << roi_defs->size());
//  TEUCHOS_TEST_FOR_EXCEPTION(roi_defs->size()!=1,std::runtime_error,"Error, global ROI file requires one region of interest definition");
//
//  std::map<int_t,DICe::Conformal_Area_Def>::iterator map_it=roi_defs->begin();
//  TEUCHOS_TEST_FOR_EXCEPTION(map_it->second.boundary()->size()!=1,std::runtime_error,
//    "Error, only one polygon is allowed for a global ROI boundary.");
//  // cast the boundary shape to a polygon and test if successful:
//  Teuchos::RCP<DICe::Polygon> boundary_polygon =
//      Teuchos::rcp_dynamic_cast<DICe::Polygon>((*map_it->second.boundary())[0]);
//  TEUCHOS_TEST_FOR_EXCEPTION(boundary_polygon==Teuchos::null,std::runtime_error,"Error, failed cast to polygon.");
//  const int_t num_boundary_vertices = boundary_polygon->num_vertices();
//  DEBUG_MSG("Number of vertices in the boundary polygon: " << num_boundary_vertices);
//  if(num_boundary_vertices!=4){
//    *outStream << "Error, wrong number of boundary vertices" << std::endl;
//    errorFlag++;
//  }
//  for(int_t i=0;i<num_boundary_vertices;++i){
//    pts_x.push_back((*boundary_polygon->vertex_coordinates_x())[i]);
//    pts_y.push_back((*boundary_polygon->vertex_coordinates_y())[i]);
//  }
//
//  // gather all the excluded regions in the roi
//  const int_t num_excluded_shapes = map_it->second.excluded_area()->size();
//  Teuchos::ArrayRCP<scalar_t> holes_x(num_excluded_shapes);
//  Teuchos::ArrayRCP<scalar_t> holes_y(num_excluded_shapes);
//  std::vector<int_t> segments_left;
//  std::vector<int_t> segments_right;
//  DEBUG_MSG("Number of excluded shapes: " << num_excluded_shapes);
//  std::vector<int_t> num_excluded_vertices(num_excluded_shapes);
//  if(map_it->second.excluded_area()->size()>0){
//    for(size_t i=0;i<map_it->second.excluded_area()->size();++i){
//      DEBUG_MSG("Excluded shape " << i);
//      // cast the excluded shape to a polygon and throw is unsuccessful
//      Teuchos::RCP<DICe::Polygon> excluded_polygon =
//          Teuchos::rcp_dynamic_cast<DICe::Polygon>((*map_it->second.excluded_area())[i]);
//      TEUCHOS_TEST_FOR_EXCEPTION(excluded_polygon==Teuchos::null,std::runtime_error,"Error, failed cast to polygon.");
//      num_excluded_vertices[i] = excluded_polygon->num_vertices();
//      DEBUG_MSG("Number of vertices in the excluded polygon: " << num_excluded_vertices[i]);
//      int_t start_index = pts_x.size();
//      //std::cout << "start index " << start_index << std::endl;
//
//      for(int_t p=0;p<num_excluded_vertices[i];++p){
//        if(p==num_excluded_vertices[i]-1){
//          segments_left.push_back(start_index + p + 1);
//          segments_right.push_back(start_index + 1);
//          //std::cout << " adding segment " << segments_left[segments_left.size()-1] << " "  << segments_right[segments_right.size()-1] << std::endl;
//        }
//        else{
//          segments_left.push_back(start_index + p + 1);
//          segments_right.push_back(start_index + p + 2);
//          //std::cout << " adding segment " << segments_left[segments_left.size()-1] << " "  << segments_right[segments_right.size()-1] << std::endl;
//        }
//        pts_x.push_back((*excluded_polygon->vertex_coordinates_x())[p]);
//        pts_y.push_back((*excluded_polygon->vertex_coordinates_y())[p]);
//      }
//      // find a point internal to the excluded hole
//      // get the extents of the polygon
//      std::set<std::pair<int_t,int_t> > owned_pixels = excluded_polygon->get_owned_pixels();
//      const int_t min_x = excluded_polygon->min_x();
//      const int_t max_x = excluded_polygon->max_x();
//      const int_t min_y = excluded_polygon->min_y();
//      const int_t max_y = excluded_polygon->max_y();
//      TEUCHOS_TEST_FOR_EXCEPTION(max_x-min_x < 4,std::runtime_error,"Error, excluded regions is too small");
//      TEUCHOS_TEST_FOR_EXCEPTION(max_y-min_y < 4,std::runtime_error,"Error, excluded regions is too small");
//      const int_t pt_y = (max_y - min_y)/2 + min_y;
//      int_t pt_x = min_x;
//      int_t valid_pixels = 0;
//      for(int_t x=min_x;x<=max_x;++x){
//        if(owned_pixels.find(std::pair<int_t,int_t>(pt_y,x))!=owned_pixels.end()){
//          if(valid_pixels>1){ // prevent the pixel along the shape edge from being selected
//            pt_x = x;
//            break;
//          }
//          valid_pixels++;
//        }
//      }
//      holes_x[i] = pt_x;
//      holes_y[i] = pt_y;
//      DEBUG_MSG("Interior point for the hole " << pt_x << " " << pt_y);
//      if(num_excluded_vertices[i]!=4){
//        *outStream << "Error, wrong number of excluded vertices" << std::endl;
//        errorFlag++;
//      }
//      // add the excluded area segments as neumann boundary segments
//
//    }
//  }
//  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_left(&segments_left[0],0,segments_left.size(),false);
//  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_right(&segments_right[0],0,segments_left.size(),false);
//
//  // gather the boundary segments
//  const int_t num_boundary_segments = subset_file_info->boundary_condition_defs->size();
//  DEBUG_MSG("Number of dirichlet boundary segments: " << num_boundary_segments);
//  if(num_boundary_segments!=2){
//    *outStream << "Error, wrong number of dirichlet boundary segments" << std::endl;
//    errorFlag++;
//  }
//  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_left(num_boundary_segments);
//  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_right(num_boundary_segments);
//  for(int_t i=0;i<num_boundary_segments;++i){
//    dirichlet_boundary_segments_left[i] = (*subset_file_info->boundary_condition_defs)[i].left_vertex_id_+1; // +1 because ids are 1-based in Triangle
//    dirichlet_boundary_segments_right[i] = (*subset_file_info->boundary_condition_defs)[i].right_vertex_id_+1;
//  }
//
//
//  Teuchos::ArrayRCP<scalar_t> points_x(&pts_x[0],0,pts_x.size(),false);
//  Teuchos::ArrayRCP<scalar_t> points_y(&pts_y[0],0,pts_y.size(),false);
//
//  Teuchos::RCP<DICe::mesh::Mesh> mesh = generate_tri6_mesh(
//    points_x,
//    points_y,
//    holes_x,
//    holes_y,
//    dirichlet_boundary_segments_left,
//    dirichlet_boundary_segments_right,
//    neumann_boundary_segments_left,
//    neumann_boundary_segments_right,
//    max_size_constraint,
//    "triangle_test_mesh.e");
  const scalar_t max_size_constraint = 75.0;
  const std::string file_name = "roi.txt";
  Teuchos::RCP<DICe::mesh::Mesh> mesh = generate_tri6_mesh(file_name,max_size_constraint,"triangle_test_mesh.e");

  DICe::mesh::create_output_exodus_file(mesh,"");
  DICe::mesh::create_exodus_output_variable_names(mesh);
  DICe::mesh::exodus_output_dump(mesh,1,1.0);

  *outStream << " checking the output mesh dimensions" << std::endl;

  if(mesh->num_elem()!=104){
    *outStream << "Error, wrong number of elements" << std::endl;
    errorFlag++;
  }
  if(mesh->num_nodes()!=241){
    *outStream << "Error, wrong number of nodes" << std::endl;
    errorFlag++;
  }
  if(mesh->num_node_sets()!=2){
    *outStream << "Error, wrong number of nodes sets" << std::endl;
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

