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

#include <DICe_TriangleUtils.h>
#include <DICe_MeshIO.h>
#include <DICe_Mesh.h>
#include <DICe_Parser.h>

#include <triangle.h>

namespace DICe {

Teuchos::RCP<DICe::mesh::Mesh> generate_tri6_mesh(Teuchos::ArrayRCP<scalar_t> points_x,
  Teuchos::ArrayRCP<scalar_t> points_y,
  const scalar_t & max_size_constraint,
  const std::string & output_file_name){
  return generate_tri6_mesh(points_x,
    points_y,
    Teuchos::null,
    Teuchos::null,
    Teuchos::null,
    Teuchos::null,
    Teuchos::null,
    Teuchos::null,
    max_size_constraint,
    output_file_name);
}

Teuchos::RCP<DICe::mesh::Mesh> generate_tri6_mesh(const std::string & roi_file_name,
  const scalar_t & max_size_constraint,
  const std::string & output_file_name){

  std::vector<scalar_t> pts_x;
  std::vector<scalar_t> pts_y;
  const Teuchos::RCP<Subset_File_Info> subset_file_info = DICe::read_subset_file(roi_file_name);
  Teuchos::RCP<std::map<int_t,DICe::Conformal_Area_Def> > roi_defs = subset_file_info->conformal_area_defs;
  DEBUG_MSG("Number of conformal area defs: " << roi_defs->size());
  TEUCHOS_TEST_FOR_EXCEPTION(roi_defs->size()!=1,std::runtime_error,"Error, global ROI file requires one region of interest definition");

  std::map<int_t,DICe::Conformal_Area_Def>::iterator map_it=roi_defs->begin();
  TEUCHOS_TEST_FOR_EXCEPTION(map_it->second.boundary()->size()!=1,std::runtime_error,
    "Error, only one polygon is allowed for a global ROI boundary.");
  // cast the boundary shape to a polygon and test if successful:
  Teuchos::RCP<DICe::Polygon> boundary_polygon =
      Teuchos::rcp_dynamic_cast<DICe::Polygon>((*map_it->second.boundary())[0]);
  TEUCHOS_TEST_FOR_EXCEPTION(boundary_polygon==Teuchos::null,std::runtime_error,"Error, failed cast to polygon.");
  const int_t num_boundary_vertices = boundary_polygon->num_vertices();
  DEBUG_MSG("Number of vertices in the boundary polygon: " << num_boundary_vertices);
  for(int_t i=0;i<num_boundary_vertices;++i){
    pts_x.push_back((*boundary_polygon->vertex_coordinates_x())[i]);
    pts_y.push_back((*boundary_polygon->vertex_coordinates_y())[i]);
  }

  // gather all the excluded regions in the roi
  const int_t num_excluded_shapes = map_it->second.has_excluded_area() ? map_it->second.excluded_area()->size(): 0;
  DEBUG_MSG("Number of excluded shapes: " << num_excluded_shapes);
  Teuchos::ArrayRCP<scalar_t> holes_x(num_excluded_shapes);
  Teuchos::ArrayRCP<scalar_t> holes_y(num_excluded_shapes);
  std::vector<int_t> segments_left;
  std::vector<int_t> segments_right;
  std::vector<int_t> num_excluded_vertices(num_excluded_shapes);
  if(num_excluded_shapes>0){
    for(size_t i=0;i<map_it->second.excluded_area()->size();++i){
      DEBUG_MSG("Excluded shape " << i);
      // cast the excluded shape to a polygon and throw is unsuccessful
      Teuchos::RCP<DICe::Polygon> excluded_polygon =
          Teuchos::rcp_dynamic_cast<DICe::Polygon>((*map_it->second.excluded_area())[i]);
      TEUCHOS_TEST_FOR_EXCEPTION(excluded_polygon==Teuchos::null,std::runtime_error,"Error, failed cast to polygon.");
      num_excluded_vertices[i] = excluded_polygon->num_vertices();
      DEBUG_MSG("Number of vertices in the excluded polygon: " << num_excluded_vertices[i]);
      int_t start_index = pts_x.size();
      //std::cout << "start index " << start_index << std::endl;

      for(int_t p=0;p<num_excluded_vertices[i];++p){
        if(p==num_excluded_vertices[i]-1){
          segments_left.push_back(start_index + p + 1);
          segments_right.push_back(start_index + 1);
          //std::cout << " adding segment " << segments_left[segments_left.size()-1] << " "  << segments_right[segments_right.size()-1] << std::endl;
        }
        else{
          segments_left.push_back(start_index + p + 1);
          segments_right.push_back(start_index + p + 2);
          //std::cout << " adding segment " << segments_left[segments_left.size()-1] << " "  << segments_right[segments_right.size()-1] << std::endl;
        }
        pts_x.push_back((*excluded_polygon->vertex_coordinates_x())[p]);
        pts_y.push_back((*excluded_polygon->vertex_coordinates_y())[p]);
      }
      // find a point internal to the excluded hole
      // get the extents of the polygon
      std::set<std::pair<int_t,int_t> > owned_pixels = excluded_polygon->get_owned_pixels();
      const int_t min_x = excluded_polygon->min_x();
      const int_t max_x = excluded_polygon->max_x();
      const int_t min_y = excluded_polygon->min_y();
      const int_t max_y = excluded_polygon->max_y();
      TEUCHOS_TEST_FOR_EXCEPTION(max_x-min_x < 4,std::runtime_error,"Error, excluded regions is too small");
      TEUCHOS_TEST_FOR_EXCEPTION(max_y-min_y < 4,std::runtime_error,"Error, excluded regions is too small");
      const int_t pt_y = (max_y - min_y)/2 + min_y;
      int_t pt_x = min_x;
      int_t valid_pixels = 0;
      for(int_t x=min_x;x<=max_x;++x){
        if(owned_pixels.find(std::pair<int_t,int_t>(pt_y,x))!=owned_pixels.end()){
          if(valid_pixels>1){ // prevent the pixel along the shape edge from being selected
            pt_x = x;
            break;
          }
          valid_pixels++;
        }
      }
      holes_x[i] = pt_x;
      holes_y[i] = pt_y;
      DEBUG_MSG("Interior point for the hole " << pt_x << " " << pt_y);
    }
  }
  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_left(&segments_left[0],0,segments_left.size(),false);
  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_right(&segments_right[0],0,segments_left.size(),false);

  // gather the boundary segments
  const int_t num_boundary_segments = subset_file_info->boundary_condition_defs->size();
  DEBUG_MSG("Number of dirichlet boundary segments: " << num_boundary_segments);
  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_left(num_boundary_segments);
  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_right(num_boundary_segments);
  for(int_t i=0;i<num_boundary_segments;++i){
    dirichlet_boundary_segments_left[i] = (*subset_file_info->boundary_condition_defs)[i].left_vertex_id_+1; // +1 because ids are 1-based in Triangle
    dirichlet_boundary_segments_right[i] = (*subset_file_info->boundary_condition_defs)[i].right_vertex_id_+1;
  }
  Teuchos::ArrayRCP<scalar_t> points_x(&pts_x[0],0,pts_x.size(),false);
  Teuchos::ArrayRCP<scalar_t> points_y(&pts_y[0],0,pts_y.size(),false);
  Teuchos::RCP<DICe::mesh::Mesh> mesh = generate_tri6_mesh(
    points_x,
    points_y,
    holes_x,
    holes_y,
    dirichlet_boundary_segments_left,
    dirichlet_boundary_segments_right,
    neumann_boundary_segments_left,
    neumann_boundary_segments_right,
    max_size_constraint,
    output_file_name);
  return mesh;
}

Teuchos::RCP<DICe::mesh::Mesh> generate_tri6_mesh(Teuchos::ArrayRCP<scalar_t> points_x,
  Teuchos::ArrayRCP<scalar_t> points_y,
  Teuchos::ArrayRCP<scalar_t> holes_x,
  Teuchos::ArrayRCP<scalar_t> holes_y,
  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_left,
  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_right,
  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_left,
  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_right,
  const scalar_t & max_size_constraint,
  const std::string & output_file_name){

  TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()<=0||points_y.size()<=0,std::runtime_error,"Error, points size must be greater than 0");
  TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()!=points_y.size(),std::runtime_error,"Error, points must be the same size");
  TEUCHOS_TEST_FOR_EXCEPTION(output_file_name=="",std::runtime_error,"Error, ivalid output file name");

  const bool has_holes = holes_x!=Teuchos::null;
  int_t num_holes = 0;
  if(has_holes){
    TEUCHOS_TEST_FOR_EXCEPTION(holes_x.size()!=holes_y.size(),std::runtime_error,"Error, holes arrays must be the same size");
    num_holes = holes_x.size();
  }

  // set up the traingle data structures

  struct triangulateio in, out;

  // deal with the vertices

  in.numberofpoints = points_x.size();
  in.numberofpointattributes = 0;
  in.pointmarkerlist = (int *) NULL;
  in.pointlist = new REAL[in.numberofpoints*2];
  for(int_t i=0;i<points_x.size();++i){
    in.pointlist[i*2+0] = points_x[i];
    in.pointlist[i*2+1] = points_y[i];
  }

  // deal with segments (defines the boundary edges)

  const bool has_dirich_segments = dirichlet_boundary_segments_left!=Teuchos::null;
  const bool has_neumann_segments = neumann_boundary_segments_left!=Teuchos::null;
  int_t num_dirich = 4; // the default is the segments between the first four sets of coords
  int_t num_neum = 0;
  if(has_dirich_segments){
    TEUCHOS_TEST_FOR_EXCEPTION(dirichlet_boundary_segments_left.size()!=dirichlet_boundary_segments_left.size(),
      std::runtime_error,"Error, dirichlet segments arrays are not the same size.");
    num_dirich = dirichlet_boundary_segments_left.size();
  }
  if(has_neumann_segments){
    TEUCHOS_TEST_FOR_EXCEPTION(neumann_boundary_segments_left.size()!=neumann_boundary_segments_left.size(),
      std::runtime_error,"Error, dirichlet segments arrays are not the same size.");
    num_neum = neumann_boundary_segments_left.size();
  }
  const int_t num_segments = num_dirich + num_neum;
  DEBUG_MSG("generate_tri6_mesh(): number of dirichlet segments: " << num_dirich <<
    " number of neumann segments: " << num_neum << " total: " << num_segments);

  int_t seg_offset = 0;
  in.numberofsegments = num_segments;
  in.segmentlist = new int[in.numberofsegments*2];
  in.segmentmarkerlist = new int[in.numberofsegments];
  if(!has_dirich_segments&&!has_neumann_segments){
    DEBUG_MSG("generate_tri6_mesh(): no input boundary conditions given, using default dirichlet around outer edge.");
    TEUCHOS_TEST_FOR_EXCEPTION(num_segments!=4,std::runtime_error,"Error, wrong number of segments.");
    TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()<4,std::runtime_error,"Error, too few points.");
      in.segmentlist[0] = 1;in.segmentlist[1] = 2;
      in.segmentlist[2] = 2;in.segmentlist[3] = 3;
      in.segmentlist[4] = 3;in.segmentlist[5] = 4;
      in.segmentlist[6] = 4;in.segmentlist[7] = 1;
      in.segmentmarkerlist[0] = 2; // hard coded id for dirichlet bcs
      in.segmentmarkerlist[1] = 2; // hard coded id for dirichlet bcs
      in.segmentmarkerlist[2] = 2; // hard coded id for dirichlet bcs
      in.segmentmarkerlist[3] = 2; // hard coded id for dirichlet bcs
  }
  else{
    for(int_t i=0;i<num_dirich;++i){
      in.segmentlist[i*2+0] = dirichlet_boundary_segments_left[i];
      in.segmentlist[i*2+1] = dirichlet_boundary_segments_right[i];
      in.segmentmarkerlist[i] = 2; // hard coded id for dirichlet bcs
      seg_offset++;
    }
    for(int_t i=0;i<num_neum;++i){
      in.segmentlist[(i+seg_offset)*2+0] = neumann_boundary_segments_left[i];
      in.segmentlist[(i+seg_offset)*2+1] = neumann_boundary_segments_right[i];
      in.segmentmarkerlist[i+seg_offset] = 3; // hard coded id for dirichlet bcs
    }
  }

  // deal with holes

  in.numberofholes = num_holes;
  DEBUG_MSG("generate_tri6_mesh(): number of holes: " << num_holes);
  in.holelist = new REAL[in.numberofholes*2];
  for(int_t i=0;i<num_holes;++i){
    in.holelist[i*2+0] = holes_x[i];
    in.holelist[i*2+1] = holes_y[i];
  }
  in.numberofregions = 0;

  // output containers
  out.pointlist = (REAL *) NULL;
  out.trianglelist = (int *) NULL;
  out.triangleattributelist = (REAL *) NULL;
  out.pointmarkerlist = (int *) NULL;
  out.segmentlist = (int *) NULL;
  out.segmentmarkerlist = (int *) NULL;
  out.edgelist = (int *) NULL;
  out.edgemarkerlist = (int *) NULL;

  // use the triangle library to generate the triangle elements

  // using the o2 flag here to get quadratic tris with 6 nodes each
  std::stringstream arg_ss;
  arg_ss << "qpcAeo2a" << max_size_constraint;
  char args[1024];
  strncpy(args, arg_ss.str().c_str(), sizeof(args));
  args[sizeof(args) - 1] = 0;
  DEBUG_MSG("generate_tri6_mesh() called with args: " << args);
//  char args[] = arg_ss.str().c_str();
  triangulate(args,&in,&out,NULL);

  DEBUG_MSG("generate_tri6_mesh(): number of boundary segments: " << out.numberofsegments);
  std::set<int_t> dirichlet_boundary_nodes;
  std::set<int_t> neumann_boundary_nodes;
  std::set<int_t> lagrange_boundary_nodes;
  //std::cout << "segments: " << std::endl;
  for(int_t i=0;i<out.numberofsegments;++i){
    //std::cout << i << " left index " << out.segmentlist[i*2+0] << " right " << out.segmentlist[i*2+1] << " marker " << out.segmentmarkerlist[i] << std::endl;
    if(out.segmentmarkerlist[i]==2){ // 2 is the hard-coded id for dirichlet
      dirichlet_boundary_nodes.insert(out.segmentlist[i*2+0]);
      dirichlet_boundary_nodes.insert(out.segmentlist[i*2+1]);
      lagrange_boundary_nodes.insert(out.segmentlist[i*2+0]);
      lagrange_boundary_nodes.insert(out.segmentlist[i*2+0]);
    }
    if(out.segmentmarkerlist[i]==3){ // 3 is the hard-coded id for neumann
      neumann_boundary_nodes.insert(out.segmentlist[i*2+0]);
      neumann_boundary_nodes.insert(out.segmentlist[i*2+1]);
    }
  }
  //std::cout << "edges: " << std::endl;
  //for(int_t i=0;i<out.numberofedges;++i){
  //  std::cout << i << " left index " << out.edgelist[i*2+0] << " right " << out.edgelist[i*2+1] << std::endl;
  //}

  // convert the resulting mesh to an exodus mesh

  Teuchos::ArrayRCP<int_t> connectivity(out.numberoftriangles*out.numberofcorners); // numberofcorners is num nodes per elem
  for(int_t i=0;i<out.numberoftriangles;++i){
    for (int_t j = 0; j < out.numberofcorners; j++) {
      connectivity[i*out.numberofcorners + j] = out.trianglelist[i * out.numberofcorners + j];
      // search the connectivity for edge elements and side sets
      //if(has_dirich_segments||has_neumann_segments){
        for(int_t k=0;k<out.numberofsegments;++k){
          if(out.segmentlist[k*2] == out.trianglelist[i*out.numberofcorners + j]){
            for(int_t m=0;m<out.numberofcorners;++m){
              if(out.segmentlist[k*2+1] == out.trianglelist[i*out.numberofcorners +m]){
                // now that the element and side are known for boundary nodes add the middle node
                int_t third_node_pos = -1;
                if((j==0&&m==1)||(j==1&&m==0)){
                  third_node_pos = 5;
                }
                else if((j==1&&m==2)||(j==2&&m==1)){
                  third_node_pos = 3;
                }
                else if((j==0&&m==2)||(j==2&&m==0)){
                  third_node_pos = 4;
                }
                else{
                  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid segment found");
                }
                if(out.segmentmarkerlist[k]==2){
                  dirichlet_boundary_nodes.insert(out.trianglelist[i*out.numberofcorners + third_node_pos]);
                  // mid nodes do not get added the lagrange set
                }
                if(out.segmentmarkerlist[k]==3){
                  neumann_boundary_nodes.insert(out.trianglelist[i*out.numberofcorners + third_node_pos]);
                }
              }
            } // right end of segment
          } // left end of segment
        }
      //}
    }
  }
  DEBUG_MSG("generate_tri6_mesh(): number of triangle edges: " << out.numberofedges);
  DEBUG_MSG("generate_tri6_mesh(): number of dirichlet nodes: " << dirichlet_boundary_nodes.size());
  DEBUG_MSG("generate_tri6_mesh(): number of lagrange nodes: " << lagrange_boundary_nodes.size());
  DEBUG_MSG("generate_tri6_mesh(): number of neumann nodes: " << neumann_boundary_nodes.size());

  DEBUG_MSG("dirichlet boundary nodes");
  std::set<int_t>::const_iterator it = dirichlet_boundary_nodes.begin();
  std::set<int_t>::const_iterator it_end = dirichlet_boundary_nodes.end();
  for(;it!=it_end;++it){
    DEBUG_MSG(*it);
  }
  DEBUG_MSG("neumann boundary nodes");
  it = neumann_boundary_nodes.begin();
  it_end = neumann_boundary_nodes.end();
  for(;it!=it_end;++it){
    DEBUG_MSG(*it);
  }
  DEBUG_MSG("lagrange boundary nodes");
  it = lagrange_boundary_nodes.begin();
  it_end = lagrange_boundary_nodes.end();
  for(;it!=it_end;++it){
    DEBUG_MSG(*it);
  }
  Teuchos::ArrayRCP<scalar_t> node_coords_x(out.numberofpoints); // numberofcorners is num nodes per elem
  Teuchos::ArrayRCP<scalar_t> node_coords_y(out.numberofpoints); // numberofcorners is num nodes per elem
  for(int_t i=0;i<out.numberofpoints;++i){
    node_coords_x[i] = out.pointlist[i*2+0];
    node_coords_y[i] = out.pointlist[i*2+1];
  }
  Teuchos::RCP<DICe::mesh::Mesh> mesh =
      DICe::mesh::create_tri6_exodus_mesh(node_coords_x,
        node_coords_y,connectivity,dirichlet_boundary_nodes,
        neumann_boundary_nodes,lagrange_boundary_nodes,output_file_name);

  delete[] in.pointlist;
  delete[] in.holelist;
  delete[] in.segmentlist;
  delete[] in.segmentmarkerlist;
  delete[] out.pointlist;
  delete[] out.pointmarkerlist;
  delete[] out.trianglelist;
  delete[] out.triangleattributelist;
  delete[] out.segmentlist;
  delete[] out.segmentmarkerlist;
  delete[] out.edgelist;
  delete[] out.edgemarkerlist;

  return mesh;
}


}// End DICe Namespace
