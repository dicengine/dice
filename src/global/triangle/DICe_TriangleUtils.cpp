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
#include <DICe_Mesh.h>
#include <DICe_Parser.h>

#include <triangle.h>

namespace DICe {

DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::mesh::Mesh> generate_tri_mesh(const DICe::mesh::Base_Element_Type elem_type,
  Teuchos::ArrayRCP<scalar_t> points_x,
  Teuchos::ArrayRCP<scalar_t> points_y,
  const scalar_t & max_size_constraint,
  const std::string & output_file_name,
  const bool enforce_lagrange_bc,
  const bool use_regular_grid){
  return generate_tri_mesh(elem_type,
    points_x,
    points_y,
    Teuchos::null,
    Teuchos::null,
    Teuchos::null,
    Teuchos::null,
    Teuchos::null,
    Teuchos::null,
    max_size_constraint,
    output_file_name,
    enforce_lagrange_bc,
    use_regular_grid);
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::mesh::Mesh> generate_tri_mesh(const DICe::mesh::Base_Element_Type elem_type,
  const std::string & roi_file_name,
  const scalar_t & max_size_constraint,
  const std::string & output_file_name){

  TEUCHOS_TEST_FOR_EXCEPTION(elem_type!=DICe::mesh::TRI6&&elem_type!=DICe::mesh::TRI3,std::runtime_error,
    "Error, invalid element type");
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
  Teuchos::RCP<DICe::mesh::Mesh> mesh = Teuchos::null;
  std::vector<Boundary_Condition_Def> stored_bcs;
  if(subset_file_info->use_regular_grid){
    DEBUG_MSG("generate_tri_mesh(): creating a regular grid tri mesh (not Delaunay)");
    TEUCHOS_TEST_FOR_EXCEPTION(num_boundary_vertices!=4,std::runtime_error,
      "Error, for regular grid only 4 boundary vertices are allowed.");
    TEUCHOS_TEST_FOR_EXCEPTION(map_it->second.has_excluded_area(),std::runtime_error,
      "Error, for regular grid excluded areas are not allowed.");
    TEUCHOS_TEST_FOR_EXCEPTION((*boundary_polygon->vertex_coordinates_x())[0]!=(*boundary_polygon->vertex_coordinates_x())[3],
      std::runtime_error,"Error, x0 must = x3 (must be a rectangular shape)");
    TEUCHOS_TEST_FOR_EXCEPTION((*boundary_polygon->vertex_coordinates_x())[1]!=(*boundary_polygon->vertex_coordinates_x())[2],
      std::runtime_error,"Error, x1 must = x2 (must be a rectangular shape)");
    TEUCHOS_TEST_FOR_EXCEPTION((*boundary_polygon->vertex_coordinates_y())[0]!=(*boundary_polygon->vertex_coordinates_y())[1],
      std::runtime_error,"Error, y0 must = y1 (must be a rectangular shape)");
    TEUCHOS_TEST_FOR_EXCEPTION((*boundary_polygon->vertex_coordinates_y())[2]!=(*boundary_polygon->vertex_coordinates_y())[3],
      std::runtime_error,"Error, y2 must = y3 (must be a rectangular shape)");
    const scalar_t begin_x = (*boundary_polygon->vertex_coordinates_x())[0]-0.5; // the offsets are so that the mesh edge aligns with a pixel edge
    const scalar_t end_x = (*boundary_polygon->vertex_coordinates_x())[1]+0.5;
    const scalar_t begin_y = (*boundary_polygon->vertex_coordinates_y())[0]-0.5;
    const scalar_t end_y = (*boundary_polygon->vertex_coordinates_y())[3]+0.5;
    std::vector<int_t> dirichlet_sides;
    std::vector<int_t> neumann_sides;
    const int_t num_boundary_segments = subset_file_info->boundary_condition_defs->size();
    TEUCHOS_TEST_FOR_EXCEPTION(num_boundary_segments>4,std::runtime_error,"Error invalid number of bcs");
    DEBUG_MSG("Number of dirichlet and neumann boundary segments: " << num_boundary_segments);
    for(int_t i=0;i<num_boundary_segments;++i){
      const int_t id_left = (*subset_file_info->boundary_condition_defs)[i].left_vertex_id_;
      const int_t id_right = (*subset_file_info->boundary_condition_defs)[i].right_vertex_id_;
      // convert shape vertex ids to sides of the ROI
      int_t side = -1;
      if((id_left==0&&id_right==1)||(id_left==1&&id_right==0))
        side = 0;
      else if ((id_left==1&&id_right==2)||(id_left==2&&id_right==1))
        side = 1;
      else if ((id_left==2&&id_right==3)||(id_left==3&&id_right==2))
        side = 2;
      else if ((id_left==0&&id_right==3)||(id_left==3&&id_right==0))
        side = 3;
      TEUCHOS_TEST_FOR_EXCEPTION(side==-1,std::runtime_error,"Error, invalid side");
      // neumann bcs
      if((*subset_file_info->boundary_condition_defs)[i].is_neumann_){
        neumann_sides.push_back(side);
      }
      // dirchlet bcs
      else{
        //DEBUG_MSG("generate_tri_mesh(): regular_grid storing bc id " << i);
        stored_bcs.push_back((*subset_file_info->boundary_condition_defs)[i]);
        dirichlet_sides.push_back(side);
      }
    }
    mesh = generate_regular_tri_mesh(elem_type,
      begin_x,
      end_x,
      begin_y,
      end_y,
      max_size_constraint,
      dirichlet_sides,
      neumann_sides,
      output_file_name,
      subset_file_info->enforce_lagrange_bc);
  }
  else{
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
        // cast the excluded shape to a polygon and throw if unsuccessful
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
        TEUCHOS_TEST_FOR_EXCEPTION(max_x-min_x < 4,std::runtime_error,"Error, excluded region is too small");
        TEUCHOS_TEST_FOR_EXCEPTION(max_y-min_y < 4,std::runtime_error,"Error, excluded region is too small");
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

    // add the user defined neumann bcs:
    const int_t num_boundary_segments = subset_file_info->boundary_condition_defs->size();
    DEBUG_MSG("Number of dirichlet and neumann boundary segments: " << num_boundary_segments);
    int_t num_user_defined_neumann = 0;
    for(int_t i=0;i<num_boundary_segments;++i){
      if((*subset_file_info->boundary_condition_defs)[i].is_neumann_){
        segments_left.push_back((*subset_file_info->boundary_condition_defs)[i].left_vertex_id_+1);
        segments_right.push_back((*subset_file_info->boundary_condition_defs)[i].right_vertex_id_+1);
        num_user_defined_neumann++;
      }
    }
    DEBUG_MSG("Number of user-defined Neumann segments: " << num_user_defined_neumann);
    Teuchos::ArrayRCP<int_t> neumann_boundary_segments_left(&segments_left[0],0,segments_left.size(),false);
    Teuchos::ArrayRCP<int_t> neumann_boundary_segments_right(&segments_right[0],0,segments_left.size(),false);

    // gather the boundary segments
    const int_t num_dirichlet_segments = num_boundary_segments - num_user_defined_neumann;
    DEBUG_MSG("Number of user-defined Dirichlet segments: " << num_dirichlet_segments);
    Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_left(num_dirichlet_segments);
    Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_right(num_dirichlet_segments);
    int_t seg_index = 0;
    for(int_t i=0;i<num_boundary_segments;++i){
      if((*subset_file_info->boundary_condition_defs)[i].is_neumann_) continue;
      stored_bcs.push_back((*subset_file_info->boundary_condition_defs)[i]);
      dirichlet_boundary_segments_left[seg_index] = (*subset_file_info->boundary_condition_defs)[i].left_vertex_id_+1; // +1 because ids are 1-based in Triangle
      dirichlet_boundary_segments_right[seg_index] = (*subset_file_info->boundary_condition_defs)[i].right_vertex_id_+1;
      seg_index++;
    }
    Teuchos::ArrayRCP<scalar_t> points_x(&pts_x[0],0,pts_x.size(),false);
    Teuchos::ArrayRCP<scalar_t> points_y(&pts_y[0],0,pts_y.size(),false);
    mesh = generate_tri_mesh(elem_type,
      points_x,
      points_y,
      holes_x,
      holes_y,
      dirichlet_boundary_segments_left,
      dirichlet_boundary_segments_right,
      neumann_boundary_segments_left,
      neumann_boundary_segments_right,
      max_size_constraint,
      output_file_name,
      subset_file_info->enforce_lagrange_bc,
      false);
  }
  mesh->set_bc_defs(stored_bcs);
  mesh->set_ic_values(subset_file_info->ic_value_x,subset_file_info->ic_value_y);
  return mesh;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::mesh::Mesh> generate_tri_mesh(const DICe::mesh::Base_Element_Type elem_type,
  Teuchos::ArrayRCP<scalar_t> points_x,
  Teuchos::ArrayRCP<scalar_t> points_y,
  Teuchos::ArrayRCP<scalar_t> holes_x,
  Teuchos::ArrayRCP<scalar_t> holes_y,
  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_left,
  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_right,
  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_left,
  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_right,
  const scalar_t & max_size_constraint,
  const std::string & output_file_name,
  const bool enforce_lagrange_bc,
  const bool use_regular_grid){

  TEUCHOS_TEST_FOR_EXCEPTION(elem_type!=DICe::mesh::TRI6&&elem_type!=DICe::mesh::TRI3,std::runtime_error,
    "Error, invalid elem type");
  TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()<=0||points_y.size()<=0,std::runtime_error,"Error, points size must be greater than 0");
  TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()!=points_y.size(),std::runtime_error,"Error, points must be the same size");
  TEUCHOS_TEST_FOR_EXCEPTION(output_file_name=="",std::runtime_error,"Error, ivalid output file name");

  const bool has_holes = holes_x!=Teuchos::null;
  int_t num_holes = 0;
  if(has_holes){
    TEUCHOS_TEST_FOR_EXCEPTION(holes_x.size()!=holes_y.size(),std::runtime_error,"Error, holes arrays must be the same size");
    num_holes = holes_x.size();
  }
  Teuchos::RCP<DICe::mesh::Mesh> mesh;

  if(use_regular_grid){
    TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()!=4,std::runtime_error,"Points must be size four for regular grid");
    TEUCHOS_TEST_FOR_EXCEPTION(points_y.size()!=4,std::runtime_error,"Points must be size four for regular grid");
    // FIXME relax all Dirichlet BCs later
    std::vector<int_t> dirichlet_sides(4);
    for(int_t i=0;i<4;++i)
      dirichlet_sides[i] = i;
    std::vector<int_t> neumann_sides(4,0);
    mesh = generate_regular_tri_mesh(elem_type,
      points_x[0],
      points_x[1],
      points_y[0],
      points_y[2],
      max_size_constraint,
      dirichlet_sides,
      neumann_sides,
      output_file_name,
      enforce_lagrange_bc);
    std::vector<Boundary_Condition_Def> stored_bcs;
    // create the bc defs
    Boundary_Condition_Def bc_def_1;
    bc_def_1.has_value_ = true;
    bc_def_1.comp_ = 1;
    Boundary_Condition_Def bc_def_2;
    bc_def_2.has_value_ = true;
    bc_def_2.comp_ = 0;
    Boundary_Condition_Def bc_def_3;
    bc_def_3.has_value_ = true;
    bc_def_3.comp_ = 1;
    Boundary_Condition_Def bc_def_4;
    bc_def_4.has_value_ = true;
    bc_def_4.comp_ = 0;
    stored_bcs.push_back(bc_def_1);
    stored_bcs.push_back(bc_def_2);
    stored_bcs.push_back(bc_def_3);
    stored_bcs.push_back(bc_def_4);
    mesh->set_bc_defs(stored_bcs);
  }
  else{
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
    int_t num_dirich = 0; // the default is the segments between the first four sets of coords
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
    DEBUG_MSG("generate_tri_mesh(): number of dirichlet segments: " << num_dirich <<
      " number of neumann segments: " << num_neum << " total: " << num_segments);

    int_t seg_offset = 0;
    std::vector<Boundary_Condition_Def> stored_bcs;
    if(!has_dirich_segments&&!has_neumann_segments){
      in.numberofsegments = 4;
      in.segmentlist = new int[in.numberofsegments*2];
      in.segmentmarkerlist = new int[in.numberofsegments];
      DEBUG_MSG("generate_tri_mesh(): no input boundary conditions given, using default dirichlet around outer edge.");
      //TEUCHOS_TEST_FOR_EXCEPTION(num_segments!=4,std::runtime_error,"Error, wrong number of segments.");
      TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()<4,std::runtime_error,"Error, too few points.");
      in.segmentlist[0] = 1;in.segmentlist[1] = 2;
      in.segmentlist[2] = 2;in.segmentlist[3] = 3;
      in.segmentlist[4] = 3;in.segmentlist[5] = 4;
      in.segmentlist[6] = 4;in.segmentlist[7] = 1;
      in.segmentmarkerlist[0] = 1; // hard coded id for dirichlet bcs
      in.segmentmarkerlist[1] = 2; // hard coded id for dirichlet bcs
      in.segmentmarkerlist[2] = 3; // hard coded id for dirichlet bcs
      in.segmentmarkerlist[3] = 4; // hard coded id for dirichlet bcs

      // create the bc defs
      Boundary_Condition_Def bc_def_1;
      bc_def_1.has_value_ = true;
      bc_def_1.comp_ = 1;
      Boundary_Condition_Def bc_def_2;
      bc_def_2.has_value_ = true;
      bc_def_2.comp_ = 0;
      Boundary_Condition_Def bc_def_3;
      bc_def_3.has_value_ = true;
      bc_def_3.comp_ = 1;
      Boundary_Condition_Def bc_def_4;
      bc_def_4.has_value_ = true;
      bc_def_4.comp_ = 0;
      stored_bcs.push_back(bc_def_1);
      stored_bcs.push_back(bc_def_2);
      stored_bcs.push_back(bc_def_3);
      stored_bcs.push_back(bc_def_4);
    }
    else{
      in.numberofsegments = num_segments;
      in.segmentlist = new int[in.numberofsegments*2];
      in.segmentmarkerlist = new int[in.numberofsegments];
      for(int_t i=0;i<num_dirich;++i){
        in.segmentlist[i*2+0] = dirichlet_boundary_segments_left[i];
        in.segmentlist[i*2+1] = dirichlet_boundary_segments_right[i];
        in.segmentmarkerlist[i] = i+1;
        seg_offset++;
      }
      for(int_t i=0;i<num_neum;++i){
        in.segmentlist[(i+seg_offset)*2+0] = neumann_boundary_segments_left[i];
        in.segmentlist[(i+seg_offset)*2+1] = neumann_boundary_segments_right[i];
        in.segmentmarkerlist[i+seg_offset] = -10000; // hard coded -10000 for neumann bcs
      }
    }

    // deal with holes

    in.numberofholes = num_holes;
    DEBUG_MSG("generate_tri_mesh(): number of holes: " << num_holes);
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
    if(elem_type==DICe::mesh::TRI6)
      arg_ss << "qpcDAeo2a" << max_size_constraint;
    else
      arg_ss << "qpcDAea" << max_size_constraint;
    char args[1024];
    strncpy(args, arg_ss.str().c_str(), sizeof(args));
    args[sizeof(args) - 1] = 0;
    DEBUG_MSG("generate_tri_mesh() called with args: " << args);
    //  char args[] = arg_ss.str().c_str();
    triangulate(args,&in,&out,NULL);

    DEBUG_MSG("generate_tri_mesh(): number of boundary segments: " << out.numberofsegments);
    std::vector<std::pair<int_t,int_t>> dirichlet_boundary_nodes;
    std::set<int_t> neumann_boundary_nodes;
    std::set<int_t> lagrange_boundary_nodes;
    if(enforce_lagrange_bc){
      lagrange_boundary_nodes.insert(1);
      //lagrange_boundary_nodes.insert(out.segmentlist[i*2+0]);
    }
    //std::cout << "segments: " << std::endl;
    for(int_t i=0;i<out.numberofsegments;++i){
      //std::cout << i << " left index " << out.segmentlist[i*2+0] << " right " << out.segmentlist[i*2+1] << " marker " << out.segmentmarkerlist[i] << std::endl;
      if(out.segmentmarkerlist[i]>0){ // positive is for dirichlet
        dirichlet_boundary_nodes.push_back(std::pair<int_t,int_t>(out.segmentlist[i*2+0],out.segmentmarkerlist[i]));
        dirichlet_boundary_nodes.push_back(std::pair<int_t,int_t>(out.segmentlist[i*2+1],out.segmentmarkerlist[i]));
        //      if(enforce_lagrange_bc){
        //        lagrange_boundary_nodes.insert(out.segmentlist[i*2+0]);
        //      }
      }
      if(out.segmentmarkerlist[i]==-10000){ // -10000 is for neumann
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
        if(elem_type==DICe::mesh::TRI6){
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
                  if(out.segmentmarkerlist[k]>0){
                    dirichlet_boundary_nodes.push_back(std::pair<int_t,int_t>(out.trianglelist[i*out.numberofcorners + third_node_pos],out.segmentmarkerlist[k]));
                    // mid nodes do not get added the lagrange set
                  }
                  if(out.segmentmarkerlist[k]==-10000){ // -10000 is the neumann hard coded bc id
                    neumann_boundary_nodes.insert(out.trianglelist[i*out.numberofcorners + third_node_pos]);
                  }
                }
              } // right end of segment
            } // left end of segment
          }
        } // end TRI6
      }
    }
    DEBUG_MSG("generate_tri_mesh(): number of triangle edges: " << out.numberofedges);
    DEBUG_MSG("generate_tri_mesh(): number of dirichlet nodes: " << dirichlet_boundary_nodes.size());
    DEBUG_MSG("generate_tri_mesh(): number of lagrange nodes: " << lagrange_boundary_nodes.size());
    DEBUG_MSG("generate_tri_mesh(): number of neumann nodes: " << neumann_boundary_nodes.size());

    DEBUG_MSG("dirichlet boundary nodes");
    //std::map<int_t,int_t>::const_iterator mit = dirichlet_boundary_nodes.begin();
    //std::map<int_t,int_t>::const_iterator mit_end = dirichlet_boundary_nodes.end();
    for(size_t i=0;i<dirichlet_boundary_nodes.size();++i){
      DEBUG_MSG("node " << dirichlet_boundary_nodes[i].first << " bc id " << dirichlet_boundary_nodes[i].second);
    }
    DEBUG_MSG("neumann boundary nodes");
    std::set<int_t>::const_iterator it = neumann_boundary_nodes.begin();
    std::set<int_t>::const_iterator it_end = neumann_boundary_nodes.end();
    for(;it!=it_end;++it){
      DEBUG_MSG("node " << *it << " bc id " << -10000);
    }
    DEBUG_MSG("lagrange boundary nodes");
    it = lagrange_boundary_nodes.begin();
    it_end = lagrange_boundary_nodes.end();
    for(;it!=it_end;++it){
      DEBUG_MSG("node " << *it << " bc id " << -1);
    }
    Teuchos::ArrayRCP<scalar_t> node_coords_x(out.numberofpoints); // numberofcorners is num nodes per elem
    Teuchos::ArrayRCP<scalar_t> node_coords_y(out.numberofpoints); // numberofcorners is num nodes per elem
    Teuchos::ArrayRCP<int_t> node_map(out.numberofpoints,0);
    for(int_t i=0;i<out.numberofpoints;++i){
      node_coords_x[i] = out.pointlist[i*2+0];
      node_coords_y[i] = out.pointlist[i*2+1];
      node_map[i] = i + 1;
    }
    Teuchos::ArrayRCP<int_t> elem_map(out.numberoftriangles,0);
    for(int_t i=0;i<out.numberoftriangles;++i)
      elem_map[i] = i + 1;
    mesh = DICe::mesh::create_point_or_tri_mesh(elem_type,node_coords_x,
      node_coords_y,connectivity,node_map,elem_map,dirichlet_boundary_nodes,
      neumann_boundary_nodes,lagrange_boundary_nodes,output_file_name);

    if(!has_dirich_segments&&!has_neumann_segments){
      mesh->set_bc_defs(stored_bcs);
    }

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
  }
  return mesh;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::mesh::Mesh> generate_regular_tri_mesh(const DICe::mesh::Base_Element_Type elem_type,
  const scalar_t & begin_x,
  const scalar_t & end_x,
  const scalar_t & begin_y,
  const scalar_t & end_y,
  const scalar_t & h,
  std::vector<int_t> & dirichlet_sides,
  std::vector<int_t> & neumann_sides,
  const std::string & output_file_name,
  const bool enforce_lagrange_bc){

  TEUCHOS_TEST_FOR_EXCEPTION(elem_type!=DICe::mesh::TRI6&&elem_type!=DICe::mesh::TRI3,std::runtime_error,
    "Error, invalid elem type");
  const scalar_t width = end_x - begin_x;
  TEUCHOS_TEST_FOR_EXCEPTION(width<=0,std::runtime_error,"Error, invalid width");
  const scalar_t height = end_y - begin_y;
  TEUCHOS_TEST_FOR_EXCEPTION(height<=0,std::runtime_error,"Error, invalid height");
  TEUCHOS_TEST_FOR_EXCEPTION(h>width||h>height,std::runtime_error,"Error, invalid mesh size");

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

  DEBUG_MSG("generate_regular_tri_mesh(): x ticks");
  for(int_t i=0;i<num_nodes_x;++i){
    DEBUG_MSG("x_coord " << x_ticks[i]);
  }
  DEBUG_MSG("generate_regular_tri_mesh(): y ticks");
  for(int_t i=0;i<num_nodes_y;++i){
    DEBUG_MSG("y_coord " << y_ticks[i]);
  }
  const int_t num_corner_nodes = num_nodes_x * num_nodes_y;

  // set up the corner nodes
  std::vector<scalar_t> x_coords;
  std::vector<scalar_t> y_coords;
  for(int_t j=0;j<num_nodes_y;++j){
    for(int_t i=0;i<num_nodes_x;++i){
      x_coords.push_back(x_ticks[i]);
      y_coords.push_back(y_ticks[j]);
    }
  }
  if(elem_type==DICe::mesh::TRI6){
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
  }
  const int_t num_nodes = x_coords.size();
  TEUCHOS_TEST_FOR_EXCEPTION(x_coords.size()!=y_coords.size(),std::runtime_error,"Error, coord vectors should be equal size");
  DEBUG_MSG("generate_regular_tri_mesh(): num_corner_nodes " << num_corner_nodes);
  DEBUG_MSG("generate_regular_tri_mesh(): num_nodes " << num_nodes);

//  for(int_t i=0;i<num_nodes;++i){
//    std::cout << "node " << i << " x " << x_coords[i] << " y " << y_coords[i] << std::endl;
//  }
  Teuchos::ArrayRCP<scalar_t> node_coords_x(&x_coords[0],0,x_coords.size(),false);
  Teuchos::ArrayRCP<scalar_t> node_coords_y(&y_coords[0],0,y_coords.size(),false);
  const int_t num_col_x = num_nodes_x - 1;
  const int_t num_col_y = num_nodes_y - 1;
  const int_t num_elem = (num_nodes_x-1)*(num_nodes_y-1)*2;
  DEBUG_MSG("generate_regular_tri_mesh(): num elements: " << num_elem);

  const int_t num_nodes_per_elem = elem_type==DICe::mesh::TRI6 ? 6 : 3;

  Teuchos::ArrayRCP<int_t> connectivity(num_elem*num_nodes_per_elem,-1);
  const int_t num_diag_per_row = 2*num_nodes_x-1;
  const int_t num_mid_per_row = num_col_x;
  const int_t diag_offset = num_corner_nodes;
  const int_t mid_offset = num_corner_nodes + num_diag_per_row*num_col_y;
  //std::cout << " num_diag_per_row " << num_diag_per_row << " num_mid_per_row " << num_mid_per_row << " diag_offset " << diag_offset << " mid_offset " << mid_offset << std::endl;

  int_t swap_ids[] = {0,1,2,4,5,3};
  for(int_t j=0;j<num_col_y;++j){
    for(int_t i=0;i<num_col_x;++i){
      // lower row of tris
      const int_t lower_id = j*(num_col_x*2) + i;
      //std::cout << " lower id " << lower_id << std::endl;
      int_t lower_nodes[num_nodes_per_elem];
      lower_nodes[0] = j*num_nodes_x + i;
      lower_nodes[1] = j*num_nodes_x + i + 1;
      lower_nodes[2] = (j+1)*num_nodes_x + i;
      if(elem_type==DICe::mesh::TRI6){
        lower_nodes[3] = mid_offset + j*num_mid_per_row + i;
        lower_nodes[4] = diag_offset + j*num_diag_per_row + (i*2) + 1;
        lower_nodes[5] = diag_offset + j*num_diag_per_row + (i*2);
      }
      for(int_t n=0;n<num_nodes_per_elem;++n)
        connectivity[lower_id*num_nodes_per_elem+n] = lower_nodes[swap_ids[n]]+1;
//      std::cout << " conn: ";
//      for(int_t n=0;n<6;++n)
//        std::cout << " " << connectivity[lower_id*6+n];
//      std::cout << std::endl;
      // upper row of tris
      const int_t upper_id = j*(num_col_x*2) + num_col_x + i;
      //std::cout << " upper id " << upper_id << std::endl;
      int_t upper_nodes[num_nodes_per_elem];
      upper_nodes[2] = j*num_nodes_x + i + 1;
      upper_nodes[0] = (j+1)*num_nodes_x + i + 1;
      upper_nodes[1] = (j+1)*num_nodes_x + i;
      if(elem_type==DICe::mesh::TRI6){
        upper_nodes[5] = diag_offset + j*num_diag_per_row + (i+1)*2;
        upper_nodes[3] = mid_offset + (j+1)*num_mid_per_row + i;
        upper_nodes[4] = diag_offset + j*num_diag_per_row + (i*2) + 1;
      }
      for(int_t n=0;n<num_nodes_per_elem;++n)
        connectivity[upper_id*num_nodes_per_elem+n] = upper_nodes[swap_ids[n]]+1;
//      std::cout << " conn: ";
//      for(int_t n=0;n<6;++n)
//        std::cout << " " << connectivity[upper_id*6+n];
//      std::cout << std::endl;
    }
  }
  std::vector<int_t> bottom_edge_nodes;
  std::vector<int_t> bottom_lag_nodes;
  for(int_t j=0;j<num_nodes_x;++j){
    bottom_edge_nodes.push_back(j+1);
    bottom_lag_nodes.push_back(j+1);
  }
  if(elem_type==DICe::mesh::TRI6)
    for(int_t j=0;j<num_nodes_x-1;++j)
      bottom_edge_nodes.push_back(j+1+mid_offset);
  std::vector<int_t> left_edge_nodes;
  std::vector<int_t> left_lag_nodes;
  for(int_t j=0;j<num_nodes_y;++j){
    left_edge_nodes.push_back(j*num_nodes_x + 1);
    left_lag_nodes.push_back(j*num_nodes_x + 1);
  }
  if(elem_type==DICe::mesh::TRI6)
    for(int_t j=0;j<num_nodes_y-1;++j)
      left_edge_nodes.push_back(diag_offset + j*num_diag_per_row + 1);
  std::vector<int_t> right_edge_nodes;
  std::vector<int_t> right_lag_nodes;
  for(int_t j=0;j<num_nodes_y;++j){
    right_edge_nodes.push_back(j*num_nodes_x + num_nodes_x);
    right_lag_nodes.push_back(j*num_nodes_x + num_nodes_x);
  }
  if(elem_type==DICe::mesh::TRI6)
    for(int_t j=0;j<num_nodes_y-1;++j)
      right_edge_nodes.push_back(diag_offset + j*num_diag_per_row + num_diag_per_row);
  std::vector<int_t> top_edge_nodes;
  std::vector<int_t> top_lag_nodes;
  const int_t offset = (num_nodes_y-1)*num_nodes_x;
  for(int_t j=0;j<num_nodes_x;++j){
    top_edge_nodes.push_back(j+1+offset);
    top_lag_nodes.push_back(j+1+offset);
  }
  if(elem_type==DICe::mesh::TRI6)
    for(int_t j=0;j<num_nodes_x-1;++j)
      top_edge_nodes.push_back(mid_offset + (num_nodes_y-1)*num_mid_per_row + j + 1);
  //std::cout << "bottom edge " << std::endl;
  //for(size_t i=0;i<bottom_edge_nodes.size();++i){
  //  std::cout << bottom_edge_nodes[i] << std::endl;
  //}
  //std::cout << "right edge " << std::endl;
  //for(size_t i=0;i<right_edge_nodes.size();++i){
  //  std::cout << right_edge_nodes[i] << std::endl;
  //}
  //std::cout << "top edge " << std::endl;
  //for(size_t i=0;i<top_edge_nodes.size();++i){
  //  std::cout << top_edge_nodes[i] << std::endl;
  //}
  //std::cout << "left edge " << std::endl;
  //for(size_t i=0;i<left_edge_nodes.size();++i){
  //  std::cout << left_edge_nodes[i] << std::endl;
  //}

  std::vector<std::pair<int_t,int_t> > dirichlet_bcs;
  std::set<int_t> lag_bcs;
  if(enforce_lagrange_bc)
    lag_bcs.insert(1); // only one node gets a lag bc
  // set up the boundary conditions
  for(size_t i=0;i<dirichlet_sides.size();++i){
    // lower edge
    if(dirichlet_sides[i]==0){
      //dirichlet_bcs.insert(std::pair<int_t,std::vector<int_t> >(i+1,bottom_edge_nodes));
      for(size_t j=0;j<bottom_edge_nodes.size();++j)
        dirichlet_bcs.push_back(std::pair<int_t,int_t>(bottom_edge_nodes[j],i+1));
//      if(enforce_lagrange_bc){
//        for(size_t j=0;j<bottom_lag_nodes.size();++j)
//          lag_bcs.insert(bottom_lag_nodes[j]);
//      }
    }
    // right edge
    else if(dirichlet_sides[i]==1){
      //dirichlet_bcs.insert(std::pair<int_t,std::vector<int_t> >(i+1,right_edge_nodes));
      for(size_t j=0;j<right_edge_nodes.size();++j)
        dirichlet_bcs.push_back(std::pair<int_t,int_t>(right_edge_nodes[j],i+1));
//      if(enforce_lagrange_bc){
//        for(size_t j=0;j<right_lag_nodes.size();++j)
//          lag_bcs.insert(right_lag_nodes[j]);
//      }
    }
    // top edge
    else if(dirichlet_sides[i]==2){
      //dirichlet_bcs.insert(std::pair<int_t,std::vector<int_t> >(i+1,top_edge_nodes));
      for(size_t j=0;j<top_edge_nodes.size();++j)
        dirichlet_bcs.push_back(std::pair<int_t,int_t>(top_edge_nodes[j],i+1));
//      if(enforce_lagrange_bc){
//        for(size_t j=0;j<top_lag_nodes.size();++j)
//          lag_bcs.insert(top_lag_nodes[j]);
//      }
    }
    // left edge
    else if(dirichlet_sides[i]==3){
      //dirichlet_bcs.insert(std::pair<int_t,std::vector<int_t> >(i+1,left_edge_nodes));
      for(size_t j=0;j<left_edge_nodes.size();++j)
        dirichlet_bcs.push_back(std::pair<int_t,int_t>(left_edge_nodes[j],i+1));
//      if(enforce_lagrange_bc){
//        for(size_t j=0;j<left_lag_nodes.size();++j)
//          lag_bcs.insert(left_lag_nodes[j]);
//      }
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid side");
    }
  }
  std::set<int_t> neumann_bcs;
  // set up the boundary conditions
  for(size_t i=0;i<neumann_sides.size();++i){
    // lower edge
    if(neumann_sides[i]==0){
      for(size_t j=0;j<bottom_edge_nodes.size();++j)
        neumann_bcs.insert(bottom_edge_nodes[j]);
    }
    // right edge
    else if(neumann_sides[i]==1){
      for(size_t j=0;j<right_edge_nodes.size();++j)
        neumann_bcs.insert(right_edge_nodes[j]);
    }
    // top edge
    else if(neumann_sides[i]==2){
      for(size_t j=0;j<top_edge_nodes.size();++j)
        neumann_bcs.insert(top_edge_nodes[j]);
    }
    // left edge
    else if(neumann_sides[i]==3){
      for(size_t j=0;j<left_edge_nodes.size();++j)
        neumann_bcs.insert(left_edge_nodes[j]);
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid side");
    }
  }
  Teuchos::ArrayRCP<int_t> node_map(num_nodes,0);
  for(int_t i=0;i<num_nodes;++i)
    node_map[i] = i+1;

  Teuchos::ArrayRCP<int_t> elem_map(num_elem,0);
  for(int_t i=0;i<num_elem;++i)
    elem_map[i] = i + 1;


  Teuchos::RCP<DICe::mesh::Mesh> mesh =
      DICe::mesh::create_point_or_tri_mesh(elem_type,
        node_coords_x,
        node_coords_y,connectivity,node_map,elem_map,dirichlet_bcs,
        neumann_bcs,lag_bcs,output_file_name);

  mesh->set_is_regular_grid(true);

  return mesh;
}



}// End DICe Namespace
