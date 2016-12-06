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

#include <DICe_Mesh.h>

#include <Teuchos_Tuple.hpp>

namespace DICe {
namespace mesh {

Element::Element(const connectivity_vector & connectivity,
  const int_t global_id,
  const int_t local_id,
  const int_t block_id,
  const int_t block_local_id) :
      Mesh_Object(global_id,local_id),
      block_id_(block_id),
      block_local_id_(block_local_id),
      connectivity_(connectivity)
{}

Subelement::Subelement(Teuchos::RCP<Element> element,
  const int_t local_id,
  const int_t global_id) :
      Mesh_Object(global_id,local_id),
      block_id_(element->block_id()),
      connectivity_(element->get_connectivity()),
      parent_element_(element)
{}
Subelement::Subelement(const connectivity_vector & connectivity,
  Teuchos::RCP<Element> element,
  const int_t local_id,
  const int_t global_id,
  const int_t block_id) :
      Mesh_Object(global_id,local_id),
      block_id_(block_id),
      connectivity_(connectivity),
      parent_element_(element)
{}

Mesh_Object::Mesh_Object(const int_t global_id,
  const int_t local_id) :
  global_id_(global_id),
  local_id_(local_id),
  overlap_local_id_(-1),
  overlap_neighbor_local_id_(-1),
  initial_num_node_relations_(0),
  initial_num_elem_relations_(0)
{}

void
Mesh_Object::add_shallow_relation(const field_enums::Entity_Rank entity_rank,
  const int_t global_id) const
{
  // cast away the constness because the mesh object is part of a set
  const DICe::mesh::Mesh_Object & cobj = *this;
  DICe::mesh::Mesh_Object & obj = const_cast<DICe::mesh::Mesh_Object&>(cobj);

  shallow_relations_map::iterator it=obj.get_shallow_relations_map()->find(entity_rank);
  if(it==obj.get_shallow_relations_map()->end())
  {
    std::set<int_t> ord_set;
    ord_set.insert(global_id);
    obj.get_shallow_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::set<int_t> >(entity_rank,ord_set));
  }
  else
  {
    std::set<int_t> ord_set = it->second;
    ord_set.insert(global_id);
    obj.get_shallow_relations_map()->erase(it);
    obj.get_shallow_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::set<int_t> >(entity_rank,ord_set));
  }
}

// TODO condense all of these below
void
Mesh_Object::add_deep_relation(const Teuchos::RCP<Node> node) const
{
  // cast away the constness because the mesh object is part of a set
  const DICe::mesh::Mesh_Object & cobj = *this;
  DICe::mesh::Mesh_Object & obj = const_cast<DICe::mesh::Mesh_Object&>(cobj);

  deep_relations_map::iterator it=obj.get_deep_relations_map()->find(field_enums::NODE_RANK);
  if(it==obj.get_deep_relations_map()->end())
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set;
    ptr_set.push_back(node);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::NODE_RANK,ptr_set));
  }
  else
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set = it->second;
    ptr_set.push_back(node);
    obj.get_deep_relations_map()->erase(it);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::NODE_RANK,ptr_set));
  }
}

void
Mesh_Object::add_deep_relation(const Teuchos::RCP<Element> element) const
{
  // cast away the constness because the mesh object is part of a set
  const DICe::mesh::Mesh_Object & cobj = *this;
  DICe::mesh::Mesh_Object & obj = const_cast<DICe::mesh::Mesh_Object&>(cobj);

  deep_relations_map::iterator it=obj.get_deep_relations_map()->find(field_enums::ELEMENT_RANK);
  if(it==obj.get_deep_relations_map()->end())
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set;
    ptr_set.push_back(element);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::ELEMENT_RANK,ptr_set));
  }
  else
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set = it->second;
    ptr_set.push_back(element);
    obj.get_deep_relations_map()->erase(it);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::ELEMENT_RANK,ptr_set));
  }
}

void
Mesh_Object::add_deep_relation(const Teuchos::RCP<Internal_Face_Edge> internal_face_edge) const
{
  // cast away the constness because the mesh object is part of a set
  const DICe::mesh::Mesh_Object & cobj = *this;
  DICe::mesh::Mesh_Object & obj = const_cast<DICe::mesh::Mesh_Object&>(cobj);

  deep_relations_map::iterator it=obj.get_deep_relations_map()->find(field_enums::INTERNAL_FACE_EDGE_RANK);
  if(it==obj.get_deep_relations_map()->end())
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set;
    ptr_set.push_back(internal_face_edge);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::INTERNAL_FACE_EDGE_RANK,ptr_set));
  }
  else
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set = it->second;
    ptr_set.push_back(internal_face_edge);
    obj.get_deep_relations_map()->erase(it);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::INTERNAL_FACE_EDGE_RANK,ptr_set));
  }
}

//void
//Mesh_Object::add_deep_relation(const Teuchos::RCP<External_Face_Edge> external_face_edge) const
//{
//  // cast away the constness because the mesh object is part of a set
//  const DICe::mesh::Mesh_Object & cobj = *this;
//  DICe::mesh::Mesh_Object & obj = const_cast<DICe::mesh::Mesh_Object&>(cobj);
//
//  deep_relations_map::iterator it=obj.get_deep_relations_map()->find(field_enums::EXTERNAL_FACE_EDGE_RANK);
//  if(it==obj.get_deep_relations_map()->end())
//  {
//    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set;
//    ptr_set.push_back(external_face_edge);
//    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::EXTERNAL_FACE_EDGE_RANK,ptr_set));
//  }
//  else
//  {
//    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set = it->second;
//    ptr_set.push_back(external_face_edge);
//    obj.get_deep_relations_map()->erase(it);
//    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::EXTERNAL_FACE_EDGE_RANK,ptr_set));
//  }
//}


void
Mesh_Object::add_deep_relation(const Teuchos::RCP<Internal_Cell> internal_cell) const
{
  // cast away the constness because the mesh object is part of a set
  const DICe::mesh::Mesh_Object & cobj = *this;
  DICe::mesh::Mesh_Object & obj = const_cast<DICe::mesh::Mesh_Object&>(cobj);

  deep_relations_map::iterator it=obj.get_deep_relations_map()->find(field_enums::INTERNAL_CELL_RANK);
  if(it==obj.get_deep_relations_map()->end())
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set;
    ptr_set.push_back(internal_cell);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::INTERNAL_CELL_RANK,ptr_set));
  }
  else
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set = it->second;
    ptr_set.push_back(internal_cell);
    obj.get_deep_relations_map()->erase(it);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::INTERNAL_CELL_RANK,ptr_set));
  }
}

void
Mesh_Object::add_deep_relation(const Teuchos::RCP<Bond> bond) const
{
  // cast away the constness because the mesh object is part of a set
  const DICe::mesh::Mesh_Object & cobj = *this;
  DICe::mesh::Mesh_Object & obj = const_cast<DICe::mesh::Mesh_Object&>(cobj);

  deep_relations_map::iterator it=obj.get_deep_relations_map()->find(field_enums::BOND_RANK);
  if(it==obj.get_deep_relations_map()->end())
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set;
    ptr_set.push_back(bond);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::BOND_RANK,ptr_set));
  }
  else
  {
    std::vector<Teuchos::RCP<Mesh_Object> > ptr_set = it->second;
    ptr_set.push_back(bond);
    obj.get_deep_relations_map()->erase(it);
    obj.get_deep_relations_map()->insert(std::pair<field_enums::Entity_Rank,std::vector<Teuchos::RCP<Mesh_Object> > >(field_enums::BOND_RANK,ptr_set));
  }
}

int_t
Mesh_Object::is_elem_relation(const int_t gid,
  scalar_t & sign)
{
  for(size_t i=0;i<deep_bond_relations_.size();++i)
  {
    if(deep_bond_relations_[i]->get_left_global_id()==gid) {sign = 1.0; return deep_bond_relations_[i]->local_id();} // left elem is positive
    else if(deep_bond_relations_[i]->get_right_global_id()==gid) {sign = -1.0; return deep_bond_relations_[i]->local_id();} // right elem is negative
  }
  return -1;
}

void
Mesh_Object::set_initial_num_relations(const int_t num_relations,
  const field_enums::Entity_Rank relation_rank) const
{
  // cast away the constness because the mesh object is part of a set
  const DICe::mesh::Mesh_Object & cobj = *this;
  DICe::mesh::Mesh_Object & obj = const_cast<DICe::mesh::Mesh_Object&>(cobj);
  if(relation_rank==field_enums::ELEMENT_RANK)
    obj.initial_num_elem_relations_ = num_relations;
  else if(relation_rank==field_enums::NODE_RANK)
    obj.initial_num_node_relations_ = num_relations;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,"set_initial_num_relations: unknown entity rank");
}

Node::Node(const int_t global_id,
  const int_t local_id) :
  //local_id(local_id),
  Mesh_Object(global_id,local_id)
{}

Mesh::Mesh(const std::string & input_filename,
  const std::string & output_filename) :
  spatial_dimension_(-1),
  is_initialized_(false),
  max_num_elem_relations_(0),
  max_num_node_relations_(0),
  max_num_subelem_face_relations_(0),
  mean_num_elem_relations_(0),
  mean_num_node_relations_(0),
  mean_num_subelem_face_relations_(0),
  min_num_elem_relations_(0),
  min_num_node_relations_(0),
  min_num_subelem_face_relations_(0),
  max_num_entries_per_row_(0),
  input_exoid_(-1),
  output_exoid_(-1),
  face_edge_output_exoid_(-1),
  input_filename_(input_filename),
  output_filename_(output_filename),
  face_edge_output_filename_("face_edge_" + output_filename),
  control_volumes_are_initialized_(false),
  cell_sizes_are_initialized_(false),
  ic_value_x_(0.0),
  ic_value_y_(0.0),
  is_regular_grid_(false)
{
  comm_ = Teuchos::rcp(new MultiField_Comm());
  element_set_ = Teuchos::rcp(new element_set);
  subelement_set_ = Teuchos::rcp(new subelement_set);
  internal_face_edge_set_ = Teuchos::rcp(new internal_face_edge_set);
  //boundary_face_edge_set_ = Teuchos::rcp(new external_face_edge_set);
  edge_set_ = Teuchos::rcp(new edge_set);
  bond_set_ = Teuchos::rcp(new bond_set);
  internal_cell_set_ = Teuchos::rcp(new internal_cell_set);
  node_set_ = Teuchos::rcp(new node_set);
}

/// Create the field maps for the mixed formulation elements
void
Mesh::create_mixed_node_field_maps(Teuchos::RCP<Mesh> alt_mesh){
  DEBUG_MSG("Creating the mixed element and node field maps for the mesh");

  const int_t indexBase = 0;
  const int_t spa_dim = spatial_dimension();
  const int_t p_rank = comm_->get_rank();

  // get the number of nodes in the alt_mesh
  const int_t alt_num_nodes_this_proc = alt_mesh->num_nodes();
  // Some nodes are shared among different processors so the node_overlap_map is not 1-to-1!
  Teuchos::Array<int_t>::size_type num_nodes_this_proc = num_nodes();
  const int_t offset = num_nodes_this_proc*spa_dim;
  Teuchos::Array<int_t> node_list_mixed (num_nodes_this_proc*spa_dim + alt_num_nodes_this_proc); // add one for the lagrange multiplier on each lm node
  DICe::mesh::node_set::const_iterator node_it = get_node_set()->begin();
  DICe::mesh::node_set::const_iterator node_end = get_node_set()->end();
  int_t node_index = 0;
  for(;node_it!=node_end;++node_it)
  {
    for(int_t dim=0;dim<spa_dim;++dim)
    {
      const int_t index_stride = node_index * spa_dim + dim;
      node_list_mixed[index_stride] = node_it->first * spa_dim + dim;
    }
    node_index++;
  }
  node_it = alt_mesh->get_node_set()->begin();
  node_end = alt_mesh->get_node_set()->end();
  node_index = 0;
  for(;node_it!=node_end;++node_it)
  {
    const int_t index_stride = offset + node_index;
    node_list_mixed[index_stride] = node_it->first + vector_node_dist_map_->get_max_global_index();
    node_index++;
  }
  mixed_vector_node_overlap_map_ = Teuchos::rcp (new MultiField_Map(-1, node_list_mixed, indexBase, *comm_));
  //std::cout << " MIXED_VECTOR_NODE_OVERLAP MAP: " << std::endl;
  //mixed_vector_node_overlap_map_->describe();

  // go through all your local nodes, if the remote index list matches your processor than add it to the new_dist_list
  // otherwise another proc will pick it up
  const int_t total_num_nodes = mixed_vector_node_overlap_map_->get_num_global_elements();
  Teuchos::Array<int_t> nodeIDList(total_num_nodes);
  Teuchos::Array<int_t> GIDList(total_num_nodes);
  const int_t min_gid = mixed_vector_node_overlap_map_->get_min_global_index();
  Teuchos::Array<int_t> gids_on_this_proc_mixed;
  for(int_t i=0;i<total_num_nodes;++i)
    GIDList[i] = min_gid + i;
  mixed_vector_node_overlap_map_->get_remote_index_list(GIDList,nodeIDList);
  for(int_t i=0;i<total_num_nodes;++i)
    if(nodeIDList[i]==p_rank) // only add the nodes that have this processor as their remote index
      gids_on_this_proc_mixed.push_back(min_gid+i);
  mixed_vector_node_dist_map_ = Teuchos::rcp(new MultiField_Map(-1,gids_on_this_proc_mixed, indexBase, *comm_));
  //std::cout << " MIXED_VECTOR_NODE_DIST MAP: " << std::endl;
  //mixed_vector_node_dist_map_->describe();
}

void
Mesh::create_elem_node_field_maps(const bool force_elem_and_node_maps_to_match){
  DEBUG_MSG("Creating the element and node field maps for the mesh");
  const int_t indexBase = 0;
  const int_t spa_dim = spatial_dimension();
  const int_t p_rank = comm_->get_rank();

  Teuchos::Array<int_t> my_proc(1,p_rank);
  proc_map_ = Teuchos::rcp (new MultiField_Map(-1, my_proc, 0, *comm_));

  Teuchos::Array<int_t> all_proc;
  for(int_t i=0;i<comm_->get_size();++i)
     all_proc.push_back(i);
  all_own_all_proc_map_ = Teuchos::rcp (new MultiField_Map(-1, all_proc, 0, *comm_));

  // NODE DIST MAP
  // Some nodes are shared among different processors so the node_overlap_map is not 1-to-1!
  Teuchos::Array<int_t>::size_type num_nodes_this_proc = num_nodes();
  Teuchos::Array<int_t> node_list (num_nodes_this_proc);
  Teuchos::Array<int_t> node_list_vectorized (num_nodes_this_proc * spa_dim);
  DICe::mesh::node_set::const_iterator node_it = get_node_set()->begin();
  DICe::mesh::node_set::const_iterator node_end = get_node_set()->end();
  int_t node_index = 0;
  for(;node_it!=node_end;++node_it)
  {
    node_list[node_index] = node_it->first;
    for(int_t dim=0;dim<spa_dim;++dim)
    {
      const int_t index_stride = node_index * spa_dim + dim;
      node_list_vectorized[index_stride] = node_it->first * spa_dim + dim;
    }
    node_index++;
  }
  scalar_node_overlap_map_ = Teuchos::rcp (new MultiField_Map(-1, node_list, indexBase, *comm_));
  //std::cout << " SCALAR_NODE_OVERLAP MAP: " << std::endl;
  //scalar_node_overlap_map->describe();
  vector_node_overlap_map_ = Teuchos::rcp (new MultiField_Map(-1, node_list_vectorized, indexBase, *comm_));
  //std::cout << " VECTOR_NODE_OVERLAP MAP: " << std::endl;
  //vector_node_overlap_map->describe();


  // ELEM DIST MAP AND VECTORIZED MAP
  // This map should be 1-to-1
  Teuchos::Array<int_t>::size_type num_elem_this_proc = num_elem();
  Teuchos::Array<int_t> elem_list (num_elem_this_proc);
  Teuchos::Array<int_t> elem_list_vectorized (num_elem_this_proc * spa_dim);
  DICe::mesh::element_set::const_iterator elem_it = get_element_set()->begin();
  DICe::mesh::element_set::const_iterator elem_end = get_element_set()->end();
  int_t elem_index = 0;
  for(;elem_it!=elem_end;++elem_it)
  {
    elem_list[elem_index] = elem_it->get()->global_id();
    for(int_t dim=0;dim<spa_dim;++dim)
    {
      const int_t index_stride = elem_index * spa_dim + dim;
      const int_t stride = elem_it->get()->global_id() * spa_dim + dim;
      elem_list_vectorized[index_stride] = stride;
    }
    elem_index++;
  }

  scalar_elem_dist_map_ = Teuchos::rcp (new MultiField_Map(-1, elem_list, indexBase, *comm_));
  //std::cout << " ELEM DIST MAP " << std::endl;
  //scalar_elem_dist_map->describe();
  vector_elem_dist_map_ = Teuchos::rcp (new MultiField_Map(-1, elem_list_vectorized, indexBase, *comm_));

  if(force_elem_and_node_maps_to_match){
    scalar_node_dist_map_ = Teuchos::rcp(new MultiField_Map(-1,elem_list, indexBase, *comm_));
    vector_node_dist_map_ = Teuchos::rcp(new MultiField_Map(-1,elem_list_vectorized, indexBase, *comm_));
  }
  else{
    // go through all your local nodes, if the remote index list matches your processor then add it to the new_dist_list
    // otherwise another proc will pick it up
    const int_t total_num_nodes = scalar_node_overlap_map_->get_num_global_elements();
    Teuchos::Array<int_t> nodeIDList(total_num_nodes);
    Teuchos::Array<int_t> GIDList(total_num_nodes);
    const int_t min_gid = scalar_node_overlap_map_->get_min_global_index();
    //const int_t max_gid = scalar_node_overlap_map_->get_max_global_index();
    Teuchos::Array<int_t> gids_on_this_proc;
    Teuchos::Array<int_t> gids_on_this_proc_vectorized;
    for(int_t i=0;i<total_num_nodes;++i)
    {
      GIDList[i] = min_gid + i;
    }
    scalar_node_overlap_map_->get_remote_index_list(GIDList,nodeIDList);

    for(int_t i=0;i<total_num_nodes;++i)
    {
      if(nodeIDList[i]==p_rank) // only add the nodes that have this processor as their remote index
      {
        gids_on_this_proc.push_back(min_gid + i);
        for(int_t dim=0;dim<spa_dim;++dim)
          gids_on_this_proc_vectorized.push_back((min_gid + i)*spa_dim+dim);
      }
    }
    scalar_node_dist_map_ = Teuchos::rcp(new MultiField_Map(-1,gids_on_this_proc, indexBase, *comm_));
    //std::cout << " SCALAR_NODE_DIST MAP: " << std::endl;
    //scalar_node_dist_map->describe();
    vector_node_dist_map_ = Teuchos::rcp(new MultiField_Map(-1,gids_on_this_proc_vectorized, indexBase, *comm_));
    //std::cout << " VECTOR_NODE_DIST MAP: " << std::endl;
    //vector_node_dist_map->describe();
  }

  for(node_it = get_node_set()->begin(); node_it!=node_end;++node_it)
  {
    // since the map used by exodus for the nodes is the overlap map (nodes appear on multiple processors)
    // the local id as created by exodus is really the overlap_local_id. Here the local_id gets saved
    // off as it should in the overlap_local_id member data, and later we replace the local_id data
    // with the true local_id from the distributed map
    node_it->second->update_overlap_local_id(node_it->second->local_id());
    // replace the local id with the value corresponding to the dist map
    // NOTE some will be -1 if they are not locally owned
    node_it->second->update_local_id(scalar_node_dist_map_->get_local_element(node_it->first));
  }
}

void
Mesh::create_face_cell_field_maps(){
  DEBUG_MSG("Creating the face field maps for the mesh");
  const int_t indexBase = 0;
  const int_t spa_dim = spatial_dimension();

  DICe::mesh::element_set::const_iterator elem_it = get_element_set()->begin();
  DICe::mesh::element_set::const_iterator elem_end = get_element_set()->end();

  // INTERNAL CELL DISTRIBUTION MAPS

  bool post_warning = false;
  // count up the number of faces/cells per element:
  Teuchos::Array<int_t>::size_type num_faces_this_proc = 0;
  Teuchos::Array<int_t>::size_type num_cells_this_proc = 0;
  // search all the elements for thier element type
  elem_it = get_element_set()->begin();
  for(;elem_it!=elem_end;++elem_it)
  {
    Teuchos::RCP<DICe::mesh::Element> elem = *elem_it;
    const Base_Element_Type elem_type = get_block_type_map()->find(elem_it->get()->block_id())->second;
    if(elem_type == HEX8)
    {
      num_faces_this_proc+=6*6;
      num_cells_this_proc+=6*4;
    }
    else if(elem_type == TETRA4 || elem_type == TETRA)  // FIXME: we should only accept tetra4, tetra is a poor selection in cubit
    {
      num_faces_this_proc+=6;
      num_cells_this_proc+=4;
    }
    else if(elem_type == QUAD4)
    {
      num_faces_this_proc+=6;
      num_cells_this_proc+=6;
    }
    else if(elem_type == TRI3 || elem_type ==TRI6)
    {
      num_faces_this_proc+=3;
      num_cells_this_proc+=3;
    }
    else if(elem_type == PYRAMID5)
      post_warning = true;
    else if(elem_type == SPHERE) // sphere gets you kicked out of this method
      return;
    else
    {
      std::stringstream oss;
      oss << "create_field(): unknown element type: " << tostring(elem_type) << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
  }
  if(post_warning)
    std::cout << "    WARNING: Subelement distribution maps will not be created for PYRAMID5 elements" << std::endl;

  scalar_cell_dist_map_ = Teuchos::rcp (new MultiField_Map(-1,num_cells_this_proc, indexBase, *comm_));
  //std::cout << " SCALAR CELL DIST MAP " << std::endl;
  //scalar_cell_dist_map->describe();
  vector_cell_dist_map_ = Teuchos::rcp (new MultiField_Map(-1,num_cells_this_proc*spa_dim, indexBase, *comm_));
  //std::cout << " VECTOR CELL DIST MAP " << std::endl;
  //vector_cell_dist_map->describe();

  //int_t num_boundary_faces_this_proc = 0;
  // gather up all the external faces and add them to the list:
  std::set<std::pair<int_t,int_t>> elem_face_pairs;
  const int_t num_sets = side_set_info_.ids.size();
  for(int_t id_it=1;id_it<=num_sets;++id_it){
    const int_t set_index = side_set_info_.get_set_index(id_it);
    const int_t num_sides_this_set = side_set_info_.num_side_per_set[set_index];
    const int_t start_index = side_set_info_.elem_ind[set_index];
    for(int_t i=0;i<num_sides_this_set;++i){
      const int_t side_index = start_index+i;
      const int_t elem_gid = side_set_info_.ss_global_elem_list[side_index];
      const int_t side_id = side_set_info_.ss_side_list[side_index];
      // check if this elem is local to this process:
      if(scalar_elem_dist_map_->get_local_element(elem_gid)<0) continue;
      // add another face if the element is local to this process
      TEUCHOS_TEST_FOR_EXCEPTION(elem_face_pairs.find(std::pair<int_t,int_t>(elem_gid,side_id))!=elem_face_pairs.end(),
        std::runtime_error,"Error: Only one side set can be specified per element edge.");
      elem_face_pairs.insert(std::pair<int_t,int_t>(elem_gid,side_id));
      //num_boundary_faces_this_proc+=2; // each boundary face is split in two
      num_faces_this_proc+=2; // each boundary face is split in two
    }
  }
  //DEBUG_MSG("[" << p_rank << "] There are " << num_boundary_faces_this_proc << " boundary faces on this processor");

  DEBUG_MSG("[" << get_comm()->get_rank() << "] There are " << num_faces_this_proc << " internal faces and " << num_cells_this_proc << " cells on this processor");
  scalar_face_dist_map_ = Teuchos::rcp (new MultiField_Map(-1,num_faces_this_proc, indexBase, *comm_));
  //std::cout << " SCALAR FACE DIST MAP " << std::endl;
  //scalar_face_dist_map->describe();
  vector_face_dist_map_ = Teuchos::rcp (new MultiField_Map(-1,num_faces_this_proc*spa_dim, indexBase, *comm_));
  //std::cout << " VECTOR FACE DIST MAP " << std::endl;
  //vector_face_dist_map->describe();


//  scalar_boundary_face_dist_map_ = Teuchos::rcp (new MultiField_Map<int_t>(-1,num_faces_this_proc, indexBase, *comm_));
//  vector_boundary_face_dist_map_ = Teuchos::rcp (new MultiField_Map<int_t>(-1,num_faces_this_proc*spa_dim, indexBase, *comm_));

  // SUBELEMENT MAPS:

  // count up the number of faces/cells per element:
  Teuchos::Array<int_t>::size_type num_subelem_this_proc = 0;
  post_warning = false;

  // search all the elements for thier element type
  elem_it = get_element_set()->begin();
for(;elem_it!=elem_end;++elem_it)
  {
    Teuchos::RCP<DICe::mesh::Element> elem = *elem_it;
    const Base_Element_Type elem_type = get_block_type_map()->find(elem_it->get()->block_id())->second;
    if(elem_type == HEX8)
      num_subelem_this_proc+=6;
    else if(elem_type == TETRA4 || elem_type == TETRA)  // FIXME: we should only accept tetra4, tetra is a poor selection in cubit
      num_subelem_this_proc+=1;
    else if(elem_type == QUAD4)
      num_subelem_this_proc+=2;
    else if(elem_type == TRI3 || elem_type == TRI6)
      num_subelem_this_proc+=1;
    else if(elem_type == PYRAMID5)
      post_warning = true;
    else
    {
      std::stringstream oss;
      oss << "create_subelem_dist_map(): unknown element type: " << tostring(elem_type) << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
    }
  }
  if(post_warning)
    std::cout << "    WARNING: Subelement distribution maps will not be created for PYRAMID5 elements" << std::endl;

  scalar_subelem_dist_map_ = Teuchos::rcp (new MultiField_Map(-1, num_subelem_this_proc, indexBase, *comm_));
  //std::cout << "SCALAR SUBELEM MAP " << std::endl;
  //scalar_subelem_dist_map->describe();
  vector_subelem_dist_map_ = Teuchos::rcp (new MultiField_Map(-1, num_subelem_this_proc*spa_dim, indexBase, *comm_));
  //std::cout << "VECTOR SUBELEM MAP " << std::endl;
  //vector_subelem_dist_map->describe();
}


int_t
Mesh::num_elem_in_block(const int_t block_id)const
{
  int_t num_elem = 0;
  DICe::mesh::element_set::const_iterator elem_it = element_set_->begin();
  DICe::mesh::element_set::const_iterator elem_end = element_set_->end();
  for(;elem_it!=elem_end;++elem_it)
  {
    if(elem_it->get()->block_id()==block_id) num_elem++;
  }
  return num_elem;
}

Teuchos::RCP<Shape_Function_Evaluator> Shape_Function_Evaluator_Factory::create(const Base_Element_Type elem_type)
{
  if(elem_type==CVFEM_TRI)
    return Teuchos::rcp(new CVFEM_Linear_Tri3());
  else if(elem_type==CVFEM_TETRA)
    return Teuchos::rcp(new CVFEM_Linear_Tet4());
  else if(elem_type==HEX8)
    return Teuchos::rcp(new FEM_Linear_Hex8());
  else if(elem_type==QUAD4)
    return Teuchos::rcp(new FEM_Linear_Quad4());
  else if(elem_type==TRI3)
    return Teuchos::rcp(new FEM_Linear_Tri3());
  else if(elem_type==TRI6)
    return Teuchos::rcp(new FEM_Quadratic_Tri6());
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Shape_Function_Evaluator_Factory: invalid element type: " + tostring(elem_type));
  }
}


bool
Mesh::field_exists(const std::string & field_name){
  // only one field is allowed per field spec
  field_registry::iterator field_it = field_registry_.begin();
  field_registry::iterator field_end = field_registry_.end();
  for(;field_it!=field_end;++field_it)
    if(field_it->first.get_name_label()==field_name) return true;
  return false;
}

void
Mesh::create_field(const field_enums::Field_Spec & field_spec)
{
  // only one field is allowed per field spec
  if(field_registry_.find(field_spec)!=field_registry_.end())
  {
    std::cout << field_spec.get_name_label() << " " << field_spec.get_state() << std::endl;
    std::cout << field_registry_.find(field_spec)->first.get_name_label() << " " << field_registry_.find(field_spec)->first.get_state() << std::endl;
    std::stringstream oss;
    oss << " MG_Mesh::create_field(): duplicate request: " << field_spec.get_name_label() << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  Teuchos::RCP<MultiField_Map> map;
  if(field_spec.get_rank()==field_enums::NODE_RANK && field_spec.get_field_type()==field_enums::SCALAR_FIELD_TYPE)
    map = scalar_node_dist_map_;
  else if(field_spec.get_rank()==field_enums::NODE_RANK && field_spec.get_field_type()==field_enums::VECTOR_FIELD_TYPE)
    map = vector_node_dist_map_;
  else if(field_spec.get_rank()==field_enums::NODE_RANK && field_spec.get_field_type()==field_enums::MIXED_VECTOR_FIELD_TYPE)
    map = mixed_vector_node_dist_map_;
  else if(field_spec.get_rank()==field_enums::ELEMENT_RANK && field_spec.get_field_type()==field_enums::VECTOR_FIELD_TYPE)
    map = vector_elem_dist_map_;
  else if(field_spec.get_rank()==field_enums::ELEMENT_RANK && field_spec.get_field_type()==field_enums::SCALAR_FIELD_TYPE)
    map = scalar_elem_dist_map_;
  else if(field_spec.get_rank()==field_enums::INTERNAL_CELL_RANK && field_spec.get_field_type()==field_enums::VECTOR_FIELD_TYPE)
    map = vector_cell_dist_map_;
  else if(field_spec.get_rank()==field_enums::INTERNAL_CELL_RANK && field_spec.get_field_type()==field_enums::SCALAR_FIELD_TYPE)
    map = scalar_cell_dist_map_;
  else if(field_spec.get_rank()==field_enums::INTERNAL_FACE_EDGE_RANK && field_spec.get_field_type()==field_enums::VECTOR_FIELD_TYPE)
    map = vector_face_dist_map_;
  else if(field_spec.get_rank()==field_enums::INTERNAL_FACE_EDGE_RANK && field_spec.get_field_type()==field_enums::SCALAR_FIELD_TYPE)
    map = scalar_face_dist_map_;
  //else if(field_spec.get_rank()==field_enums::EXTERNAL_FACE_EDGE_RANK && field_spec.get_field_type()==field_enums::VECTOR_FIELD_TYPE)
  //  map = vector_boundary_face_dist_map_;
  //else if(field_spec.get_rank()==field_enums::EXTERNAL_FACE_EDGE_RANK && field_spec.get_field_type()==field_enums::SCALAR_FIELD_TYPE)
  //  map = scalar_boundary_face_dist_map_;
  else if(field_spec.get_rank()==field_enums::SUBELEMENT_RANK && field_spec.get_field_type()==field_enums::VECTOR_FIELD_TYPE)
    map = vector_subelem_dist_map_;
  else if(field_spec.get_rank()==field_enums::SUBELEMENT_RANK && field_spec.get_field_type()==field_enums::SCALAR_FIELD_TYPE)
    map = scalar_subelem_dist_map_;
  else
  {
    std::stringstream oss;
    oss << " MG_Mesh::create_field(): unknown rank and tensor order combination specified: " << tostring(field_spec.get_rank()) << " " << tostring(field_spec.get_field_type()) << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  Teuchos::RCP<MultiField> field_ptr = Teuchos::rcp(new MultiField(map,1,true));
  field_registry_.insert(std::pair<field_enums::Field_Spec,Teuchos::RCP<MultiField > >(field_spec,field_ptr));
}

void
Mesh::re_register_field(const field_enums::Field_Spec & existing_field_spec,
  const field_enums::Field_Spec & new_field_spec)
{
  if(field_registry_.find(existing_field_spec)==field_registry_.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Requested field does not exist" + existing_field_spec.get_name_label());
  }
  // double check that all the specs agree
  TEUCHOS_TEST_FOR_EXCEPTION(existing_field_spec.get_field_type()!=new_field_spec.get_field_type(),std::invalid_argument,"Field types must agree.");
  TEUCHOS_TEST_FOR_EXCEPTION(existing_field_spec.get_rank()!=new_field_spec.get_rank(),std::invalid_argument,"Field ranks must agree.");
  Teuchos::RCP<MultiField > field = get_field(existing_field_spec);
  field_registry_.insert(std::pair<field_enums::Field_Spec,Teuchos::RCP<MultiField > >(new_field_spec,field));
}

std::pair<field_enums::Field_Spec,Teuchos::RCP<MultiField> >
Mesh::get_field(const std::string & field_name)
{
  // if the requested field spec has more than one state, always retrieve the state n_plus_one spec
  // search the field specs of every field for this name:
  field_registry::const_iterator reg_it = field_registry_.begin();
  const field_registry::const_iterator reg_end = field_registry_.end();
  bool field_found = false;
  bool multi_state_field = false;
  field_enums::Field_Spec fs;
  for(;reg_it!=reg_end;++reg_it)
  {
    if(reg_it->first.get_name_label()==field_name)
    {
      if(field_found) multi_state_field = true;
      field_found = true;
      fs = reg_it->first;
    }
  }
  if(field_found)
  {
    if(multi_state_field) fs.set_state(field_enums::STATE_N_PLUS_ONE);
    return std::pair<field_enums::Field_Spec,Teuchos::RCP<MultiField > >(fs,field_registry_.find(fs)->second);
  }
  if(comm_->get_rank()==0) std::cout << " The following fields are defined: "  << std::endl;
  for(reg_it = field_registry_.begin();reg_it!=reg_end;++reg_it)
  {
    if(comm_->get_rank()==0) std::cout << "  " << reg_it->first.get_name_label() << std::endl;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Field was not found in the registry: " + field_name);
}

field_enums::Field_Spec
Mesh::get_field_spec(const std::string & field_name,
  const field_enums::Field_State state){
  // if the requested field spec has more than one state, always retrieve the state n_plus_one spec
  // search the field specs of every field for this name:
  field_registry::const_iterator reg_it = field_registry_.begin();
  const field_registry::const_iterator reg_end = field_registry_.end();
  bool field_found = false;
  bool multi_state_field = false;
  field_enums::Field_Spec fs;
  for(;reg_it!=reg_end;++reg_it)
  {
    if(reg_it->first.get_name_label()==field_name)
    {
      if(field_found) multi_state_field = true;
      field_found = true;
      fs = reg_it->first;
    }
  }
  if(field_found)
  {
    if(multi_state_field) fs.set_state(state);
    return fs;
  }
  if(comm_->get_rank()==0) std::cout << " The following fields are defined: "  << std::endl;
  for(reg_it = field_registry_.begin();reg_it!=reg_end;++reg_it)
  {
    if(comm_->get_rank()==0) std::cout << "  " << reg_it->first.get_name_label() << std::endl;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Field was not found in the registry: " + field_name);
}


std::vector<std::string>
Mesh::get_field_names(const field_enums::Entity_Rank entity_rank,
  const field_enums::Field_Type field_type,
  const bool only_the_printable)const
{
  std::vector<std::string> names;
  field_registry::const_iterator field_it = field_registry_.begin();
  field_registry::const_iterator field_end = field_registry_.end();
  for(;field_it!=field_end;++field_it)
  {
    if(field_it->first.get_field_type()==field_type && field_it->first.get_rank()==entity_rank)
    {
      if(!field_it->first.is_printable()&&only_the_printable) continue;
        names.push_back(field_it->first.get_name_label());
    }
  }
  return names;
}

Teuchos::RCP<MultiField >
Mesh::get_overlap_field(const field_enums::Field_Spec & field_spec)
{
  Teuchos::RCP<MultiField_Map> map;
  if(field_spec.get_rank()==field_enums::NODE_RANK && field_spec.get_field_type()==field_enums::SCALAR_FIELD_TYPE)
    map = scalar_node_overlap_map_;
  else if(field_spec.get_rank()==field_enums::NODE_RANK && field_spec.get_field_type()==field_enums::VECTOR_FIELD_TYPE)
    map = vector_node_overlap_map_;
  else if(field_spec.get_rank()==field_enums::NODE_RANK && field_spec.get_field_type()==field_enums::MIXED_VECTOR_FIELD_TYPE)
    map = mixed_vector_node_overlap_map_;
  else
  {
    std::stringstream oss;
    oss << " MG_Mesh::get_overlap_field(): unknown rank and tensor order combination specified: " << tostring(field_spec.get_rank()) << " " << tostring(field_spec.get_field_type()) << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  const Teuchos::RCP<MultiField > to_field = field_import(field_spec,map);
  return to_field;
}

Teuchos::RCP<MultiField>
Mesh::field_import(const field_enums::Field_Spec & field_spec,
  Teuchos::RCP<MultiField_Map> to_map)
{
  Teuchos::RCP<MultiField > from_field = field_registry_.find(field_spec)->second;
  Teuchos::RCP<MultiField > field_gather = Teuchos::rcp( new MultiField(to_map,1,true)); // FIXME: need to check this should be one

  MultiField_Exporter exporter(*to_map,*from_field->get_map());
  //export_type exporter (to_map->get(),from_field->get_map()->get());
  field_gather->do_import(from_field,exporter);
  //field_gather->get()->doImport((*from_field->get()), exporter, Tpetra::INSERT);
  return field_gather;
}

void
Mesh::field_overlap_export(const Teuchos::RCP<MultiField > from_field,
  const field_enums::Field_Spec & to_field_spec,
  const Combine_Mode mode)
{
  Teuchos::RCP<MultiField_Map> map;
  if(to_field_spec.get_rank()==field_enums::NODE_RANK && to_field_spec.get_field_type()==field_enums::SCALAR_FIELD_TYPE)
    map = scalar_node_overlap_map_;
  else if(to_field_spec.get_rank()==field_enums::NODE_RANK && to_field_spec.get_field_type()==field_enums::VECTOR_FIELD_TYPE)
    map = vector_node_overlap_map_;
  else if(to_field_spec.get_rank()==field_enums::NODE_RANK && to_field_spec.get_field_type()==field_enums::MIXED_VECTOR_FIELD_TYPE)
    map = mixed_vector_node_overlap_map_;
  else{
    std::stringstream oss;
    oss << " MG_Mesh::field_overlap_export(): unknown rank and tensor order combination specified: " << tostring(to_field_spec.get_rank()) << " " << tostring(to_field_spec.get_field_type()) << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  if(field_registry_.find(to_field_spec)==field_registry_.end()){
    std::stringstream oss;
    oss << " MG_Mesh::field_overlap_export(): could not find the destination field: " << to_field_spec.get_name_label() << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,oss.str());
  }
  Teuchos::RCP<MultiField > to_field = field_registry_.find(to_field_spec)->second;
  //export_type exporter (map->get(),to_field->get_map()->get());
  MultiField_Exporter exporter(*map,*to_field->get_map());
  to_field->do_export(from_field,exporter,mode);
//  to_field->get()->doExport((*from_field->get()), exporter, mode);
}

void
Mesh::print_field_info()
{
  if(comm_->get_rank()!=0) return;
#ifndef DICE_DEBUG_MSG
  return;
#endif
  std::cout << "  =============== REGISTERED NODAL SCALAR FIELDS ===============" << std::endl;
  std::vector<std::string> nodal_scalar_fields = get_field_names(field_enums::NODE_RANK,field_enums::SCALAR_FIELD_TYPE,false);
  for (size_t i = 0; i < nodal_scalar_fields.size(); ++i){
    std::cout << "  " << nodal_scalar_fields[i] << std::endl;
  }
  std::cout << "  =============== REGISTERED NODAL VECTOR FIELDS ===============" << std::endl;
  std::vector<std::string> nodal_vector_fields = get_field_names(field_enums::NODE_RANK,field_enums::VECTOR_FIELD_TYPE,false);
  for (size_t i = 0; i < nodal_vector_fields.size(); ++i){
    for(int_t j=0; j<spatial_dimension();++j)
      std::cout << "  " << nodal_vector_fields[i] + index_to_component_string(j) << std::endl;
  }
  std::cout << "  ============== REGISTERED NODAL MIXED VECTOR FIELDS ==========" << std::endl;
  std::vector<std::string> mixed_nodal_vector_fields = get_field_names(field_enums::NODE_RANK,field_enums::MIXED_VECTOR_FIELD_TYPE,false);
  for (size_t i = 0; i < mixed_nodal_vector_fields.size(); ++i){
    for(int_t j=0; j<spatial_dimension();++j)
      std::cout << "  " << mixed_nodal_vector_fields[i] + index_to_component_string(j) << std::endl;
    std::cout << "  " << mixed_nodal_vector_fields[i] + "LAM" << std::endl;
  }
  std::cout << "  ============== REGISTERED ELEMENT SCALAR FIELDS ==============" << std::endl;
  std::vector<std::string> element_scalar_fields = get_field_names(field_enums::ELEMENT_RANK,field_enums::SCALAR_FIELD_TYPE,false);
  for (size_t i = 0; i < element_scalar_fields.size(); ++i){
    std::cout << "  " << element_scalar_fields[i] << std::endl;
  }
  std::cout << "  ============== REGISTERED ELEMENT VECTOR FIELDS ==============" << std::endl;
  std::vector<std::string> element_vector_fields = get_field_names(field_enums::ELEMENT_RANK,field_enums::VECTOR_FIELD_TYPE,false);
  for (size_t i = 0; i < element_vector_fields.size(); ++i){
    for(int_t j=0; j<spatial_dimension();++j)
      std::cout << "  " << element_vector_fields[i] + index_to_component_string(j) << std::endl;
  }
  std::cout << "  ============== REGISTERED FACE EDGE SCALAR FIELDS ============" << std::endl;
  std::vector<std::string> face_scalar_fields = get_field_names(field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::SCALAR_FIELD_TYPE,false);
  for (size_t i = 0; i < face_scalar_fields.size(); ++i){
    std::cout << "  " << face_scalar_fields[i] << std::endl;
  }
  std::cout << "  ============== REGISTERED FACE EDGE VECTOR FIELDS ============" << std::endl;
  std::vector<std::string> face_vector_fields = get_field_names(field_enums::INTERNAL_FACE_EDGE_RANK,field_enums::VECTOR_FIELD_TYPE,false);
  for (size_t i = 0; i < face_vector_fields.size(); ++i){
    for(int_t j=0; j<spatial_dimension();++j)
      std::cout << "  " << face_vector_fields[i] + index_to_component_string(j) << std::endl;
  }
  //std::cout << "  ============== REGISTERED BOUNDARY FACE EDGE SCALAR FIELDS ==============" << std::endl;
  //std::vector<std::string> ex_face_scalar_fields = get_field_names(field_enums::EXTERNAL_FACE_EDGE_RANK,field_enums::SCALAR_FIELD_TYPE,false);
  //for (int_t i = 0; i < ex_face_scalar_fields.size(); ++i){
  //  std::cout << "  " << ex_face_scalar_fields[i] << std::endl;
  //}
  //std::cout << "  ============== REGISTERED BOUNDARY FACE EDGE VECTOR FIELDS ==============" << std::endl;
  //std::vector<std::string> ex_face_vector_fields = get_field_names(field_enums::EXTERNAL_FACE_EDGE_RANK,field_enums::VECTOR_FIELD_TYPE,false);
  //for (int_t i = 0; i < ex_face_vector_fields.size(); ++i){
  //  for(int_t j=0; j<spatial_dimension();++j)
  //    std::cout << "  " << ex_face_vector_fields[i] + index_to_component_string(j) << std::endl;
  //}
  std::cout <<  "  --------------------------------------------------------------------------------------" << std::endl;
}

scalar_t
Mesh::field_stats(const field_enums::Field_Spec & field_spec,
  scalar_t & min,
  scalar_t & max,
  scalar_t & avg,
  scalar_t & std_dev,
  const int_t comp,
  const field_enums::Field_Spec & marker_spec,
  const scalar_t & threshold){
  std::vector<std::string> comps;
  comps.push_back("_X");
  comps.push_back("_Y");
  field_registry::const_iterator field_it = field_registry_.find(field_spec);
  TEUCHOS_TEST_FOR_EXCEPTION(field_it==field_registry_.end(),std::runtime_error,
    "Error, invalid field " << field_spec.get_name_label());

  const int_t p_rank = comm_->get_rank();

  const bool has_marker = marker_spec != DICe::mesh::field_enums::NO_SUCH_FS;

  scalar_t failure_rate = 0.0;
  max = std::numeric_limits<scalar_t>::lowest();
  min = std::numeric_limits<scalar_t>::max();
  avg = 0.0;
  std_dev = 0.0;

  // Send the whole field to procesor 0

  // create all on zero map
  Teuchos::RCP<MultiField> field = field_it->second;
  const int_t num_values = field->get_map()->get_num_global_elements();
  Teuchos::Array<int_t> all_on_zero_ids;
  if(p_rank==0){
    all_on_zero_ids.resize(num_values);
    for(int_t i=0;i<num_values;++i)
       all_on_zero_ids[i] = i;
  }
  Teuchos::RCP<MultiField_Map> all_on_zero_map = Teuchos::rcp (new MultiField_Map(-1, all_on_zero_ids, 0, *comm_));
  Teuchos::RCP<MultiField> all_on_zero_field = Teuchos::rcp( new MultiField(all_on_zero_map,1,true));

  // create exporter with the map
  MultiField_Exporter exporter(*all_on_zero_map,*field->get_map());
  // export the field to zero
  all_on_zero_field->do_import(field,exporter);

  // bring in the marker field
  Teuchos::RCP<MultiField> marker_on_zero_field;
  Teuchos::RCP<MultiField_Map> marker_on_zero_map;
  Teuchos::Array<int_t> marker_on_zero_ids;
  if(has_marker){
    field_registry::const_iterator marker_it = field_registry_.find(marker_spec);
    TEUCHOS_TEST_FOR_EXCEPTION(marker_it==field_registry_.end(),std::runtime_error,
      "Error, invalid marker field " << marker_spec.get_name_label());
    Teuchos::RCP<MultiField> marker_field = marker_it->second;
    // for a scalar field the array list above can be reused
    if(field_spec.get_field_type()==DICe::mesh::field_enums::SCALAR_FIELD_TYPE){
      marker_on_zero_map = Teuchos::rcp (new MultiField_Map(-1, all_on_zero_ids, 0, *comm_));
    }else{
      if(p_rank==0){
        marker_on_zero_ids.resize(marker_field->get_map()->get_num_global_elements());
        for(int_t i=0;i<marker_field->get_map()->get_num_global_elements();++i)
           marker_on_zero_ids[i] = i;
      }
      marker_on_zero_map = Teuchos::rcp (new MultiField_Map(-1, marker_on_zero_ids, 0, *comm_));
    }
    marker_on_zero_field = Teuchos::rcp( new MultiField(marker_on_zero_map,1,true));
    // create exporter with the map
    MultiField_Exporter exporter_marker(*marker_on_zero_map,*marker_field->get_map());
    // export the field to zero
    marker_on_zero_field->do_import(marker_field,exporter_marker);
  }

  // analyze stats

  int_t num_failed = 0;
  const int_t num_points = all_on_zero_field->get_map()->get_num_local_elements();
  TEUCHOS_TEST_FOR_EXCEPTION(num_points > 0 && p_rank > 0,std::runtime_error,"Error, only process zero should have any points here");
  if(field_spec.get_field_type()==DICe::mesh::field_enums::SCALAR_FIELD_TYPE){
    TEUCHOS_TEST_FOR_EXCEPTION(comp!=0,std::runtime_error,"Error, invalid comp for scalar field")
    TEUCHOS_TEST_FOR_EXCEPTION(p_rank==0 && num_points<=0,std::runtime_error,"Error, num_points = 0");
    for(int_t i=0;i<num_points;++i){
      if(has_marker){
        if(marker_on_zero_field->local_value(i)<=threshold){
          num_failed++;
          continue;
        }
      }
      avg += all_on_zero_field->local_value(i);
      if(all_on_zero_field->local_value(i)>max) max = all_on_zero_field->local_value(i);
      if(all_on_zero_field->local_value(i)<min) min = all_on_zero_field->local_value(i);
    }
    if(num_points - num_failed > 0)
      avg /= (num_points-num_failed);
    for(int_t i=0;i<num_points;++i){
      if(has_marker){
        if(marker_on_zero_field->local_value(i)<=threshold){
          continue;
        }
      }
      std_dev += (all_on_zero_field->local_value(i)-avg)*(all_on_zero_field->local_value(i)-avg);
    }
    if(num_points-num_failed > 0)
      std_dev = std::sqrt(std_dev/(num_points-num_failed));
  }
  else if(field_spec.get_field_type()==DICe::mesh::field_enums::VECTOR_FIELD_TYPE){
    TEUCHOS_TEST_FOR_EXCEPTION(comp!=0&&comp!=1,std::runtime_error,"Error, invalid comp for scalar field")
    for(int_t i=0;i<num_points/2;++i){
      if(has_marker){
        if(marker_on_zero_field->local_value(i)<=threshold){
          num_failed++;
          continue;
        }
      }
      int_t index = i*spatial_dimension() + comp;
      avg += all_on_zero_field->local_value(index);
      if(all_on_zero_field->local_value(index)>max) max = all_on_zero_field->local_value(index);
      if(all_on_zero_field->local_value(index)<min) min = all_on_zero_field->local_value(index);
    }
    if(num_points - num_failed> 0)
      avg /= (num_points-num_failed)*0.5;
    for(int_t i=0;i<num_points/2;++i){
      int_t index = i*spatial_dimension() + comp;
      if(has_marker){
        if(marker_on_zero_field->local_value(i)<=threshold){
          continue;
        }
      }
      std_dev += (all_on_zero_field->local_value(index)-avg)*(all_on_zero_field->local_value(index)-avg);
    }
    if(num_points - num_failed > 0)
      std_dev = std::sqrt(std_dev/(0.5*(num_points-num_failed)));
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: unknown field type in stats call");
  }

  // communicate results back to dist procs

  Teuchos::Array<int_t> results_on_zero_ids;
  if(p_rank==0){
    results_on_zero_ids.resize(5);
    for(int_t i=0;i<5;++i)
       results_on_zero_ids[i] = i;
  }
  // create a field to send out the results to the other procs:
  Teuchos::RCP<MultiField_Map> results_on_zero_map = Teuchos::rcp (new MultiField_Map(-1, results_on_zero_ids, 0, *comm_));
  Teuchos::RCP<MultiField> results_on_zero_field = Teuchos::rcp( new MultiField(results_on_zero_map,1,true));
  // set the values on zero
  if(p_rank==0){
    results_on_zero_field->local_value(0) = min;
    results_on_zero_field->local_value(1) = max;
    results_on_zero_field->local_value(2) = avg;
    results_on_zero_field->local_value(3) = std_dev;
    results_on_zero_field->local_value(4) = num_failed / scalar_node_dist_map_->get_num_global_elements();
  }
  // create the all on all map
  Teuchos::Array<int_t> results_on_all_ids;
  results_on_all_ids.resize(5);
  for(int_t i=0;i<5;++i)
    results_on_all_ids[i] = i;
  // create a field to send out the results to the other procs:
  Teuchos::RCP<MultiField_Map> results_on_all_map = Teuchos::rcp (new MultiField_Map(-1, results_on_all_ids, 0, *comm_));
  Teuchos::RCP<MultiField> results_on_all_field = Teuchos::rcp( new MultiField(results_on_all_map,1,true));

  // create exporter with the map
  MultiField_Exporter exporter_res(*results_on_all_map,*results_on_zero_field->get_map());
  // export the field to zero
  results_on_all_field->do_import(results_on_zero_field,exporter_res);

  if(p_rank==0){
    TEUCHOS_TEST_FOR_EXCEPTION(results_on_all_field->local_value(0)!=min,std::runtime_error,"Error miscommunication of field stats across processors");
    TEUCHOS_TEST_FOR_EXCEPTION(results_on_all_field->local_value(1)!=max,std::runtime_error,"Error miscommunication of field stats across processors");
    TEUCHOS_TEST_FOR_EXCEPTION(results_on_all_field->local_value(2)!=avg,std::runtime_error,"Error miscommunication of field stats across processors");
    TEUCHOS_TEST_FOR_EXCEPTION(results_on_all_field->local_value(3)!=std_dev,std::runtime_error,"Error miscommunication of field stats across processors");
  }else{
    min = results_on_all_field->local_value(0);
    max = results_on_all_field->local_value(1);
    avg = results_on_all_field->local_value(2);
    std_dev = results_on_all_field->local_value(3);
  }
  failure_rate = results_on_all_field->local_value(4);

  return failure_rate;
}

void
Mesh::print_field_stats()
{
  if(comm_->get_rank()!=0) return;

  std::cout.fill('*');
  std::cout.width(90);
  std::cout << "*" << std::endl;
  std::cout.fill(' ');
  std::cout.width(30);
  std::cout << "Field";
  std::cout.width(15);
  std::cout << "Max";
  std::cout.width(15);
  std::cout << "Min";
  std::cout.width(15);
  std::cout << "Avg";
  std::cout.width(15);
  std::cout << "Std Dev" << std::endl;
  std::cout.width(90);
  std::cout.fill('-');
  std::cout << "-" << std::endl;
  std::cout.fill(' ');
  std::vector<std::string> comps;
  comps.push_back("_X");
  comps.push_back("_Y");
  // iterate the fields
  std::vector<std::string> names;
  field_registry::const_iterator field_it = field_registry_.begin();
  field_registry::const_iterator field_end = field_registry_.end();
  for(;field_it!=field_end;++field_it)
  {
    if(!field_it->first.is_printable())continue;
    if(field_it->first.get_field_type()==DICe::mesh::field_enums::SCALAR_FIELD_TYPE){
      scalar_t max = 0.0;
      scalar_t min = 0.0;
      scalar_t avg = 0.0;
      scalar_t std_dev = 0.0;
      field_stats(field_it->first,min,max,avg,std_dev,0);
      std::cout.width(30);
      std::cout << field_it->first.get_name_label();
      std::cout.width(15);
      std::cout << max;
      std::cout.width(15);
      std::cout << min;
      std::cout.width(15);
      std::cout << avg;
      std::cout.width(15);
      std::cout << std_dev << std::endl;
    }
    else if(field_it->first.get_field_type()==DICe::mesh::field_enums::VECTOR_FIELD_TYPE){
      for(int_t dim=0;dim<spatial_dimension();++dim){
        scalar_t max = 0.0;
        scalar_t min = 0.0;
        scalar_t avg = 0.0;
        scalar_t std_dev = 0.0;
        field_stats(field_it->first,min,max,avg,std_dev,dim);
        std::cout.width(30);
        std::cout << field_it->first.get_name_label() + comps[dim];
        std::cout.width(15);
        std::cout << max;
        std::cout.width(15);
        std::cout << min;
        std::cout.width(15);
        std::cout << avg;
        std::cout.width(15);
        std::cout << std_dev << std::endl;
      }
    }
  }
  std::cout.fill('*');
  std::cout.width(90);
  std::cout << "*" << std::endl;
  std::cout.fill(' ');
}


bool
CVFEM_Linear_Tri3::is_in_element(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient){
  const scalar_t A[] = {point_coords[0], point_coords[1]};
  scalar_t B[2], C[2];
  scalar_t area=0.0;
  scalar_t sum_area=0.0;

  // N1 = Area p23/A
  B[0] = nodal_coords[2];
  B[1] = nodal_coords[3];
  C[0] = nodal_coords[4];
  C[1] = nodal_coords[5];
  area = cross(&A[0],&B[0],&C[0]);
  area =  0.5 * std::abs(area);
  sum_area += area;

  // N2 = Area p13/A
  B[0] = nodal_coords[0];
  B[1] = nodal_coords[1];
  C[0] = nodal_coords[4];
  C[1] = nodal_coords[5];
  area = cross(&A[0],&B[0],&C[0]);
  area =  0.5 * std::abs(area);
  sum_area += area;

  // N3 = Area p12/A
  B[0] = nodal_coords[0];
  B[1] = nodal_coords[1];
  C[0] = nodal_coords[2];
  C[1] = nodal_coords[3];
  area = cross(&A[0],&B[0],&C[0]);
  area =  0.5 * std::abs(area);
  sum_area += area;

  if(std::abs(sum_area - coefficient)<1E-4)
    return true;
  else
    return false;
}

void
CVFEM_Linear_Tri3::get_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){
  tri3d_natural_integration_points(order,locations,weights,num_points);
}

void
CVFEM_Linear_Tri3::evaluate_shape_functions(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient,
  scalar_t * shape_function_values)
{
  const scalar_t A[] = {point_coords[0], point_coords[1]};
  scalar_t B[2], C[2];
  scalar_t area=0.0;
  scalar_t sum_area=0.0;

  // N1 = Area p23/A
  B[0] = nodal_coords[2];
  B[1] = nodal_coords[3];
  C[0] = nodal_coords[4];
  C[1] = nodal_coords[5];
  area = cross(&A[0],&B[0],&C[0]);
  area =  0.5 * std::abs(area);
  sum_area += area;
  shape_function_values[0] = area/coefficient;

  // N2 = Area p13/A
  B[0] = nodal_coords[0];
  B[1] = nodal_coords[1];
  C[0] = nodal_coords[4];
  C[1] = nodal_coords[5];
  area = cross(&A[0],&B[0],&C[0]);
  area =  0.5 * std::abs(area);
  sum_area += area;
  shape_function_values[1] = area/coefficient;

  // N3 = Area p12/A
  B[0] = nodal_coords[0];
  B[1] = nodal_coords[1];
  C[0] = nodal_coords[2];
  C[1] = nodal_coords[3];
  area = cross(&A[0],&B[0],&C[0]);
  area =  0.5 * std::abs(area);
  sum_area += area;
  shape_function_values[2] = area/coefficient;

  // check that the shape functions sum to one at the point
  //stringstream oss;
  //std::cout << " sum_area: " << sum_area << " coefficient: " << coefficient << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(sum_area - coefficient)>1E-3,std::logic_error,"Areas do not sum to total subelem area."); // + oss.str());
}

void
CVFEM_Linear_Tri3::evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
  const scalar_t & coefficient,
  scalar_t * shape_function_derivative_values)
{
  const scalar_t x1 = nodal_coords[0];
  const scalar_t y1 = nodal_coords[1];
  const scalar_t x2 = nodal_coords[2];
  const scalar_t y2 = nodal_coords[3];
  const scalar_t x3 = nodal_coords[4];
  const scalar_t y3 = nodal_coords[5];

  shape_function_derivative_values[0] = (y2-y3)/(2.0*coefficient);
  shape_function_derivative_values[1] = (x3-x2)/(2.0*coefficient);
  shape_function_derivative_values[2] = (y3-y1)/(2.0*coefficient);
  shape_function_derivative_values[3] = (x1-x3)/(2.0*coefficient);
  shape_function_derivative_values[4] = (y1-y2)/(2.0*coefficient);
  shape_function_derivative_values[5] = (x2-x1)/(2.0*coefficient);
}

void
CVFEM_Linear_Tet4::evaluate_shape_functions(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient,
  scalar_t * shape_function_values)
{
  scalar_t volume = 0.0;
  scalar_t sum_volume = 0.0;
  scalar_t det = 0.0;
  scalar_t a[4*4];

  // N1 = Volume XABC / Volume IABC
  for(int_t i=0;i<3;++i)
  {
    a[i+0*4] = nodal_coords[1*3+i];
    a[i+1*4] = nodal_coords[2*3+i];
    a[i+2*4] = nodal_coords[3*3+i];
    a[i+3*4] = point_coords[i];
  }
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  det = determinant_4x4(&a[0]);
  volume = std::abs(det) / 6.0;
  sum_volume+=volume;
  shape_function_values[0] = volume/coefficient;

  // N2
  for(int_t i=0;i<3;++i)
  {
    a[i+0*4] = nodal_coords[0*3+i];
    a[i+1*4] = nodal_coords[2*3+i];
    a[i+2*4] = nodal_coords[3*3+i];
    a[i+3*4] = point_coords[i];
  }
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  det = determinant_4x4(&a[0]);
  volume = std::abs(det) / 6.0;
  sum_volume+=volume;
  shape_function_values[1] = volume/coefficient;

  // N3
  for(int_t i=0;i<3;++i)
  {
    a[i+0*4] = nodal_coords[0*3+i];
    a[i+1*4] = nodal_coords[1*3+i];
    a[i+2*4] = nodal_coords[3*3+i];
    a[i+3*4] = point_coords[i];
  }
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  det = determinant_4x4(&a[0]);
  volume = std::abs(det) / 6.0;
  sum_volume+=volume;
  shape_function_values[2] = volume/coefficient;

  // N4
  for(int_t i=0;i<3;++i)
  {
    a[i+0*4] = nodal_coords[0*3+i];
    a[i+1*4] = nodal_coords[1*3+i];
    a[i+2*4] = nodal_coords[2*3+i];
    a[i+3*4] = point_coords[i];
  }
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  det = determinant_4x4(&a[0]);
  volume = std::abs(det) / 6.0;
  sum_volume+=volume;
  shape_function_values[3] = volume/coefficient;

  //stringstream oss;
  //oss << " sum_volume: " << sum_volume << " coefficient: " << coefficient << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(sum_volume - coefficient)>1E-3,std::logic_error,"Areas do not sum to total subelem area.");// + oss.str());
}

bool
CVFEM_Linear_Tet4::is_in_element(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient)
{
  scalar_t volume = 0.0;
  scalar_t sum_volume = 0.0;
  scalar_t det = 0.0;
  scalar_t a[4*4];

  // N1 = Volume XABC / Volume IABC
  for(int_t i=0;i<3;++i)
  {
    a[i+0*4] = nodal_coords[1*3+i];
    a[i+1*4] = nodal_coords[2*3+i];
    a[i+2*4] = nodal_coords[3*3+i];
    a[i+3*4] = point_coords[i];
  }
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  det = determinant_4x4(&a[0]);
  volume = std::abs(det) / 6.0;
  sum_volume+=volume;

  // N2
  for(int_t i=0;i<3;++i)
  {
    a[i+0*4] = nodal_coords[0*3+i];
    a[i+1*4] = nodal_coords[2*3+i];
    a[i+2*4] = nodal_coords[3*3+i];
    a[i+3*4] = point_coords[i];
  }
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  det = determinant_4x4(&a[0]);
  volume = std::abs(det) / 6.0;
  sum_volume+=volume;

  // N3
  for(int_t i=0;i<3;++i)
  {
    a[i+0*4] = nodal_coords[0*3+i];
    a[i+1*4] = nodal_coords[1*3+i];
    a[i+2*4] = nodal_coords[3*3+i];
    a[i+3*4] = point_coords[i];
  }
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  det = determinant_4x4(&a[0]);
  volume = std::abs(det) / 6.0;
  sum_volume+=volume;

  // N4
  for(int_t i=0;i<3;++i)
  {
    a[i+0*4] = nodal_coords[0*3+i];
    a[i+1*4] = nodal_coords[1*3+i];
    a[i+2*4] = nodal_coords[2*3+i];
    a[i+3*4] = point_coords[i];
  }
  for (int_t j = 0; j < 4; j++ )
  {
    a[3+j*4] = 1.0;
  }
  det = determinant_4x4(&a[0]);
  volume = std::abs(det) / 6.0;
  sum_volume+=volume;

  if(std::abs(sum_volume - coefficient)<1E-4)
    return true;
  else
    return false;
}

void
CVFEM_Linear_Tet4::evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
  const scalar_t & coefficient,
  scalar_t * shape_function_derivative_values)
{
  scalar_t A[3];
  scalar_t B[3];
  scalar_t C[3];
  scalar_t I[3];
  scalar_t cross[3];

  for(int_t i=0;i<3;++i)
  {
    I[i] = nodal_coords[i];
    A[i] = nodal_coords[1*3+i];
    B[i] = nodal_coords[2*3+i];
    C[i] = nodal_coords[3*3+i];
  }

  const scalar_t coefficient_times_6 = coefficient * 6.0;
  // Derivs at node 0
  // -[(B-A)x(C-A)]_0 / T_I
  // -[(B-A)x(C-A)]_1 / T_I
  // -[(B-A)x(C-A)]_2 / T_I
  cross3d_with_cross_prod(&A[0],&B[0],&C[0],&cross[0]);
  shape_function_derivative_values[0] = -1.0*cross[0]/coefficient_times_6;
  shape_function_derivative_values[1] = -1.0*cross[1]/coefficient_times_6;
  shape_function_derivative_values[2] = -1.0*cross[2]/coefficient_times_6;

  // -[(C-B)x(I-B)]_0 / T_A
  // -[(C-B)x(I-B)]_1 / T_A
  // -[(C-B)x(I-B)]_2 / T_A
  cross3d_with_cross_prod(&B[0],&C[0],&I[0],&cross[0]);
  shape_function_derivative_values[3] = 1.0*cross[0]/coefficient_times_6;
  shape_function_derivative_values[4] = 1.0*cross[1]/coefficient_times_6;
  shape_function_derivative_values[5] = 1.0*cross[2]/coefficient_times_6;

  // -[(I-C)x(A-C)]_0 / T_B
  // -[(I-C)x(A-C)]_1 / T_B
  // -[(I-C)x(A-C)]_2 / T_B
  cross3d_with_cross_prod(&C[0],&I[0],&A[0],&cross[0]);
  shape_function_derivative_values[6] = -1.0*cross[0]/coefficient_times_6;
  shape_function_derivative_values[7] = -1.0*cross[1]/coefficient_times_6;
  shape_function_derivative_values[8] = -1.0*cross[2]/coefficient_times_6
     ;

  // -[(A-I)x(B-I)]_0 / T_C
  // -[(A-I)x(B-I)]_1 / T_C
  // -[(A-I)x(B-I)]_2 / T_C
  cross3d_with_cross_prod(&I[0],&A[0],&B[0],&cross[0]);
  shape_function_derivative_values[9]  = 1.0*cross[0]/coefficient_times_6;
  shape_function_derivative_values[10] = 1.0*cross[1]/coefficient_times_6;
  shape_function_derivative_values[11] = 1.0*cross[2]/coefficient_times_6;
}

bool
FEM_Linear_Hex8::is_in_element(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Method not implemented yet");
  return false;
}

void
FEM_Linear_Hex8::get_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){
  //assert(order == 1 && "Error: only first order integration has been implemented");

  const int_t spa_dim = 3;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > kron_sol, zeta1, zeta2,weight1,weight2, temp_weights;
  num_points = order*order*order;
  locations.resize(num_points);
  weights.resize(num_points);
  kron_sol.resize(num_points);
  for(int_t i=0;i<num_points;++i){
    locations[i].resize(spa_dim);
    kron_sol[i].resize(2);
  }
  int_t i=0,j=0,m=order,n=1;
  //  create a matrix of ones
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > ones;
  ones.resize(m);
  for(int_t i=0;i<m;++i)
    ones[i].resize(n);
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      ones[i][j] = 1.0;
    };
  };

  gauss_1D(zeta1, weight1, order);
  gauss_2D(zeta2, weight2, order);

  kron_sol = kronecker(ones,zeta2);

  for(i=0;i<order*order*order;i++){
    for(j=1;j<=2;j++){
      locations[i][j] = kron_sol[i][j-1];
    }
  };

  for(i=0;i<order;i++){
    for(j=i*order*order;j<(i+1)*order*order;j++){
      locations[j][0] = zeta1[i][0];
    };
  };

//  for(int_t i=0;i<weight1.size();++i)
//    for(int_t j=0;j<weight1[i].size();++j)
//      std::cout << " weight1 " << i << " " << j << " " << weight1[i][j] << std::endl;
//
//  for(int_t i=0;i<weight2.size();++i)
//    for(int_t j=0;j<weight2[i].size();++j)
//      std::cout << " weight2 " << i << " " << j << " " << weight2[i][j] << std::endl;

  temp_weights = kronecker(weight1,weight2);

  for(int_t i=0;i<num_points;++i)
    weights[i] = temp_weights[i][0];
}


void
FEM_Linear_Hex8::evaluate_shape_functions(const scalar_t * natural_coords,
  scalar_t * shape_function_values){

  const scalar_t xi   = natural_coords[0];
  const scalar_t eta  = natural_coords[1];
  const scalar_t zeta = natural_coords[2];

  shape_function_values[0] = (1.0/8.0) * (1-xi)*(1-eta)*(1+zeta);
  shape_function_values[1] = (1.0/8.0) * (1-xi)*(1-eta)*(1-zeta);
  shape_function_values[2] = (1.0/8.0) * (1-xi)*(1+eta)*(1-zeta);
  shape_function_values[3] = (1.0/8.0) * (1-xi)*(1+eta)*(1+zeta);
  shape_function_values[4] = (1.0/8.0) * (1+xi)*(1-eta)*(1+zeta);
  shape_function_values[5] = (1.0/8.0) * (1+xi)*(1-eta)*(1-zeta);
  shape_function_values[6] = (1.0/8.0) * (1+xi)*(1+eta)*(1-zeta);
  shape_function_values[7] = (1.0/8.0) * (1+xi)*(1+eta)*(1+zeta);
}

void
FEM_Linear_Hex8::evaluate_shape_function_derivatives(const scalar_t * natural_coords,
  scalar_t * shape_function_derivative_values){
  const scalar_t xi   = natural_coords[0];
  const scalar_t eta  = natural_coords[1];
  const scalar_t zeta = natural_coords[2];

  const int_t spa_dim = 3;

  shape_function_derivative_values[0*spa_dim+0] = (-1.0/8.0)*(1+zeta)*(1-eta);
  shape_function_derivative_values[1*spa_dim+0] = (-1.0/8.0)*(1-zeta)*(1-eta);
  shape_function_derivative_values[2*spa_dim+0] = (-1.0/8.0)*(1-zeta)*(1+eta);
  shape_function_derivative_values[3*spa_dim+0] = (-1.0/8.0)*(1+zeta)*(1+eta);
  shape_function_derivative_values[4*spa_dim+0] = (1.0/8.0)*(1+zeta)*(1-eta);
  shape_function_derivative_values[5*spa_dim+0] = (1.0/8.0)*(1-zeta)*(1-eta);
  shape_function_derivative_values[6*spa_dim+0] = (1.0/8.0)*(1-zeta)*(1+eta);
  shape_function_derivative_values[7*spa_dim+0] = (1.0/8.0)*(1+zeta)*(1+eta);
  shape_function_derivative_values[0*spa_dim+1] = (-1.0/8.0)*(1+zeta)*(1-xi);
  shape_function_derivative_values[1*spa_dim+1] = (-1.0/8.0)*(1-zeta)*(1-xi);
  shape_function_derivative_values[2*spa_dim+1] = (1.0/8.0)*(1-zeta)*(1-xi);
  shape_function_derivative_values[3*spa_dim+1] = (1.0/8.0)*(1+zeta)*(1-xi);
  shape_function_derivative_values[4*spa_dim+1] = (-1.0/8.0)*(1+zeta)*(1+xi);
  shape_function_derivative_values[5*spa_dim+1] = (-1.0/8.0)*(1-zeta)*(1+xi);
  shape_function_derivative_values[6*spa_dim+1] = (1.0/8.0)*(1-zeta)*(1+xi);
  shape_function_derivative_values[7*spa_dim+1] = (1.0/8.0)*(1+zeta)*(1+xi);
  shape_function_derivative_values[0*spa_dim+2] = (1.0/8.0)*(1-eta)*(1-xi);
  shape_function_derivative_values[1*spa_dim+2] = (-1.0/8.0)*(1-eta)*(1-xi);
  shape_function_derivative_values[2*spa_dim+2] = (-1.0/8.0)*(1+eta)*(1-xi);
  shape_function_derivative_values[3*spa_dim+2] = (1.0/8.0)*(1+eta)*(1-xi);
  shape_function_derivative_values[4*spa_dim+2] = (1.0/8.0)*(1-eta)*(1+xi);
  shape_function_derivative_values[5*spa_dim+2] = (-1.0/8.0)*(1-eta)*(1+xi);
  shape_function_derivative_values[6*spa_dim+2] = (-1.0/8.0)*(1+eta)*(1+xi);
  shape_function_derivative_values[7*spa_dim+2] = (1.0/8.0)*(1+eta)*(1+xi);

}

bool
FEM_Linear_Quad4::is_in_element(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Method not implemented yet");
  return false;
}

void
FEM_Linear_Quad4::get_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){
  //assert(order == 1 && "Error: only first order integration has been implemented");

  const int_t spa_dim = 2;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > zeta,weight;
  num_points = order*order;
  locations.resize(num_points);
  weights.resize(num_points);
  for(int_t i=0;i<num_points;++i){
    locations[i].resize(spa_dim);
  }
  gauss_2D(locations, weight, order);
  for(int_t i=0;i<num_points;++i)
    weights[i] = weight[i][0];
}

void
FEM_Linear_Quad4::evaluate_shape_functions(const scalar_t * natural_coords,
  scalar_t * shape_function_values){

  const scalar_t xi   = natural_coords[0];
  const scalar_t eta  = natural_coords[1];

  shape_function_values[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
  shape_function_values[1] = 0.25 * (1.0 + xi) * (1.0 - eta);
  shape_function_values[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
  shape_function_values[3] = 0.25 * (1.0 - xi) * (1.0 + eta);
}

void
FEM_Linear_Quad4::evaluate_shape_function_derivatives(const scalar_t * natural_coords,
  scalar_t * shape_function_derivative_values){
  const scalar_t xi   = natural_coords[0];
  const scalar_t eta  = natural_coords[1];
  const int_t spa_dim = 2;
  shape_function_derivative_values[0*spa_dim+0] = -0.25 * (1.0 - eta);
  shape_function_derivative_values[1*spa_dim+0] =  0.25 * (1.0 - eta);
  shape_function_derivative_values[2*spa_dim+0] =  0.25 * (1.0 + eta);
  shape_function_derivative_values[3*spa_dim+0] = -0.25 * (1.0 + eta);
  shape_function_derivative_values[0*spa_dim+1] = -0.25 * (1.0 - xi);
  shape_function_derivative_values[1*spa_dim+1] = -0.25 * (1.0 + xi);
  shape_function_derivative_values[2*spa_dim+1] = 0.25 * (1.0 + xi);
  shape_function_derivative_values[3*spa_dim+1] = 0.25 * (1.0 - xi);
}


bool
FEM_Linear_Tri3::is_in_element(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Method not implemented yet");
  return false;
}

void
FEM_Linear_Tri3::get_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){
  tri2d_natural_integration_points(order,locations,weights,num_points);
}

void
FEM_Linear_Tri3::evaluate_shape_functions(const scalar_t * natural_coords,
  scalar_t * shape_function_values){
  const scalar_t xi   = natural_coords[0];
  const scalar_t eta  = natural_coords[1];

  shape_function_values[0] = 1 - xi - eta;
  shape_function_values[1] = xi;
  shape_function_values[2] = eta;
}

void
FEM_Linear_Tri3::evaluate_shape_function_derivatives(const scalar_t * natural_coords,
  scalar_t * shape_function_derivative_values){
  // natural coords not needed because constant, but passing in to make the
  // function call similar
  const int_t spa_dim = 2;

  shape_function_derivative_values[0*spa_dim+0] = -1;
  shape_function_derivative_values[1*spa_dim+0] = 1;
  shape_function_derivative_values[2*spa_dim+0] = 0;
  shape_function_derivative_values[0*spa_dim+1] = -1;
  shape_function_derivative_values[1*spa_dim+1] = 0;
  shape_function_derivative_values[2*spa_dim+1] = 1;
}

bool
FEM_Quadratic_Tri6::is_in_element(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Method not implemented yet");
  return false;
}

void
FEM_Quadratic_Tri6::get_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){
  // Note assumes the coordinates for the natural points has two values, x and y
  tri2d_natural_integration_points(order,locations,weights,num_points);
}

void
FEM_Quadratic_Tri6::evaluate_shape_functions(const scalar_t * natural_coords,
  scalar_t * shape_function_values){

  // Note assumes the coordinates for the natural points has two values, x and y
  const scalar_t xi   = natural_coords[0];
  const scalar_t eta  = natural_coords[1];
  shape_function_values[0] = (1.0 - 2.0 * xi - 2.0 * eta) * (1.0 - xi - eta);
  shape_function_values[1] = xi * (2.0 * xi - 1.0);
  shape_function_values[2] = eta * (2.0 * eta - 1.0);
  shape_function_values[3] = 4.0 * xi * (1.0 - xi - eta);
  shape_function_values[4] = 4.0 * xi * eta;
  shape_function_values[5] = 4.0 * eta * (1.0 - xi - eta);
}

void
FEM_Quadratic_Tri6::evaluate_shape_function_derivatives(const scalar_t * natural_coords,
  scalar_t * shape_function_derivative_values){
  // Note assumes the coordinates for the natural points has two values, x and y
  const scalar_t xi   = natural_coords[0];
  const scalar_t eta  = natural_coords[1];
  const int_t spa_dim = 2;
  shape_function_derivative_values[0*spa_dim+0] = -2.0 * (1.0 - xi - eta) -1.0 * (1.0 - 2.0 * xi - 2.0 * eta);
  shape_function_derivative_values[0*spa_dim+1] = -2.0 * (1.0 - xi - eta) -1.0 * (1.0 - 2.0 * xi - 2.0 * eta);
  shape_function_derivative_values[1*spa_dim+0] = 4.0 * xi - 1.0;
  shape_function_derivative_values[1*spa_dim+1] = 0.0;
  shape_function_derivative_values[2*spa_dim+0] = 0.0;
  shape_function_derivative_values[2*spa_dim+1] = 4.0 * eta - 1.0;
  shape_function_derivative_values[3*spa_dim+0] = 4.0 * (1.0 - 2.0 * xi - eta);
  shape_function_derivative_values[3*spa_dim+1] = -4.0 * xi;
  shape_function_derivative_values[4*spa_dim+0] = 4.0 * eta;
  shape_function_derivative_values[4*spa_dim+1] = 4.0 * xi;
  shape_function_derivative_values[5*spa_dim+0] = -4.0 * eta;
  shape_function_derivative_values[5*spa_dim+1] = 4.0 * (1.0 - xi - 2.0 * eta);
}


bool
FEM_Barycentric_Tri6::is_in_element(const scalar_t * nodal_coords,
  const scalar_t * point_coords,
  const scalar_t & coefficient){
  TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Method not implemented yet");
  return false;
}

void
FEM_Barycentric_Tri6::get_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){

  if(order==3){
    num_points = 6;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(3); // there are 3 coordinates for Barycentric elements z1, z2, and z3
    weights.resize(num_points);

    locations[0][0] = 0.816847572980458; locations[0][1] = 0.091576213509771; locations[0][2] = 0.091576213509771;
    locations[1][0] = 0.091576213509771; locations[1][1] = 0.816847572980458; locations[1][2] = 0.091576213509771;
    locations[2][0] = 0.091576213509771; locations[2][1] = 0.091576213509771; locations[2][2] = 0.816847572980458;
    locations[3][0] = 0.108103018168070; locations[3][1] = 0.445948490915965; locations[3][2] = 0.445948490915965;
    locations[4][0] = 0.445948490915965; locations[4][1] = 0.108103018168070; locations[4][2] = 0.445948490915965;
    locations[5][0] = 0.445948490915965; locations[5][1] = 0.445948490915965; locations[5][2] = 0.108103018168070;
    weights[0] = 0.109951743655322;
    weights[1] = 0.109951743655322;
    weights[2] = 0.109951743655322;
    weights[3] = 0.223381589678011;
    weights[4] = 0.223381589678011;
    weights[5] = 0.223381589678011;
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid order for Barycentric_Tri6 natural integration points");
  }
}

void
FEM_Barycentric_Tri6::evaluate_shape_functions(const scalar_t * natural_coords,
  scalar_t * shape_function_values){

  // Note assumes the coordinates for the natural points has 3 values
  const scalar_t z1  = natural_coords[0];
  const scalar_t z2  = natural_coords[1];
  const scalar_t z3  = natural_coords[2];
  shape_function_values[0] = z1*(2.0*z1 - 1.0);
  shape_function_values[1] = z2*(2.0*z2 - 1.0);
  shape_function_values[2] = z3*(2.0*z3 - 1.0);
  shape_function_values[3] = 4.0*z1*z2;
  shape_function_values[4] = 4.0*z2*z3;
  shape_function_values[5] = 4.0*z3*z1;
}

void
FEM_Barycentric_Tri6::evaluate_shape_function_derivatives(const scalar_t * natural_coords,
  scalar_t * shape_function_derivative_values){
  // Note assumes the coordinates for the natural points has three values
  const scalar_t z1  = natural_coords[0];
  const scalar_t z2  = natural_coords[1];
  const scalar_t z3  = natural_coords[2];
  const int_t spa_dim = 3; // three coords per barycentric point
  shape_function_derivative_values[0*spa_dim+0] = 4.0*z1-1.0;
  shape_function_derivative_values[0*spa_dim+1] = 0.0;
  shape_function_derivative_values[0*spa_dim+2] = 0.0;
  shape_function_derivative_values[1*spa_dim+0] = 0.0;
  shape_function_derivative_values[1*spa_dim+1] = 4.0*z2-1.0;
  shape_function_derivative_values[1*spa_dim+2] = 0.0;
  shape_function_derivative_values[2*spa_dim+0] = 0.0;
  shape_function_derivative_values[2*spa_dim+1] = 0.0;
  shape_function_derivative_values[2*spa_dim+2] = 4.0*z3-1.0;
  shape_function_derivative_values[3*spa_dim+0] = 4.0*z2;
  shape_function_derivative_values[3*spa_dim+1] = 4.0*z1;
  shape_function_derivative_values[3*spa_dim+2] = 0.0;
  shape_function_derivative_values[4*spa_dim+0] = 0.0;
  shape_function_derivative_values[4*spa_dim+1] = 4.0*z3;
  shape_function_derivative_values[4*spa_dim+2] = 4.0*z2;
  shape_function_derivative_values[5*spa_dim+0] = 4.0*z3;
  shape_function_derivative_values[5*spa_dim+1] = 0.0;
  shape_function_derivative_values[5*spa_dim+2] = 4.0*z1;
}

// mesh generation utilities:
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Mesh> create_point_or_tri_mesh(const DICe::mesh::Base_Element_Type elem_type,
  Teuchos::ArrayRCP<scalar_t> node_coords_x,
  Teuchos::ArrayRCP<scalar_t> node_coords_y,
  Teuchos::ArrayRCP<int_t> connectivity, // this is based on the gids of the elements (always 1..n never 0 based)
  Teuchos::ArrayRCP<int_t> node_map,
  Teuchos::ArrayRCP<int_t> elem_map,
  std::vector<std::pair<int_t,int_t> > & dirichlet_boundary_nodes,
  std::set<int_t> & neumann_boundary_nodes,
  std::set<int_t> & lagrange_boundary_nodes,
  const std::string & serial_output_filename)
{
  DEBUG_MSG("create_exodus_mesh(): creating an exodus mesh");
  TEUCHOS_TEST_FOR_EXCEPTION(elem_type!=DICe::mesh::TRI6&&elem_type!=DICe::mesh::TRI3&&elem_type!=DICe::mesh::MESHLESS,
    std::runtime_error,
    "Error, invalid element type");
  int_t num_nodes_per_elem = 1;
  if(elem_type==DICe::mesh::TRI6)
    num_nodes_per_elem = 6;
  else if(elem_type==DICe::mesh::TRI3)
    num_nodes_per_elem = 3;
  TEUCHOS_TEST_FOR_EXCEPTION((elem_type==DICe::mesh::TRI6&&connectivity.size()%6!=0)||
    (elem_type==DICe::mesh::TRI3&&connectivity.size()%3!=0),std::runtime_error,
    "Error connectivity should be a multiple of 6 for TRI6 or 3 for TRI3");
  const int_t num_elem = connectivity.size()/num_nodes_per_elem;
  // check the input data
  TEUCHOS_TEST_FOR_EXCEPTION(node_coords_x.size()<0,std::runtime_error,
    "Error node coords x is of zero size");
  TEUCHOS_TEST_FOR_EXCEPTION(node_coords_x.size()!=node_coords_y.size(),std::runtime_error,
    "Error the dims of the node coord vectors don't match");
  const int_t num_nodes = node_coords_x.size();
  std::stringstream out_file_base;
  // find the position of the file extension
  size_t pos_out = serial_output_filename.find(".g");
  if(pos_out==std::string::npos){
    pos_out = serial_output_filename.find(".e");
  }
  if(pos_out==std::string::npos)
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, output mesh files must have the extension .g or .e");
  out_file_base << serial_output_filename;

  int_t num_procs = 1;
  int_t my_rank = 0;
#ifdef HAVE_MPI
  int_t mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  TEUCHOS_TEST_FOR_EXCEPTION(!mpi_is_initialized,std::runtime_error,"Error: if MPI is enabled, MPI must be initialized at this point.");
  MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
#endif
  if(num_procs>1)
  {
    std::stringstream temp_ss1, temp_ss2, file_name_post;
    temp_ss1 << num_procs;
    const size_t procs_num_digits = temp_ss1.str().length();
    temp_ss2 << my_rank;
    const size_t rank_num_digits = temp_ss2.str().length();
    const int_t num_fill_zeros = procs_num_digits - rank_num_digits;
    file_name_post << "." << num_procs << ".";
    for(int_t i=0;i<num_fill_zeros;++i)
      file_name_post << "0";
    file_name_post << my_rank;
    out_file_base << file_name_post.str();
  }
  std::string output_filename = out_file_base.str();
  DEBUG_MSG("Output file name: " << output_filename);

  Teuchos::RCP<DICe::mesh::Mesh> mesh = Teuchos::rcp(new DICe::mesh::Mesh(output_filename));
  const int_t p_rank = mesh->get_comm()->get_rank();
  mesh->set_input_exoid(-1);
  const int_t num_dim = 2;
  mesh->set_spatial_dimension(num_dim);

  // put all the nodes in the mesh
  for(int_t i =0;i<num_nodes;++i){
    Teuchos::RCP<Node> node_rcp = Teuchos::rcp(new DICe::mesh::Node(node_map[i],i));
    mesh->get_node_set()->insert(std::pair<int_t,Teuchos::RCP<Node> >(node_rcp->global_id(),node_rcp));
  }

  // create one block all of type TRI6 or TRI3 elems
  if(elem_type==DICe::mesh::MESHLESS)
    mesh->get_block_type_map()->insert(std::pair<int_t,Base_Element_Type>(1,MESHLESS));
  else if(elem_type==DICe::mesh::TRI6)
    mesh->get_block_type_map()->insert(std::pair<int_t,Base_Element_Type>(1,TRI6));
  else
    mesh->get_block_type_map()->insert(std::pair<int_t,Base_Element_Type>(1,TRI3));

  // read element connectivity
  Teuchos::ArrayRCP<int_t> connectivity_swap(num_nodes_per_elem);
  for(int_t j=0; j<num_elem; ++j)
  {
    connectivity_vector conn;
    //std::cout << " Element " << j << std::endl;
    // the connectivity vector needs to be re-arranged from Triangle format to exodus format
    connectivity_swap[0] = connectivity[j*num_nodes_per_elem + 0];
    if(elem_type==DICe::mesh::TRI6||elem_type==DICe::mesh::TRI3){
      connectivity_swap[1] = connectivity[j*num_nodes_per_elem + 1];
      connectivity_swap[2] = connectivity[j*num_nodes_per_elem + 2];
      if(elem_type==DICe::mesh::TRI6){
        connectivity_swap[4] = connectivity[j*num_nodes_per_elem + 3];
        connectivity_swap[5] = connectivity[j*num_nodes_per_elem + 4];
        connectivity_swap[3] = connectivity[j*num_nodes_per_elem + 5];
      }
    }
    for(int_t k=0;k<num_nodes_per_elem;++k)
    {
      const int_t global_node_id = node_map[connectivity_swap[k]-1];
      // find the node in the set with the same global id
      if(mesh->get_node_set()->find(global_node_id)!=mesh->get_node_set()->end()){
        conn.push_back(mesh->get_node_set()->find(global_node_id)->second);
      }
      else{
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"Could not find node rcp in set: " << global_node_id);
      }
    }
    // create an element and put it in the mesh
    const int_t elem_local_id = j;
    const int_t elem_global_id = elem_map[j];
    const int_t elem_index_in_block = elem_local_id;
    Teuchos::RCP<DICe::mesh::Element> element_rcp = Teuchos::rcp(new DICe::mesh::Element(conn,elem_global_id,elem_local_id,1,elem_index_in_block));
    mesh->get_element_set()->push_back(element_rcp);
  }

  // since we are going to read in some field data like coordinates or volumes, etc
  // we pause here to generate the Tpetra maps that will organize this stuff
  const bool force_elem_and_node_maps_to_match = num_nodes_per_elem == 1;
  mesh->create_elem_node_field_maps(force_elem_and_node_maps_to_match);

  mesh->create_field(field_enums::BLOCK_ID_FS);
  MultiField & block_id_field = *mesh->get_field(field_enums::BLOCK_ID_FS);
  DICe::mesh::element_set::const_iterator elem_it = mesh->get_element_set()->begin();
  DICe::mesh::element_set::const_iterator elem_end = mesh->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it){
    block_id_field.local_value(elem_it->get()->local_id()) = elem_it->get()->block_id();
  }

  //mesh->create_field(field_enums::MASTER_NODE_ID_FS);
  //MultiField & master_field = *mesh->get_field(field_enums::MASTER_NODE_ID_FS);
  //DICe::mesh::node_set::iterator node_it = mesh->get_node_set()->begin();
  //DICe::mesh::node_set::iterator node_end = mesh->get_node_set()->end();
  //for(;node_it!=node_end;++node_it){
  //  master_field.local_value(node_it->second->local_id()) = node_it->second->local_id();
  //}

  // put the element blocks in a field so they are accessible in parallel from other processors
  block_type_map::iterator blk_map_it = mesh->get_block_type_map()->begin();
  block_type_map::iterator blk_map_end = mesh->get_block_type_map()->end();
  for(;blk_map_it!=blk_map_end;++blk_map_it)
  {
    // TODO: do the same for faces/edges
    Teuchos::RCP<element_set> elem_set = Teuchos::rcp(new element_set);
    for(elem_it=mesh->get_element_set()->begin();elem_it!=elem_end;++elem_it)
    {
      if(elem_it->get()->block_id()==blk_map_it->first)
      {
        elem_set->push_back(*elem_it);
      }
    }
    mesh->get_element_sets_by_block()->insert(std::pair<int_t,Teuchos::RCP<element_set> >(blk_map_it->first,elem_set));
  }

  //std::vector<int_t> dirichlet_bc_def;
  //std::map<int_t,int_t>::const_iterator mit = dirichlet_boundary_nodes.begin();
  //std::map<int_t,int_t>::const_iterator mit_end = dirichlet_boundary_nodes.end();
  std::map<int_t,std::vector<int_t> > bc_node_collections;
  for(size_t i=0;i<dirichlet_boundary_nodes.size();++i){
    if(bc_node_collections.find(dirichlet_boundary_nodes[i].second)==bc_node_collections.end())
      bc_node_collections.insert(std::pair<int_t,std::vector<int_t> >(dirichlet_boundary_nodes[i].second,std::vector<int_t>()));
    bc_node_collections.find(dirichlet_boundary_nodes[i].second)->second.push_back(dirichlet_boundary_nodes[i].first);
  }
  if(bc_node_collections.size()>0){
    std::map<int_t,std::vector<int_t> >::iterator col_it = bc_node_collections.begin();
    std::map<int_t,std::vector<int_t> >::iterator col_end = bc_node_collections.end();
    for(;col_it!=col_end;++col_it){
      mesh->get_node_bc_sets()->insert(std::pair<int_t,std::vector<int_t> >(col_it->first,col_it->second));
    }
  }
  std::vector<int_t> neumann_bc_def;
  std::set<int_t>::const_iterator it = neumann_boundary_nodes.begin();
  std::set<int_t>::const_iterator it_end = neumann_boundary_nodes.end();
  it = neumann_boundary_nodes.begin();
  it_end = neumann_boundary_nodes.end();
  for(;it!=it_end;++it){
    neumann_bc_def.push_back(*it);
  }
  if(neumann_bc_def.size()>0)
    mesh->get_node_bc_sets()->insert(std::pair<int_t,std::vector<int_t> >(0,neumann_bc_def)); // all neumann boundary nodes go in node set 0
  std::vector<int_t> lagrange_bc_def;
  it = lagrange_boundary_nodes.begin();
  it_end = lagrange_boundary_nodes.end();
  for(;it!=it_end;++it){
    lagrange_bc_def.push_back(*it);
  }
  if(lagrange_bc_def.size()>0)
    mesh->get_node_bc_sets()->insert(std::pair<int_t,std::vector<int_t> >(-1,lagrange_bc_def)); // all lagrange boundary nodes go in node set -1

  mesh->set_initialized();

  //mesh->create_face_cell_field_maps();

  // initialize the fields needed for coordinates etc:
  mesh->create_field(field_enums::INITIAL_COORDINATES_FS);
  Teuchos::RCP<MultiField > initial_coords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);

  for(int_t i=0;i<num_nodes;++i)  // i represents the local_id
  {
    initial_coords->local_value(i*num_dim+0) = node_coords_x[i];
    initial_coords->local_value(i*num_dim+1) = node_coords_y[i];
  }

  // export the coordinates back to the non-overlap field
  mesh->field_overlap_export(initial_coords, field_enums::INITIAL_COORDINATES_FS, INSERT);
  //std::cout << "INITIAL COORDS: " << std::endl;
  //initial_coords_ptr->vec()->describe();

  // CELL COORDINATES
  mesh->create_field(field_enums::INITIAL_CELL_COORDINATES_FS);
  Teuchos::RCP<MultiField > coords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);
  MultiField & initial_cell_coords = *mesh->get_field(field_enums::INITIAL_CELL_COORDINATES_FS);

  //compute the centroid from the coordinates of the nodes;
  for(elem_it=mesh->get_element_set()->begin();elem_it!=elem_end;++elem_it)
  {
    const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
    scalar_t centroid[num_dim];
    for(int_t i=0;i<num_dim;++i) centroid[i]=0.0;
    for(size_t node_it=0;node_it<connectivity.size();++node_it)
    {
      const Teuchos::RCP<DICe::mesh::Node> node = connectivity[node_it];
      centroid[0] += coords->local_value(node.get()->overlap_local_id()*num_dim+0);
      centroid[1] += coords->local_value(node.get()->overlap_local_id()*num_dim+1);
    }
    for(int_t dim=0;dim<num_dim;++dim)
    {
      centroid[dim] /= connectivity.size();
      initial_cell_coords.local_value(elem_it->get()->local_id()*num_dim+dim) = centroid[dim];
    }
  }
  //std::cout << " INITIAL CELL COORDS: " << std::endl;
  //initial_cell_coords.vec()->describe();

  // PROCESSOR ID

  // create the cell_coords, cell_radius and cell_size fields:
  mesh->create_field(field_enums::PROCESSOR_ID_FS);
  MultiField & proc_id = *mesh->get_field(field_enums::PROCESSOR_ID_FS);
  elem_it = mesh->get_element_set()->begin();
  for(;elem_it!=elem_end;++elem_it){
    proc_id.local_value(elem_it->get()->local_id()) = p_rank;
  }

  DEBUG_MSG("  ------------------ Analysis Model Definition (processsor " << p_rank <<") --------------------------");
  DEBUG_MSG("  Title:             " << "autogenerated Triangle mesh, or subset points");
  DEBUG_MSG("  Output file:       " << output_filename);
  DEBUG_MSG("  Spatial dimension: " << num_dim);
  DEBUG_MSG("  Nodes:             " << num_nodes);
  DEBUG_MSG("  Elements:          " << num_elem);
  DEBUG_MSG("  Dirichlet nodes:   " << dirichlet_boundary_nodes.size());
  DEBUG_MSG("  Neumann nodes:     " << neumann_boundary_nodes.size());
  //DEBUG_MSG("  Element blocks:    " << num_elem_blk);
  DEBUG_MSG("  Node sets:         " << mesh->num_node_sets());
  //DEBUG_MSG("  Side sets:         " << num_side_sets);
  DEBUG_MSG("  --------------------------------------------------------------------------------------");

  return mesh;
}

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Mesh> create_tri3_mesh_from_tri6(Teuchos::RCP<Mesh> tri6_mesh,
  const std::string & serial_output_filename){
  DEBUG_MSG("create_tri3_mesh_from_tri6(): creating a tri3 mesh from a tri6 one");
  // find the position of the file extension
  std::stringstream out_file_base;
  size_t pos_out = serial_output_filename.find(".g");
  if(pos_out==std::string::npos){
    pos_out = serial_output_filename.find(".e");
  }
  if(pos_out==std::string::npos)
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, output mesh files must have the extension .g or .e");
  out_file_base << serial_output_filename;

  const int_t spa_dim = tri6_mesh->spatial_dimension();
  int_t num_procs = 1;
  int_t my_rank = 0;
#ifdef HAVE_MPI
  int_t mpi_is_initialized = 0;
  MPI_Initialized(&mpi_is_initialized);
  TEUCHOS_TEST_FOR_EXCEPTION(!mpi_is_initialized,std::runtime_error,"Error: if MPI is enabled, MPI must be initialized at this point.");
  MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
#endif
  if(num_procs>1)
  {
    std::stringstream temp_ss1, temp_ss2, file_name_post;
    temp_ss1 << num_procs;
    const size_t procs_num_digits = temp_ss1.str().length();
    temp_ss2 << my_rank;
    const size_t rank_num_digits = temp_ss2.str().length();
    const int_t num_fill_zeros = procs_num_digits - rank_num_digits;
    file_name_post << "." << num_procs << ".";
    for(int_t i=0;i<num_fill_zeros;++i)
      file_name_post << "0";
    file_name_post << my_rank;
    out_file_base << file_name_post.str();
  }
  std::string output_filename = out_file_base.str();
  DEBUG_MSG("Output file name: " << output_filename);

  // iterate the elements, taking only the first three indices of the connectivity to
  // populate the node list:
  std::map<int_t,scalar_t> coords_x;
  std::map<int_t,scalar_t> coords_y;

  // element loop
  TEUCHOS_TEST_FOR_EXCEPTION(tri6_mesh->get_block_type_map()->begin()->second!=TRI6,std::runtime_error,
    "Error, the input mesh must be TRI6 elements");
  Teuchos::RCP<MultiField > tri6_initial_coords = tri6_mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);
  DICe::mesh::element_set::iterator elem_it = tri6_mesh->get_element_set()->begin();
  DICe::mesh::element_set::iterator elem_end = tri6_mesh->get_element_set()->end();
  //const int_t tri6_num_nodes_per_elem = 6;
  const int_t tri3_num_nodes_per_elem = 3;
  for(;elem_it!=elem_end;++elem_it)
  {
    //std::cout << "ELEM: " << elem_it->get()->global_id() << std::endl;
    const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
    for(int_t nd=0;nd<tri3_num_nodes_per_elem;++nd){ // only collect the first three nodes
      const int_t node_gid = connectivity[nd]->global_id();
      const int_t node_olid = connectivity[nd]->overlap_local_id();
      coords_x.insert(std::pair<int_t,scalar_t>(node_gid,tri6_initial_coords->local_value(node_olid*spa_dim+0)));
      coords_y.insert(std::pair<int_t,scalar_t>(node_gid,tri6_initial_coords->local_value(node_olid*spa_dim+1)));
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x.size()!=coords_y.size(),std::runtime_error,"Error, coords should be the same size");
  const int_t num_nodes = coords_x.size();
  int_t * local_to_master_node_map = new int_t[num_nodes];

  Teuchos::RCP<DICe::mesh::Mesh> mesh = Teuchos::rcp(new DICe::mesh::Mesh(output_filename));
  const int_t p_rank = mesh->get_comm()->get_rank();
  mesh->set_input_exoid(-1);
  const int_t num_dim = 2;
  mesh->set_spatial_dimension(num_dim);

  //  std::cout << "nodes that got collected " << std::endl;
  std::map<int_t,scalar_t>::iterator it=coords_x.begin();
  std::map<int_t,scalar_t>::iterator it_end=coords_x.end();
  int_t node_index = 0;
  std::map<int_t,int_t> master_to_local_node_map;
  for(;it!=it_end;++it){
    local_to_master_node_map[node_index] = it->first;
    //master_to_local_node_map.insert(std::pair<int_t,int_t>(it->first,node_index));
    Teuchos::RCP<Node> node_rcp = Teuchos::rcp(new DICe::mesh::Node(it->first,node_index));
    mesh->get_node_set()->insert(std::pair<int_t,Teuchos::RCP<Node> >(node_rcp->global_id(),node_rcp));
    node_index++;
    //std::cout << "gid " << it->first << " x " << it->second << std::endl;
  }

  // create one block all of type TRI3 elems
  mesh->get_block_type_map()->insert(std::pair<int_t,Base_Element_Type>(1,TRI3));

  for(elem_it = tri6_mesh->get_element_set()->begin();elem_it!=elem_end;++elem_it)
  {
    connectivity_vector conn;
    const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
    for(int_t k=0;k<tri3_num_nodes_per_elem;++k)
    {
      //TEUCHOS_TEST_FOR_EXCEPTION(master_to_local_node_map.find(connectivity[k]->global_id())==master_to_local_node_map.end(),std::runtime_error,
      //  "Error, invalid master to local node.");
      const int_t global_node_id = connectivity[k]->global_id();//master_to_local_node_map.find(connectivity[k]->global_id())->second;
      // find the node in the set with the same global id
      if(mesh->get_node_set()->find(global_node_id)!=mesh->get_node_set()->end()){
        conn.push_back(mesh->get_node_set()->find(global_node_id)->second);
      }
      else{
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"Could not find node rcp in set " << global_node_id);
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)conn.size()!=tri3_num_nodes_per_elem,std::runtime_error,"Error, connectivity is of invalid size");
    const int_t elem_local_id = elem_it->get()->local_id();
    const int_t elem_global_id = elem_it->get()->global_id();
    const int_t elem_index_in_block = elem_it->get()->index_in_block();
    Teuchos::RCP<DICe::mesh::Element> element_rcp = Teuchos::rcp(new DICe::mesh::Element(conn,elem_global_id,elem_local_id,1,elem_index_in_block));
    mesh->get_element_set()->push_back(element_rcp);
  }

  // since we are going to read in some field data like coordinates or volumes, etc
  // we pause here to generate the Tpetra maps that will organize this stuff
  mesh->create_elem_node_field_maps();

  mesh->create_field(field_enums::BLOCK_ID_FS);
  MultiField & block_id_field = *mesh->get_field(field_enums::BLOCK_ID_FS);
  elem_it = mesh->get_element_set()->begin();
  elem_end = mesh->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it){
    block_id_field.local_value(elem_it->get()->local_id()) = elem_it->get()->block_id();
  }

  //mesh->create_field(field_enums::MASTER_NODE_ID_FS);
  //MultiField & master_field = *mesh->get_field(field_enums::MASTER_NODE_ID_FS);
  //DICe::mesh::node_set::iterator node_it = mesh->get_node_set()->begin();
  //DICe::mesh::node_set::iterator node_end = mesh->get_node_set()->end();
  //for(;node_it!=node_end;++node_it){
  //  master_field.local_value(node_it->second->local_id()) = local_to_master_node_map[node_it->second->local_id()];
  //}

  // put the element blocks in a field so they are accessible in parallel from other processors
  block_type_map::iterator blk_map_it = mesh->get_block_type_map()->begin();
  block_type_map::iterator blk_map_end = mesh->get_block_type_map()->end();
  for(;blk_map_it!=blk_map_end;++blk_map_it)
  {
    // TODO: do the same for faces/edges
    Teuchos::RCP<element_set> elem_set = Teuchos::rcp(new element_set);
    for(elem_it=mesh->get_element_set()->begin();elem_it!=elem_end;++elem_it)
    {
      if(elem_it->get()->block_id()==blk_map_it->first)
      {
        elem_set->push_back(*elem_it);
      }
    }
    mesh->get_element_sets_by_block()->insert(std::pair<int_t,Teuchos::RCP<element_set> >(blk_map_it->first,elem_set));
  }
  // TODO deal with boundary conditions on the lagrange mesh
//
//  std::vector<int_t> dirichlet_bc_def;
//  std::set<int_t>::const_iterator it = dirichlet_boundary_nodes.begin();
//  std::set<int_t>::const_iterator it_end = dirichlet_boundary_nodes.end();
//  for(;it!=it_end;++it){
//    dirichlet_bc_def.push_back(*it);
//  }
//  if(dirichlet_bc_def.size()>0)
//    mesh->get_node_bc_sets()->insert(std::pair<int_t,std::vector<int_t> >(0,dirichlet_bc_def)); // all dirichlet boundary nodes go in node set 0
//  std::vector<int_t> neumann_bc_def;
//  it = neumann_boundary_nodes.begin();
//  it_end = neumann_boundary_nodes.end();
//  for(;it!=it_end;++it){
//    neumann_bc_def.push_back(*it);
//  }
//  if(neumann_bc_def.size()>0)
//    mesh->get_node_bc_sets()->insert(std::pair<int_t,std::vector<int_t> >(1,neumann_bc_def)); // all neumann boundary nodes go in node set 1
//
  mesh->set_initialized();

  // initialize the fields needed for coordinates etc:
  mesh->create_field(field_enums::INITIAL_COORDINATES_FS);
  Teuchos::RCP<MultiField > initial_coords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);

  for(int_t i=0;i<num_nodes;++i)  // i represents the local_id
  {
    initial_coords->local_value(i*num_dim+0) = coords_x.find(local_to_master_node_map[i])->second;
    initial_coords->local_value(i*num_dim+1) = coords_y.find(local_to_master_node_map[i])->second;
  }

  // export the coordinates back to the non-overlap field
  mesh->field_overlap_export(initial_coords, field_enums::INITIAL_COORDINATES_FS, INSERT);
  //std::cout << "INITIAL COORDS: " << std::endl;
  //initial_coords_ptr->vec()->describe();

  // CELL COORDINATES
  mesh->create_field(field_enums::INITIAL_CELL_COORDINATES_FS);

  Teuchos::RCP<MultiField > mcoords = mesh->get_overlap_field(field_enums::INITIAL_COORDINATES_FS);
  MultiField & initial_cell_coords = *mesh->get_field(field_enums::INITIAL_CELL_COORDINATES_FS);

  //compute the centroid from the coordinates of the nodes;
  for(elem_it=mesh->get_element_set()->begin();elem_it!=elem_end;++elem_it)
  {
    const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
    scalar_t centroid[num_dim];
    for(int_t i=0;i<num_dim;++i) centroid[i]=0.0;
    for(size_t node_it=0;node_it<connectivity.size();++node_it)
    {
      const Teuchos::RCP<DICe::mesh::Node> node = connectivity[node_it];
      centroid[0] += mcoords->local_value(node.get()->overlap_local_id()*spa_dim+0);
      centroid[1] += mcoords->local_value(node.get()->overlap_local_id()*spa_dim+1);
    }
    for(int_t dim=0;dim<num_dim;++dim)
    {
      centroid[dim] /= connectivity.size();
      initial_cell_coords.local_value(elem_it->get()->local_id()*num_dim+dim) = centroid[dim];
    }
  }
  //std::cout << " INITIAL CELL COORDS: " << std::endl;
  //initial_cell_coords.vec()->describe();

  // PROCESSOR ID

  // create the cell_coords, cell_radius and cell_size fields:
  mesh->create_field(field_enums::PROCESSOR_ID_FS);
  MultiField & proc_id = *mesh->get_field(field_enums::PROCESSOR_ID_FS);
  elem_it = mesh->get_element_set()->begin();
  for(;elem_it!=elem_end;++elem_it){
    proc_id.local_value(elem_it->get()->local_id()) = p_rank;
  }

  DEBUG_MSG("  ------------------ Analysis Model Definition (processsor " << p_rank <<") --------------------------");
  DEBUG_MSG("  Title:             " << "autogenerated linear Triangle mesh");
  DEBUG_MSG("  Output file:       " << output_filename);
  DEBUG_MSG("  Spatial dimension: " << num_dim);
  DEBUG_MSG("  Nodes:             " << num_nodes);
  DEBUG_MSG("  Elements:          " << tri6_mesh->num_elem());
  DEBUG_MSG("  --------------------------------------------------------------------------------------");

  delete[] local_to_master_node_map;

  return mesh;
}


} // namespace mesh

DICE_LIB_DLL_EXPORT
void tri3d_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){
  TEUCHOS_TEST_FOR_EXCEPTION(order < 0 || order > 5,std::runtime_error,"");
  const int_t spa_dim = 3;
  // Dunavant quadrature:
  if(order==1){
    num_points = 1;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);

    locations[0][0] = 0.333333333333333; locations[0][1] = 0.333333333333333; locations[0][2] = 0.333333333333333;
    weights[0] = 1.000000000000000;
  }
  else if(order==2){
    num_points = 3;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);
    locations[0][0] = 0.666666666666667; locations[0][1] = 0.166666666666667; locations[0][2] = 0.166666666666667;
    locations[1][0] = 0.166666666666667; locations[1][1] = 0.666666666666667; locations[1][2] = 0.166666666666667;
    locations[2][0] = 0.166666666666667; locations[2][1] = 0.166666666666667; locations[2][2] = 0.666666666666667;
    weights[0] = 0.333333333333333;
    weights[1] = 0.333333333333333;
    weights[2] = 0.333333333333333;
  }
  else if(order==3){
    num_points = 4;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);
    locations[0][0] = 0.333333333333333; locations[0][1] = 0.333333333333333; locations[0][2] = 0.333333333333333;
    locations[1][0] = 0.600000000000000; locations[1][1] = 0.200000000000000; locations[1][2] = 0.200000000000000;
    locations[2][0] = 0.200000000000000; locations[2][1] = 0.600000000000000; locations[2][2] = 0.200000000000000;
    locations[3][0] = 0.200000000000000; locations[3][1] = 0.200000000000000; locations[3][2] = 0.600000000000000;
    weights[0] = -0.562500000000000;
    weights[1] = 0.520833333333333;
    weights[2] = 0.520833333333333;
    weights[3] = 0.520833333333333;
  }
  else if(order==4){
    num_points = 6;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);
    locations[0][0] = 0.108103018168070; locations[0][1] = 0.445948490915965; locations[0][2] = 0.445948490915965;
    locations[1][0] = 0.445948490915965; locations[1][1] = 0.108103018168070; locations[1][2] = 0.445948490915965;
    locations[2][0] = 0.445948490915965; locations[2][1] = 0.445948490915965; locations[2][2] = 0.108103018168070;
    locations[3][0] = 0.816847572980459; locations[3][1] = 0.091576213509771; locations[3][2] = 0.091576213509771;
    locations[4][0] = 0.091576213509771; locations[4][1] = 0.816847572980459; locations[4][2] = 0.091576213509771;
    locations[5][0] = 0.091576213509771; locations[5][1] = 0.091576213509771; locations[5][2] = 0.816847572980459;
    weights[0] = 0.223381589678011;
    weights[1] = 0.223381589678011;
    weights[2] = 0.223381589678011;
    weights[3] = 0.109951743655322;
    weights[4] = 0.109951743655322;
    weights[5] = 0.109951743655322;
  }
  else if(order==5){
    num_points = 7;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);
    locations[0][0] = 0.333333333333333; locations[0][1] = 0.333333333333333; locations[0][2] = 0.333333333333333;
    locations[1][0] = 0.059715871789770; locations[1][1] = 0.470142064105115; locations[1][2] = 0.470142064105115;
    locations[2][0] = 0.470142064105115; locations[2][1] = 0.059715871789770; locations[2][2] = 0.470142064105115;
    locations[3][0] = 0.470142064105115; locations[3][1] = 0.470142064105115; locations[3][2] = 0.059715871789770;
    locations[4][0] = 0.797426985353087; locations[4][1] = 0.101286507323456; locations[4][2] = 0.101286507323456;
    locations[5][0] = 0.101286507323456; locations[5][1] = 0.797426985353087; locations[5][2] = 0.101286507323456;
    locations[6][0] = 0.101286507323456; locations[6][1] = 0.101286507323456; locations[6][2] = 0.797426985353087;
    weights[0] = 0.225000000000000;
    weights[1] = 0.132394152788506;
    weights[2] = 0.132394152788506;
    weights[3] = 0.132394152788506;
    weights[4] = 0.125939180544827;
    weights[5] = 0.125939180544827;
    weights[6] = 0.125939180544827;
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid pixel integration order.");
  }
}

DICE_LIB_DLL_EXPORT
void tri2d_nonexact_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){

  // note 0.5 factor for being a triangle is included in the gauss weight
  TEUCHOS_TEST_FOR_EXCEPTION(order<=0,std::runtime_error,"");
  const int_t spa_dim = 2;
  //std::cout << "ORDER " << order << std::endl;
  const scalar_t h = 1.0 / order;
  //std::cout << "H " << h << std::endl;
  // determine the number of points
  num_points = 0;
  for(int_t i=0;i<=order;++i){
    num_points += i;
  }
  //std::cout << "NUM_POINTS " << num_points << std::endl;
  TEUCHOS_TEST_FOR_EXCEPTION(num_points<=0,std::runtime_error,"");
  const scalar_t weight = 0.5 / (num_points-0.5*order); // There are order number half cells along the diag

  // resize the storage
  locations.resize(num_points);
  for(int_t i=0;i<num_points;++i)
    locations[i].resize(spa_dim);
  weights.resize(num_points);

  int_t index = 0;
  for(int_t j=0;j<order;++j){
    scalar_t y = 0.5*h + j*h;
    for(int_t i=0;i<=order-j-1;++i){
      scalar_t x = 0.5*h + i*h;
      locations[index][0] = x;
      locations[index][1] = y;
      weights[index] = i==order-j-1?0.5*weight:weight;
      index++;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(index!=num_points,std::runtime_error,"");
}

DICE_LIB_DLL_EXPORT
void tri2d_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points){
  TEUCHOS_TEST_FOR_EXCEPTION(order < -7 || order > 7,std::runtime_error,"");
  const int_t spa_dim = 2;
  // Dunavant quadrature:
  if(order==1){
    num_points = 1;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);

    locations[0][0] = 0.333333333333333; locations[0][1] = 0.333333333333333;
    weights[0] = 0.5;
  }
  else if(order==-2){
    num_points = 3;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);
    locations[0][0] = 0.5; locations[0][1] = 0.0;
    locations[1][0] = 0.0; locations[1][1] = 0.5;
    locations[2][0] = 0.5; locations[2][1] = 0.5;
    weights[0] = 0.1666666666666667; // weights sum to 1/2 because that is the area of the iso-tri
    weights[1] = 0.1666666666666667;
    weights[2] = 0.1666666666666667;
  }
  else if(order==2){
    num_points = 3;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);
    locations[0][0] = 0.1666666666666667; locations[0][1] = 0.1666666666666667;
    locations[1][0] = 0.6666666666666667; locations[1][1] = 0.1666666666666667;
    locations[2][0] = 0.1666666666666667; locations[2][1] = 0.6666666666666667;
    weights[0] = 0.1666666666666667; // weights sum to 1/2 because that is the area of the iso-tri
    weights[1] = 0.1666666666666667;
    weights[2] = 0.1666666666666667;
  }
  else if(order==3){
    num_points = 4;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);
    locations[0][0] = 0.333333333333333; locations[0][1] = 0.333333333333333;
    locations[1][0] = 0.600000000000000; locations[1][1] = 0.200000000000000;
    locations[2][0] = 0.200000000000000; locations[2][1] = 0.600000000000000;
    locations[3][0] = 0.200000000000000; locations[3][1] = 0.200000000000000;
    weights[0] = -0.281250000000000;// weights sum to 1/2 because that is the area of the iso-tri
    weights[1] = 0.260416666666667;
    weights[2] = 0.260416666666667;
    weights[3] = 0.260416666666667;
  }
  else if(order==4){
    num_points = 6;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);

    locations[0][0] = 0.44594849091597; locations[0][1] = 0.44594849091597;
    locations[1][0] = 0.44594849091597; locations[1][1] = 0.10810301816807;
    locations[2][0] = 0.10810301816807; locations[2][1] = 0.44594849091597;
    locations[3][0] = 0.09157621350977; locations[3][1] = 0.09157621350977;
    locations[4][0] = 0.09157621350977; locations[4][1] = 0.81684757298046;
    locations[5][0] = 0.81684757298046; locations[5][1] = 0.09157621350977;
    weights[0] = 0.11169079483901;// weights sum to 1/2 because that is the area of the iso-tri
    weights[1] = 0.11169079483901;
    weights[2] = 0.11169079483901;
    weights[3] = 0.05497587182766;
    weights[4] = 0.05497587182766;
    weights[5] = 0.05497587182766;
  }
  else if(order==5){
    num_points = 7;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);

    locations[0][0] = 0.333333333333333; locations[0][1] = 0.333333333333333;
    locations[1][0] = 0.797426985353087; locations[1][1] = 0.101286507323456;
    locations[2][0] = 0.101286507323456; locations[2][1] = 0.797426985353087;
    locations[3][0] = 0.101286507323456; locations[3][1] = 0.101286507323456;
    locations[4][0] = 0.470142064105115; locations[4][1] = 0.059715871789770;
    locations[5][0] = 0.059715871789770; locations[5][1] = 0.470142064105115;
    locations[6][0] = 0.470142064105115; locations[6][1] = 0.470142064105115;
    weights[0] = 0.112500000000000;// weights sum to 1/2 because that is the area of the iso-tri
    weights[1] = 0.06296959027241;
    weights[2] = 0.06296959027241;
    weights[3] = 0.06296959027241;
    weights[4] = 0.06619707639425;
    weights[5] = 0.06619707639425;
    weights[6] = 0.06619707639425;
  }
  else if(order==6){
    num_points = 12;
    locations.resize(num_points);
    for(int_t i=0;i<num_points;++i)
      locations[i].resize(spa_dim);
    weights.resize(num_points);

    locations[0][0] = 0.24928674517091; locations[0][1] = 0.24928674517091;
    locations[1][0] = 0.24928674517091; locations[1][1] = 0.50142650965818;
    locations[2][0] = 0.50142650965818; locations[2][1] = 0.24928674517091;
    locations[3][0] = 0.06308901449150; locations[3][1] = 0.06308901449150;
    locations[4][0] = 0.06308901449150; locations[4][1] = 0.87382197101700;
    locations[5][0] = 0.87382197101700; locations[5][1] = 0.06308901449150;
    locations[6][0] = 0.31035245103378; locations[6][1] = 0.63650249912140;
    locations[7][0] = 0.63650249912140; locations[7][1] = 0.05314504984482;
    locations[8][0] = 0.05314504984482; locations[8][1] = 0.31035245103378;
    locations[9][0] = 0.63650249912140; locations[9][1] = 0.31035245103378;
    locations[10][0] = 0.31035245103378; locations[10][1] = 0.05314504984482;
    locations[11][0] = 0.05314504984482; locations[11][1] = 0.63650249912140;
    weights[0] = 0.05839313786319;// weights sum to 1/2 because that is the area of the iso-tri
    weights[1] = 0.05839313786319;
    weights[2] = 0.05839313786319;
    weights[3] = 0.02542245318511;
    weights[4] = 0.02542245318511;
    weights[5] = 0.02542245318511;
    weights[6] = 0.04142553780919;
    weights[7] = 0.04142553780919;
    weights[8] = 0.04142553780919;
    weights[9] = 0.04142553780919;
    weights[10] = 0.04142553780919;
    weights[11] = 0.04142553780919;
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error: invalid pixel integration order.");
  }
}

//Compute the cross product AB x AC
DICE_LIB_DLL_EXPORT
scalar_t cross(const scalar_t * a_coords,
  const scalar_t * b_coords,
  const scalar_t * c_coords)
{
  scalar_t AB[2];
  scalar_t AC[2];
  AB[0] = b_coords[0]-a_coords[0];
  AB[1] = b_coords[1]-a_coords[1];
  AC[0] = c_coords[0]-a_coords[0];
  AC[1] = c_coords[1]-a_coords[1];
  const scalar_t cross = AB[0] * AC[1] - AB[1] * AC[0];
  return cross;
}
//Compute the cross product AB x AC
DICE_LIB_DLL_EXPORT
scalar_t cross3d(const scalar_t * a_coords,
  const scalar_t * b_coords,
  const scalar_t * c_coords)
{
  scalar_t AB[3];
  scalar_t AC[3];
  AB[0] = b_coords[0]-a_coords[0];
  AB[1] = b_coords[1]-a_coords[1];
  AB[2] = b_coords[2]-a_coords[2];
  AC[0] = c_coords[0]-a_coords[0];
  AC[1] = c_coords[1]-a_coords[1];
  AC[2] = c_coords[2]-a_coords[2];
  const scalar_t cross_0 = AB[1] * AC[2] - AC[1] * AB[2];
  const scalar_t cross_1 = AC[0] * AB[2] - AB[0] * AC[2];
  const scalar_t cross_2 = AB[0] * AC[1] - AB[1] * AC[0];
  return std::sqrt(cross_0 * cross_0 + cross_1 * cross_1 + cross_2 * cross_2);
}

//Compute the cross product AB x AC return the area of the parallelogram and update the normal pointer
DICE_LIB_DLL_EXPORT
scalar_t cross3d_with_normal(const scalar_t * a_coords,
  const scalar_t * b_coords,
  const scalar_t * c_coords,
  scalar_t * normal)
{
  scalar_t AB[3];
  scalar_t AC[3];
  AB[0] = b_coords[0]-a_coords[0];
  AB[1] = b_coords[1]-a_coords[1];
  AB[2] = b_coords[2]-a_coords[2];
  AC[0] = c_coords[0]-a_coords[0];
  AC[1] = c_coords[1]-a_coords[1];
  AC[2] = c_coords[2]-a_coords[2];
  const scalar_t cross_0 = AB[1] * AC[2] - AC[1] * AB[2];
  const scalar_t cross_1 = AC[0] * AB[2] - AB[0] * AC[2];
  const scalar_t cross_2 = AB[0] * AC[1] - AB[1] * AC[0];

  const scalar_t mag = std::sqrt(cross_0 * cross_0 + cross_1 * cross_1 + cross_2 * cross_2);

  normal[0] = cross_0/mag;
  normal[1] = cross_1/mag;
  normal[2] = cross_2/mag;

  return mag;
}

//Compute the cross product AB x AC return the area of the parallelogram and update the normal pointer
//TODO condense these three functions to use the same subfunction
DICE_LIB_DLL_EXPORT
void cross3d_with_cross_prod(const scalar_t * a_coords,
  const scalar_t * b_coords,
  const scalar_t * c_coords,
  scalar_t * cross_prod)
{
  scalar_t AB[3];
  scalar_t AC[3];
  AB[0] = b_coords[0]-a_coords[0];
  AB[1] = b_coords[1]-a_coords[1];
  AB[2] = b_coords[2]-a_coords[2];
  AC[0] = c_coords[0]-a_coords[0];
  AC[1] = c_coords[1]-a_coords[1];
  AC[2] = c_coords[2]-a_coords[2];
  cross_prod[0] = AB[1] * AC[2] - AC[1] * AB[2];
  cross_prod[1] = AC[0] * AB[2] - AB[0] * AC[2];
  cross_prod[2] = AB[0] * AC[1] - AB[1] * AC[0];
}

DICE_LIB_DLL_EXPORT
scalar_t determinant_4x4(const scalar_t * a)
{
  const scalar_t det =
      a[0+0*4] * (
          a[1+1*4] * ( a[2+2*4] * a[3+3*4] - a[2+3*4] * a[3+2*4] )
          - a[1+2*4] * ( a[2+1*4] * a[3+3*4] - a[2+3*4] * a[3+1*4] )
          + a[1+3*4] * ( a[2+1*4] * a[3+2*4] - a[2+2*4] * a[3+1*4] ) )
          - a[0+1*4] * (
              a[1+0*4] * ( a[2+2*4] * a[3+3*4] - a[2+3*4] * a[3+2*4] )
              - a[1+2*4] * ( a[2+0*4] * a[3+3*4] - a[2+3*4] * a[3+0*4] )
              + a[1+3*4] * ( a[2+0*4] * a[3+2*4] - a[2+2*4] * a[3+0*4] ) )
              + a[0+2*4] * (
                  a[1+0*4] * ( a[2+1*4] * a[3+3*4] - a[2+3*4] * a[3+1*4] )
                  - a[1+1*4] * ( a[2+0*4] * a[3+3*4] - a[2+3*4] * a[3+0*4] )
                  + a[1+3*4] * ( a[2+0*4] * a[3+1*4] - a[2+1*4] * a[3+0*4] ) )
                  - a[0+3*4] * (
                      a[1+0*4] * ( a[2+1*4] * a[3+2*4] - a[2+2*4] * a[3+1*4] )
                      - a[1+1*4] * ( a[2+0*4] * a[3+2*4] - a[2+2*4] * a[3+0*4] )
                      + a[1+2*4] * ( a[2+0*4] * a[3+1*4] - a[2+1*4] * a[3+0*4] ) );
  return det;
}

DICE_LIB_DLL_EXPORT
void gauss_1D(Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & r,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & w,
  int_t gauss_order){

  r.clear();
  w.clear();
  r.resize(gauss_order);
  w.resize(gauss_order);
  for(int_t i=0;i<gauss_order;++i){
    r[i].resize(1);
    w[i].resize(1);
  }

  switch(gauss_order){
  case 1:
    r[0][0] = 0.0;
    w[0][0] = 2.0;
    break;
  case 2:
    r[0][0] = -1.0/std::sqrt(3.0);
    r[1][0] = -1.0*r[0][0];
    w[0][0] = 1.0;
    w[1][0] = 1.0;
    break;
  case 3:
    r[0][0] = -1.0*std::sqrt(0.6);
    r[1][0] = 0.0;
    r[2][0] = -1.0*r[0][0];
    w[0][0] = 5.0/9.0;
    w[1][0] = 8.0/9.0;
    w[2][0] = 5.0/9.0;
    break;
  case 4:
    r[0][0] = -0.861136311594053;
    r[1][0] = -0.339981043584856;
    r[2][0] =  0.339981043584856;
    r[3][0] =  0.861136311594053;
    //
    w[0][0] = 0.347854845137454;
    w[1][0] = 0.652145154862546;
    w[2][0] = 0.652145154862546;
    w[3][0] = 0.347854845137454;
    break;
  case 5:
    r[0][0] = -0.906179845938664;
    r[1][0] = -0.538469310105683;
    r[2][0] =  0.000000000000000;
    r[3][0] =  0.538469310105683;
    r[4][0] =  0.906179845938664;
    //
    w[0][0] =  0.236926885056189;
    w[1][0] =  0.478628670499366;
    w[2][0] =  0.568888888888889;
    w[3][0] =  0.478628670499366;
    w[4][0] =  0.236926885056189;
    break;
  case 6:
    r[0][0] = -0.932469514203152;
    r[1][0] = -0.661209386466265;
    r[2][0] = -0.238619186083197;
    r[3][0] =  0.238619186083197;
    r[4][0] =  0.661209386466265;
    r[5][0] =  0.932469514203152;
    //
    w[0][0] =  0.171324492379170;
    w[1][0] =  0.360761573048139;
    w[2][0] =  0.467913934572691;
    w[3][0] =  0.467913934572691;
    w[4][0] =  0.360761573048139;
    w[5][0] =  0.171324492379170;
    break;
  case 7:
    r[0][0] = -0.949107912342759;
    r[1][0] = -0.741531185599394;
    r[2][0] = -0.405845151377397;
    r[3][0] =  0.000000000000000;
    r[4][0] =  0.405845151377397;
    r[5][0] =  0.741531185599394;
    r[6][0] =  0.949107912342759;
    //
    w[0][0] =  0.129484966168870;
    w[1][0] =  0.279705391489277;
    w[2][0] =  0.381830050505119;
    w[3][0] =  0.417959183673469;
    w[4][0] =  0.381830050505119;
    w[5][0] =  0.279705391489277;
    w[6][0] =  0.129484966168870;
    break;
  case 8:
    r[0][0] = -0.960289856497536;
    r[1][0] = -0.796666477413627;
    r[2][0] = -0.525532409916329;
    r[3][0] = -0.183434642495650;
    r[4][0] =  0.183434642495650;
    r[5][0] =  0.525532409916329;
    r[6][0] =  0.796666477413627;
    r[7][0] =  0.960289856497536;
    //
    w[0][0] =  0.101228536290376;
    w[1][0] =  0.222381034453374;
    w[2][0] =  0.313706645877887;
    w[3][0] =  0.362683783378362;
    w[4][0] =  0.362683783378362;
    w[5][0] =  0.313706645877887;
    w[6][0] =  0.222381034453374;
    w[7][0] =  0.101228536290376;
    break;
  case 9:
    r[0][0] = -0.968160239507626;
    r[1][0] = -0.836031170326636;
    r[2][0] = -0.613371432700590;
    r[3][0] = -0.324253423403809;
    r[4][0] =  0.000000000000000;
    r[5][0] =  0.324253423403809;
    r[6][0] =  0.613371432700590;
    r[7][0] =  0.836031107326636;
    r[8][0] =  0.968160239507626;
    //
    w[0][0] =  0.081274388361574;
    w[1][0] =  0.180648160694857;
    w[2][0] =  0.260610696402935;
    w[3][0] =  0.312347077040003;
    w[4][0] =  0.330239355001260;
    w[5][0] =  0.312347077040003;
    w[6][0] =  0.260610696402935;
    w[7][0] =  0.180648160694857;
    w[8][0] =  0.081274388361574;
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
    break;
  }
}

DICE_LIB_DLL_EXPORT
void gauss_2D(Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & r,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & w,
  int_t gauss_order){

  r.clear();
  w.clear();
  r.resize(gauss_order*gauss_order);
  w.resize(gauss_order*gauss_order);
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > kron_sol, zeta1,weight1;
  kron_sol.resize(gauss_order*gauss_order);
  for(int_t i=0;i<gauss_order*gauss_order;++i){
    r[i].resize(2);
    w[i].resize(1);
    kron_sol[i].resize(1);
  }
  gauss_1D(zeta1, weight1, gauss_order);

  int_t i=0,j=0;
  int_t m = gauss_order; int_t n = 1;
  //  create a matrix of ones
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > ones;
  ones.resize(m);
  for(int_t i=0;i<m;++i)
    ones[i].resize(n);
  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      ones[i][j] = 1.0;
    };
  };

  kron_sol = kronecker(ones,zeta1);

  for(i=0;i<gauss_order*gauss_order;i++){
    r[i][1] = kron_sol[i][0];
  };

  for(i=0;i<gauss_order;i++){
    for(j=i*gauss_order; j< (i+1)*gauss_order;j++){
      r[j][0] = zeta1[i][0];
    };
  };

  w = kronecker(weight1,weight1);
}


DICE_LIB_DLL_EXPORT
Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > kronecker( Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & A,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & B){

 int_t B_i = 0;
 int_t B_j = 0;

 int_t nAx = A.size();
 int_t nAy = A[0].size();
 int_t nBx = B.size();
 int_t nBy = B[0].size();

 Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > C;
 C.resize(nAx*nBx);
 for(int_t i=0;i<nAx*nBx;++i)
   C[i].resize(nAy*nBy);

 for(int_t i = 0; i<nAx;i++){
   for(int_t j = 0; j< nAy; j++){   // go through each entry in A
     for(int_t n = i*nBx; n < (i+1)*nBx; n++){
       B_i = n - i* nBx;
       for(int_t m =j*nBy; m< (j+1)*nBy; m++){
         B_j = m - j*nBy;
         C[n][m] = A[i][j] * B[B_i][B_j];
       };
     };
   };
 };
 return C;
};

DICE_LIB_DLL_EXPORT
void calc_jacobian(const scalar_t * xcap,
  const scalar_t * DN,
  scalar_t * jacobian,
  scalar_t * inv_jacobian,
  scalar_t & J,
  int_t num_elem_nodes,
  int_t dim ){

  for(int_t i=0;i<dim*dim;++i){
    jacobian[i] = 0.0;
    inv_jacobian[i] = 0.0;
  }
  J = 0.0;

  for(int_t i=0;i<dim;++i)
    for(int_t j=0;j<dim;++j)
      for(int_t k=0;k<num_elem_nodes;++k)
        jacobian[i*dim+j] += xcap[k*dim + i] * DN[k*dim+j];
//  std::cout << " jacobian " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<dim;++j)
//      std::cout << " " << jacobian[i*dim+j];
//    std::cout << std::endl;
//  }

  if(dim==2){
    J = jacobian[0]*jacobian[3] - jacobian[1]*jacobian[2];
    TEUCHOS_TEST_FOR_EXCEPTION(J<=0.0,std::runtime_error,"Error: determinant 0.0 encountered or negative det");
    inv_jacobian[0] =  jacobian[3] / J;
    inv_jacobian[1] = -jacobian[1] / J;
    inv_jacobian[2] = -jacobian[2] / J;
    inv_jacobian[3] =  jacobian[0] / J;
  }
  else if(dim==3){
    J =   jacobian[0]*jacobian[4]*jacobian[8] + jacobian[1]*jacobian[5]*jacobian[6] + jacobian[2]*jacobian[3]*jacobian[7]
        - jacobian[6]*jacobian[4]*jacobian[2] - jacobian[7]*jacobian[5]*jacobian[0] - jacobian[8]*jacobian[3]*jacobian[1];
    TEUCHOS_TEST_FOR_EXCEPTION(J<=0.0,std::runtime_error,"Error: determinant 0.0 encountered or negative det");
    inv_jacobian[0] = ( -jacobian[5]*jacobian[7] + jacobian[4]*jacobian[8]) /  J;
    inv_jacobian[1] = (  jacobian[2]*jacobian[7] - jacobian[1]*jacobian[8]) /  J;
    inv_jacobian[2] = ( -jacobian[2]*jacobian[4] + jacobian[1]*jacobian[5]) /  J;
    inv_jacobian[3] = (  jacobian[5]*jacobian[6] - jacobian[3]*jacobian[8]) /  J;
    inv_jacobian[4] = ( -jacobian[2]*jacobian[6] + jacobian[0]*jacobian[8]) /  J;
    inv_jacobian[5] = (  jacobian[2]*jacobian[3] - jacobian[0]*jacobian[5]) /  J;
    inv_jacobian[6] = ( -jacobian[4]*jacobian[6] + jacobian[3]*jacobian[7]) /  J;
    inv_jacobian[7] = (  jacobian[1]*jacobian[6] - jacobian[0]*jacobian[7]) /  J;
    inv_jacobian[8] = ( -jacobian[1]*jacobian[3] + jacobian[0]*jacobian[4]) /  J;
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");

//  std::cout << " J " << J << std::endl;
//  std::cout << " inv_jacobian " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<dim;++j)
//      std::cout << " " << inv_jacobian[i*dim+j];
//    std::cout << std::endl;
//  }

};

DICE_LIB_DLL_EXPORT
void calc_B(const scalar_t * DN,
  const scalar_t * inv_jacobian,
  int_t num_elem_nodes,
  int_t dim,
  scalar_t * solid_B){

  scalar_t * dN = new scalar_t[dim*num_elem_nodes];
  for(int_t i=0;i<dim*num_elem_nodes;++i)
    dN[i] = 0.0;

//  std::cout << " DN " << std::endl;
//  for(int_t j=0;j<num_elem_nodes;++j){
//    for(int_t i=0;i<dim;++i)
//      std::cout << " " << DN[j*dim+i];
//    std::cout << std::endl;
//  }


  // compute j_inv_transpose * DN_transpose
  for(int_t i=0;i<dim;++i)
    for(int_t j=0;j<num_elem_nodes;++j)
      for(int_t k=0;k<dim;++k)
        dN[i*num_elem_nodes+j] += inv_jacobian[k*dim + i] * DN[j*dim+k];

//  std::cout << " dN " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<num_elem_nodes;++j)
//      std::cout << " " << dN[i*num_elem_nodes+j];
//    std::cout << std::endl;
//  }

  const int_t stride = dim*num_elem_nodes;
  int_t placer = 0;
  if(dim ==3){
    for(int_t i=0;i<6*dim*num_elem_nodes;++i)
      solid_B[i] = 0.0;
    for(int_t i=0;i<num_elem_nodes;i++){
      placer = i*dim;     // hold the place in B for each node
      solid_B[0*stride + placer + 0] = dN[0*num_elem_nodes + i];
      solid_B[1*stride + placer + 1] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 2] = dN[2*num_elem_nodes + i];
      solid_B[3*stride + placer + 0] = dN[1*num_elem_nodes + i];
      solid_B[3*stride + placer + 1] = dN[0*num_elem_nodes + i];
      solid_B[4*stride + placer + 1] = dN[2*num_elem_nodes + i];
      solid_B[4*stride + placer + 2] = dN[1*num_elem_nodes + i];
      solid_B[5*stride + placer + 0] = dN[2*num_elem_nodes + i];
      solid_B[5*stride + placer + 2] = dN[0*num_elem_nodes + i];
    };
  };

  if(dim==2){
    for(int_t i=0;i<3*dim*num_elem_nodes;++i)
      solid_B[i] = 0.0;
    for(int_t i=0;i<=num_elem_nodes;i++){
      placer = i*dim;     // hold the place in B for each node
      solid_B[0*stride + placer + 0] = dN[0*num_elem_nodes + i];
      solid_B[1*stride + placer + 1] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 0] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 1] = dN[0*num_elem_nodes + i];
    };
  };

//  std::cout << " B " << std::endl;
//  for(int_t i=0;i<6;++i){
//    for(int_t j=0;j<num_elem_nodes*dim;++j)
//      std::cout << " " << solid_B[i*(num_elem_nodes*dim)+j];
//    std::cout << std::endl;
//  }
  delete [] dN;
}




} // DICe
