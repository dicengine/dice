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

#ifndef DICE_MESH_H_
#define DICE_MESH_H_

#include <DICe.h>
#include <DICe_MeshEnums.h>
#include <DICe_Parser.h>
#ifdef DICE_TPETRA
  #include "DICe_MultiFieldTpetra.h"
#else
  #include "DICe_MultiFieldEpetra.h"
#endif

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

namespace DICe{
/*!
 *  \namespace DICe::mesh
 *  @{
 */
/// computational mesh utilities associated with the global DIC method and physics
namespace mesh{

/// forward declaration
class Node;
/// forward declaration
class Element;
/// forward declaration
class Edge;
// CVFEM objects
/// forward declaration
class Subelement;
/// forward declaration
class Bond;
/// forward declaration
class Internal_Face_Edge;
/// forward declaration
class External_Face_Edge;
/// forward declaration
class Internal_Cell;
/// forward declaration
class Mesh_Object;
/// typedef
typedef std::set<int_t> ordinal_set;
/// typedef
typedef std::vector<Teuchos::RCP<Node> > connectivity_vector;
/// typedef
typedef std::map<int_t, connectivity_vector > conn_map;
/// typedef
typedef std::map<int_t,Base_Element_Type> block_type_map;
/// typedef
typedef std::map<field_enums::Entity_Rank, std::set<int_t> > shallow_relations_map;
/// typedef
typedef std::map<field_enums::Entity_Rank, std::vector<Teuchos::RCP<Mesh_Object> > > deep_relations_map;
/// typedef
typedef std::map<const field_enums::Field_Spec,const Teuchos::RCP<MultiField> > field_registry;
/// typedef
typedef std::map<int_t,std::vector<int_t> > bc_set;

/// \class Mesh_Object
/// \brief A mesh object is the base class for elements, nodes, faces, that make up the computational domain
class Mesh_Object
{
public:
  /// \brief Basic and only constructor
  /// \param global_id the global id of the mesh object
  /// \param local_id the local id of the mesh object
  Mesh_Object(const int_t global_id,
    const int_t local_id);

  /// destructor
  virtual ~Mesh_Object(){};

  /// Returns the global id of the mesh object
  int_t global_id()const{
    return global_id_;
  }

  /// Returns the local id of the mesh object
  int_t local_id()const{
    return local_id_;
  }

  /// Returns the local overlap id of mesh objects that are shared accross processors
  int_t overlap_local_id()const{
    return overlap_local_id_;
  }

  /// Returns the local neighbor overlap id (this overlap map contains neighbor relations that are shared
  /// accross processors as well)
  int_t overlap_neighbor_local_id()const{
    return overlap_neighbor_local_id_;
  }

  /// Less than operator for creating maps or sets of mesh objects
  /// \param right The mesh object to compre to
  bool operator< (const Mesh_Object & right) const{
    return global_id()< right.global_id();
  }

  /// Returns a pointer to the set of shallow relations (id of relation only)
  /// \param entity_rank The rank of the relations to gather
  const std::set<int_t> * shallow_relations(const field_enums::Entity_Rank entity_rank) const {
    return &(shallow_relations_map_.find(entity_rank)->second);
  }

  /// Returns a pointer to the set of deep relations (actual pointers to the objects themselves, not just the id as in shallow relations)
  /// \param entity_rank The rank of the relations to gather
  const std::vector<Teuchos::RCP<Mesh_Object> > * deep_relations(const field_enums::Entity_Rank entity_rank) const {
    return &(deep_relations_map_.find(entity_rank)->second);
  }

  /// Returns a pointer to the map of shallow relations for this mesh object
  shallow_relations_map * get_shallow_relations_map(){
    return &shallow_relations_map_;
  }

  /// Returns a pointer to the map of deep relations for this mesh object
  deep_relations_map * get_deep_relations_map(){
    return &deep_relations_map_;
  }

  /// Returns a pointer to the shallow elem relations vector
  // TODO find out why there are so many ways to get relations and delete the unneeded ones
  std::vector<int_t> * get_shallow_elem_relations(){
    return &shallow_elem_relations_;
  }

  /// Returns a pointer to the bond deep relations
  std::vector<Teuchos::RCP<Bond> > * get_deep_bond_relations(){
    return &deep_bond_relations_;
  }

  /// \brief Add a shallow relation to this mesh object
  /// \param entity_rank The entity rank determines which list to add the relation to
  /// \param global_id The global id of the mesh object to add as a relation
  ///
  /// Only save off the global id of the relation and nothing else.
  /// Good for off processor elements
  void add_shallow_relation(const field_enums::Entity_Rank entity_rank,
    const int_t global_id)const;

  /// \brief Add a deep relation for the given mesh object
  /// \param element The mesh object to add a deep relation for
  ///
  /// Note: Be careful with deep relations, the relations need to be local to this process
  /// and the mesh object can be altered through the relations
  void add_deep_relation(const Teuchos::RCP<Element> element)const;

  /// \brief Add a deep relation for the given mesh object
  /// \param node The mesh object to add a deep relation for
  void add_deep_relation(const Teuchos::RCP<Node> node)const;

  /// \brief Add a deep relation for the given mesh object
  /// \param internal_face_edge The mesh object to add a deep relation for
  void add_deep_relation(const Teuchos::RCP<Internal_Face_Edge> internal_face_edge)const;

//  /// \brief Add a deep relation for the given mesh object
//  /// \param internal_face_edge The mesh object to add a deep relation for
//  void add_deep_relation(const Teuchos::RCP<External_Face_Edge> external_face_edge)const;

  /// \brief Add a deep relation for the given mesh object
  /// \param internal_cell The mesh object to add a deep relation for
  void add_deep_relation(const Teuchos::RCP<Internal_Cell> internal_cell)const;

  /// \brief Add a deep relation for the given mesh object
  /// \param bond The mesh object to add a deep relation for
  void add_deep_relation(const Teuchos::RCP<Bond> bond)const;

  /// Returns true if the given global id is a relation for this element
  /// \param gid Global id of the element in question
  /// \param sign [out] The sign of the normal vector pointing to this relation (used for fluxes)
  int_t is_elem_relation(const int_t gid,
    scalar_t & sign);

  /// Returns the number of shallow relations for this mesh object
  /// \param entity_rank The rank determines what the relation type is
  size_t num_shallow_relations(const field_enums::Entity_Rank entity_rank)const{
    return shallow_relations_map_.find(entity_rank)->second.size();
  }

  /// Returns the number of deep relations for this mesh object
  /// \param entity_rank The rank determines what the relation type is
  size_t num_deep_relations(const field_enums::Entity_Rank entity_rank)const{
    return deep_relations_map_.find(entity_rank)->second.size();
  }

  /// Returns the number of initial shallow relations for this mesh object
  /// \param entity_rank The rank determines what the relation type is
  int_t num_initial_relations(const field_enums::Entity_Rank entity_rank)const{
    TEUCHOS_TEST_FOR_EXCEPTION(entity_rank!=field_enums::ELEMENT_RANK&&entity_rank!=field_enums::NODE_RANK,std::invalid_argument,"num_inital_relations(): invalid rank");
    if(entity_rank==field_enums::ELEMENT_RANK) return initial_num_elem_relations_;
    else return initial_num_node_relations_;
  }

  /// Update the local id of this mesh object
  /// \param local_id The new id
  void update_local_id(const int_t local_id){
    local_id_ = local_id;
  }

  /// Update the overlap local id of this mesh object
  /// \param overlap_local_id The new id
  void update_overlap_local_id(const int_t overlap_local_id){
    overlap_local_id_ = overlap_local_id;
  }

  /// Update the overlap neighbor local id of this mesh object (this is the index in the shared map that includes neighbor's neighbors)
  /// \param overlap_neighbor_local_id The new id
  void update_overlap_neighbor_local_id(const int_t overlap_neighbor_local_id){
    overlap_neighbor_local_id_ = overlap_neighbor_local_id;
  }

  /// Set the number of initial relations
  /// \param num_relations The number of relations
  /// \param relation_rank The type of mesh object that this number applies to
  void set_initial_num_relations(const int_t num_relations,
    const field_enums::Entity_Rank relation_rank)const;

protected:
  /// global id
  int_t global_id_;
  /// local id
  int_t local_id_;
  /// overlap local id
  int_t overlap_local_id_;
  /// overlap neighbor local id (this is the index in the list that includes neighbor's neighbors shared accross processors)
  int_t overlap_neighbor_local_id_;
  /// Map of shallow relations (ids only not pointers to the objects)
  shallow_relations_map shallow_relations_map_;
  /// Vector of shallow elem relations
  std::vector<int_t> shallow_elem_relations_;
  /// Map of deep relations (not just ids, but actual pointers to the objects)
  deep_relations_map deep_relations_map_;
  /// Vector of pointers to bond relations
  std::vector<Teuchos::RCP<Bond> > deep_bond_relations_;
  /// Number of initial node relations
  int_t initial_num_node_relations_;
  /// Number of initial element relations
  int_t initial_num_elem_relations_;
};

/// \class Element
/// \brief Standard finite element object
class Element : public Mesh_Object
{
public:
  /// Constructor
  /// \param connectivity The connectivity that defines the element nodes
  /// \param global_id the global id to set for this element
  /// \param local_id The local id for this element
  /// \param block_id The block that this element belongs to
  /// \param block_local_id The index of the block on this processor
  Element(const connectivity_vector & connectivity,
    const int_t global_id,
    const int_t local_id,
    const int_t block_id,
    const int_t block_local_id);

  /// Destructor
  ~Element(){};

  /// Returns the block id for this object
  int_t block_id()const{
    return block_id_;
  }

  /// Returns the block local id for this object
  int_t index_in_block()const{
    return block_local_id_;
  }

  /// Returns a pointer to the connectivity for this element
  const connectivity_vector * connectivity()const{
    return &connectivity_;
  }

  /// Returns a copy of the connectivity for this element
  const connectivity_vector get_connectivity()const{
    return connectivity_;
  }

  /// Returns a list of all the elements that this element has a sensitivity to in the stiffness matrix
  std::set<int_t> * global_id_sensitivities(){
    return & global_id_sensitivities_;
  }

private:
  /// block id that this element belongs to
  int_t block_id_;
  /// index of the block that this element belongs to on this processor
  int_t block_local_id_;
  /// connectivity vector that defines the nodes that make up this element
  /// needs to be a vector not a set so it keeps its order
  connectivity_vector connectivity_;
  /// set of ids that represent all the stiffness sensitivities of this element
  std::set<int_t> global_id_sensitivities_;
};

/// \class Subelement
/// \brief Elements are divided up in to subelements for CVFEM
class Subelement : public Mesh_Object
{
public:
  /// Constructor that uses a pointer to an element to subdivide the element
  /// \param element the parent element
  /// \param local_id The local id to assign to this subelement
  /// \param global_id The global id to assign to this subelement
  Subelement(Teuchos::RCP<Element> element,
    const int_t local_id,
    const int_t global_id);

  /// Constructor that uses a pointer to an element to subdivide the element as well as the connectivity vector.
  /// In this case, the block ids, etc. may be changed from the parent element
  /// \param connectivity Connectivity vector
  /// \param element The parent element
  /// \param local_id The local id to assign to this subelement
  /// \param global_id The global id to assign to this subelement
  /// \param block_id The block id to assign to this subelement
  Subelement(const connectivity_vector & connectivity,
    Teuchos::RCP<Element> element, const int_t local_id,
    const int_t global_id,
    const int_t block_id);

  /// Destuctor
  ~Subelement(){};

  /// Returns the block id for this object
  int_t block_id()const{
    return block_id_;
  }

  /// Returns a pointer to the connectivity for this element
  const connectivity_vector * connectivity()const{
    return &connectivity_;
  }

  /// Returns a pointer to the parent element that this element derived from
  Teuchos::RCP<Element> parent_element(){
    return parent_element_;
  }

private:
  /// block id for this subelement
  int_t block_id_;
  /// connectivity vector
  connectivity_vector connectivity_;
  /// pointer to the element from which this subelement was derived
  Teuchos::RCP<Element> parent_element_;
};

/// \class Bond
/// \brief Represents a connection between two mesh objects (used mostly for peridynamics)
class Bond : public Mesh_Object
{
public:
  /// Basic constructor
  /// \param local_id The local id of this bond
  /// \param left_element The left side of the bond is connected to this element
  /// \param left_is_on_processor True if the left element is local to this processor
  /// \param right_element The right side of the bond is connected to this element
  /// \param right_is_on_processor True if the right element is local to this processor
  Bond(const int_t local_id,
    Teuchos::RCP<DICe::mesh::Element> left_element,
    const bool left_is_on_processor,
    Teuchos::RCP<DICe::mesh::Element> right_element,
    const bool right_is_on_processor):
      Mesh_Object(-1,local_id),
      left_element_(left_element),
      right_element_(right_element),
      left_is_on_processor_(left_is_on_processor),
      right_is_on_processor_(right_is_on_processor)
  {};

  /// Destructor
  ~Bond(){};

  // TODO get rid of left and right elements
  /// Returns true if the left element is on processor
  bool left_is_on_processor()const{
    return left_is_on_processor_;
  }

  /// Returns true if the right element is on processor
  bool right_is_on_processor()const{
    return right_is_on_processor_;
  }

  /// Returns the right element's global id
  int_t get_right_global_id()const{
    return right_element_->global_id();
  }

  /// Returns the left element's global id
  int_t get_left_global_id()const{
    return left_element_->global_id();
  }

  /// Returns the right element's local id
  int_t get_right_local_id()const{
    return right_element_->local_id();
  }

  /// Returns the left element's local id
  int_t get_left_local_id()const{
    return left_element_->local_id();
  }

  /// Returns the right element's overlap local id
  int_t get_right_overlap_local_id()const{
    return right_element_->overlap_local_id();
  }

  /// Returns the left element's overlap local id
  int_t get_left_overlap_local_id()const{
    return left_element_->overlap_local_id();
  }

  /// Returns a pointer to the left element
  Teuchos::RCP<Element> get_left_element(){
    return left_element_;
  }

  /// Returns a pointer to the right element
  Teuchos::RCP<Element> get_right_element(){
    return right_element_;
  }

  /// Returns a pointer to the vector of neighbor bond local ids (these are all the bonds attached to the left element)
  std::vector<int_t> * get_left_neighbor_bond_local_ids(){
    return &left_neighbor_bond_local_ids_;
  }

  /// Returns a pointer to the vector of neighbor bond local ids (these are all the bonds attached to the right element)
  std::vector<int_t> * get_right_neighbor_bond_local_ids(){
    return &right_neighbor_bond_local_ids_;
  }

  /// Sets the list of bond ids
  /// \param left_neighbor_bond_local_ids The vector to assign
  void set_left_neighbor_bond_local_ids(std::vector<int_t> left_neighbor_bond_local_ids){
    left_neighbor_bond_local_ids_ = left_neighbor_bond_local_ids;
  }

  /// Sets the list of bond ids
  /// \param right_neighbor_bond_local_ids The vector to assign
  void set_right_neighbor_bond_local_ids(std::vector<int_t> right_neighbor_bond_local_ids){
    right_neighbor_bond_local_ids_ = right_neighbor_bond_local_ids;
  }

private:
  /// Pointer to the left element on the bond
  Teuchos::RCP<DICe::mesh::Element> left_element_;
  /// Pointer to the right element on the bond
  Teuchos::RCP<DICe::mesh::Element> right_element_;
  /// Vector of all the left neighbor's bonds
  std::vector<int_t> left_neighbor_bond_local_ids_;
  /// Vector of all the right neighbor's bonds
  std::vector<int_t> right_neighbor_bond_local_ids_;
  /// True if left element is on processor
  const bool left_is_on_processor_;
  /// True if right element is on processor
  const bool right_is_on_processor_;
};

/// \class Edge
/// \brief Edge mesh object
class Edge : public Mesh_Object
{
public:
  /// Constructor
  /// \param local_id The local id to assign to this edge
  /// \param global_id The global id to assign to this edge
  /// \param left_node The left node global id
  /// \param right_node The right node global id
  /// \param left_element The left element global id
  /// \param right_element The right element global id
  Edge(const int_t local_id,
    const int_t global_id,
    const int_t left_node,
    const int_t right_node,
    const int_t left_element,
    const int_t right_element):
      Mesh_Object(global_id,local_id),
      left_node_(left_node),
      right_node_(right_node),
      left_element_(left_element),
      right_element_(right_element)
  {};

  /// Destructor
  ~Edge(){};

  /// Returns the left node global id
  int_t left_node_gid()const{
    return left_node_;
  }

  /// Returns the right node global id
  int_t right_node_gid()const{
    return right_node_;
  }

  /// Returns the left element global id
  int_t left_element_gid()const{
    return left_element_;
  }

  /// Returs the right element global id
  int_t right_element_gid()const{
    return right_element_;
  }

private:
  /// left node
  const int_t left_node_;
  /// right node
  const int_t right_node_;
  /// left element
  const int_t left_element_;
  /// right element
  const int_t right_element_;
};

/// \class Internal_Face_Edge
/// \brief Internal face or edge mesh object (edge for 2D, face for 3D). These do not include the faces or edges on the boundary
class Internal_Face_Edge : public Mesh_Object
{
public:
  /// Constructor
  /// \param local_id The local id to assign to this internal face or edge
  /// \param global_id The global id to assign to this internal face or edge
  /// \param target_node The node at the tip end of the normal vector pointing out from this internal face or edge
  /// \param neighbor_node The node at the feather end of the normal vector
  /// \param parent_subelement The subelement that this internal face edge belongs to
  Internal_Face_Edge(const int_t local_id,
    const int_t global_id,
    Teuchos::RCP<Node> target_node,
    Teuchos::RCP<Node> neighbor_node,
    Teuchos::RCP<Subelement> parent_subelement):
      Mesh_Object(global_id,local_id),
      target_node_(target_node),
      neighbor_node_(neighbor_node),
      parent_subelement_(parent_subelement)
  {};

  /// Destructor
  ~Internal_Face_Edge(){};

  /// Returns the target node
  Teuchos::RCP<Node> get_target_node()const{
    return target_node_;
  }

  /// Returns the neighbor node
  Teuchos::RCP<Node> get_neighbor_node()const{
    return neighbor_node_;
  }

  /// Reutns the parent subelement that this internal face or edge belongs to
  Teuchos::RCP<Subelement> parent_subelement()const{
    return parent_subelement_;
  }

private:
  /// target node
  Teuchos::RCP<Node> target_node_;
  /// neighbor node
  Teuchos::RCP<Node> neighbor_node_;
  /// parent subelement
  Teuchos::RCP<Subelement> parent_subelement_;
};

///// \class External_Face_Edge
///// \brief External face or edge mesh object (edge for 2D, face for 3D). These are faces or edges on the boundary
//class External_Face_Edge : public Mesh_Object
//{
//public:
//  /// Constructor
//  /// \param local_id The local id to assign to this internal face or edge
//  /// \param global_id The global id to assign to this internal face or edge
//  /// \param target_node The node at the tip end of the normal vector pointing out from this internal face or edge
//  /// \param parent_subelement The subelement that this internal face edge belongs to
//  External_Face_Edge(const int_t local_id,
//    const int_t global_id,
//    Teuchos::RCP<Node> target_node,
//    Teuchos::RCP<Subelement> parent_subelement):
//    target_node_(target_node),
//    parent_subelement_(parent_subelement),
//    Mesh_Object(global_id,local_id)
//  {};
//
//  /// Destructor
//  ~External_Face_Edge(){};
//
//  /// Returns the target node
//  Teuchos::RCP<Node> get_target_node()const{return target_node_;}
//
//  /// Reutns the parent subelement that this internal face or edge belongs to
//  Teuchos::RCP<Subelement> parent_subelement()const{return parent_subelement_;}
//
//private:
//  /// target node
//  Teuchos::RCP<Node> target_node_;
//  /// parent subelement
//  Teuchos::RCP<Subelement> parent_subelement_;
//};

/// \class Internal_Cell
/// \brief Internal cell mesh object
class Internal_Cell : public Mesh_Object
{
public:
  /// Constructor
  /// \param local_id The local id to assign to this cell
  /// \param global_id The global id to assign to this cell
  /// \param target_node The only parent element node that is inside this internal cell (there is one internal cell per node in the parent element)
  /// \param parent_subelement
  Internal_Cell(const int_t local_id,
    const int_t global_id,
    Teuchos::RCP<Node> target_node,
    Teuchos::RCP<Subelement> parent_subelement):
      Mesh_Object(global_id,local_id),
      target_node_(target_node),
      parent_subelement_(parent_subelement)
  {};

  /// Destructor
  ~Internal_Cell(){};

  /// Returns the target node
  Teuchos::RCP<Node> get_target_node()const{
    return target_node_;
  }

  /// Returns the parent element
  Teuchos::RCP<Subelement> parent_subelement()const{
    return parent_subelement_;
  }

private:
  /// The only node in the parent element that is inside this internal cell or makes up one of its vertices
  Teuchos::RCP<Node> target_node_;
  /// The parent element that this internal cell sits inside
  Teuchos::RCP<Subelement> parent_subelement_;
};

/// \class Node
/// \brief Node mesh object
class Node : public Mesh_Object
{
public:
  /// Constructor
  /// \param global_id The global id of this node
  /// \param local_id The local id of this node
  Node(const int_t global_id, const int_t local_id);

  /// Destructor
  ~Node(){};
};

/// typedef
typedef std::vector<Teuchos::RCP<DICe::mesh::Element> > element_set;
/// typedef
typedef std::vector<Teuchos::RCP<DICe::mesh::Subelement> > subelement_set;
/// typedef
typedef std::vector<Teuchos::RCP<DICe::mesh::Internal_Face_Edge> > internal_face_edge_set;
/// typedef
typedef std::vector<Teuchos::RCP<DICe::mesh::External_Face_Edge> > external_face_edge_set;
/// typedef
typedef std::vector<Teuchos::RCP<DICe::mesh::Edge> > edge_set;
/// typedef
typedef std::vector<Teuchos::RCP<DICe::mesh::Bond> > bond_set;
/// typedef
typedef std::vector<Teuchos::RCP<DICe::mesh::Internal_Cell> > internal_cell_set;
/// typedef
typedef std::map<int_t,Teuchos::RCP<DICe::mesh::Node> > node_set;

/// Holds information for side sets in the mesh
struct side_set_info
{
  /// ids included
  std::vector<int_t> ids;
  /// number of sides per set
  std::vector<int_t> num_side_per_set;
  /// number of distribution factors per set
  std::vector<int_t> num_df_per_set;
  /// element indices that touch the side set
  std::vector<int_t> elem_ind;
  /// distribution factor indices
  std::vector<int_t> df_ind;
  /// local element id list for the side set
  std::vector<int_t> ss_local_elem_list;
  /// global element id list for the side set
  std::vector<int_t> ss_global_elem_list;
  /// vector of the sides
  std::vector<int_t> ss_side_list;
  /// vector of the distribution factors
  std::vector<float> dist_factors;
  /// vector of normal x-components
  std::vector<scalar_t> normal_x;
  /// vector of normal y-components
  std::vector<scalar_t> normal_y;
  /// vector of normal z-components
  std::vector<scalar_t> normal_z;
  /// the size of the particular size (area of length)
  std::vector<scalar_t> side_size;

  /// Constructor
  side_set_info() : normals_are_populated(false){}

  /// true if the normal values have been populated
  bool normals_are_populated;

  /// Returns the set index for the given id
  /// \param id The id of the set
  const int_t get_set_index(const int_t & id){
    for(size_t i=0;i<ids.size();++i)
    {
      if(id==ids[i]){
        return i;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::logic_error,"  ERROR: side set was not found in mesh: " << id);
    return 0;
  }
};

/// \class Mesh
/// \brief The discretization used by the pysics classes.
///
/// There are a number of book-keeping members of a Mesh class. These are used in
/// different ways depending on the physics being solved. For example peridynamics
/// physics will use nodes and elements quite differently than transport physics.
/// The mesh also holds the solution fields used by the physics. These fields are
/// different than schema fields in that there is a lot more information used to
/// manage these fields. There is also more complicated communication of fields
/// accross processors. The underlying MultiFields in both cases are the same so
/// schema fields and mesh fields can interact with each other, but they have been
/// separated to preserve the separation between physics and DIC.
class Mesh
{
public:
  /// Constructor
  /// \param input_filename The name of the mesh file to read as in the input mesh
  /// \param output_filename The name of the mesh file to write the output to
  Mesh(const std::string & input_filename, const std::string & output_filename);

  /// Constructor (create a mesh from scratch, no existing exodus file)
  /// \param output_filename The name of the mesh file to write the output to
  Mesh(const std::string & output_filename):
    Mesh("",output_filename){}

  /// Destructor
  ~Mesh(){};

  /// Returns a pointer to the MPI or serial communicator
  Teuchos::RCP<MultiField_Comm> get_comm()const{
    return comm_;
  }

  /// Returns the spatial dimension of the mesh
  int_t spatial_dimension()const{
    return spatial_dimension_;
  }

  /// Sets the spatial dimension of the mesh
  /// \param dim The dimension
  void set_spatial_dimension(const int_t dim){
    spatial_dimension_ = dim;
  }

  /// Returns the input exoid for the mesh file (used by exodusII calls)
  int_t get_input_exoid()const{
    return input_exoid_;
  }

  /// Sets the input exoid for the mesh file (used by exodusII calls)
  void set_input_exoid(const int_t id){
    input_exoid_ = id;
  }

  /// Returns the output exoid for the mesh file (used by exodusII calls)
  int_t get_output_exoid()const{
    return output_exoid_;
  }

  /// Returns the output exoid for the face/edge output
  int_t get_face_edge_output_exoid()const{
    return face_edge_output_exoid_;
  }

  /// Sets the output exoid for the mesh file (used by exodusII calls)
  void set_output_exoid(const int_t id){
    output_exoid_ = id;
  }

  /// Sets the output exoid for the mesh file (used by exodusII calls)
  void set_face_edge_output_exoid(const int_t id){
    face_edge_output_exoid_ = id;
  }

  /// Returns the name of the input mesh (already decorated for multiple processors)
  std::string get_input_filename()const{
    return input_filename_;
  }

  /// Returns the name of the output mesh (already decorated for multiple processors)
  std::string get_output_filename()const{
    return output_filename_;
  }

  /// Returns the name of the output mesh (already decorated for multiple processors)
  std::string get_face_edge_output_filename()const{
    return face_edge_output_filename_;
  }

  /// Returns a pointer to the map of blocks and the associated element type
  block_type_map * get_block_type_map(){
    return &block_type_map_;
  }

  /// Returns a pointer to the side set info struct
  side_set_info * get_side_set_info(){
    return &side_set_info_;
  }

  /// Returns the number of node sets
  size_t num_node_sets()const{
    return node_bc_sets_.size();
  }

  /// Returns a pointer to the node bc sets
  bc_set * get_node_bc_sets(){return &node_bc_sets_;}

  /// Create the maps the dictate how fields are communicated across processors
  ///NOTE: exodus ids are 1 based, not 0 based
  /// \param force_elem_and_node_maps_to_match useful for particle methods where the element has one node
  /// and it is desired to have the elements split in the same way the nodes are
  void create_elem_node_field_maps(const bool force_elem_and_node_maps_to_match=false);

  /// Create the field maps for the mixed formulation elements
  void create_mixed_node_field_maps(Teuchos::RCP<Mesh> alt_mesh);

  /// Create the maps the dictate how fields are communicated across processors for faces
  ///NOTE: exodus ids are 1 based, not 0 based
  void create_face_cell_field_maps();

  /// The field registry holds fields and their descriptors together in a map
  field_registry * get_field_registry(){
    return &field_registry_;
  }

  /// Returns true if the field exists
  /// \param field_name The name of the field
  bool has_field(const field_enums::Field_Name field_name){
    field_registry::iterator field_it = field_registry_.begin();
    field_registry::iterator field_end = field_registry_.end();
    for(;field_it!=field_end;++field_it)
    {
      if(field_it->first.get_name()==field_name)
        return true;
    }
    return false;
  }

  /// all this does is return a pointer to an existing field
  /// \param field_spec The field_spec that defines the sought field
  Teuchos::RCP<MultiField> get_field(const field_enums::Field_Spec & field_spec){
    if(field_spec == field_enums::NO_SUCH_FS) return Teuchos::null;
    TEUCHOS_TEST_FOR_EXCEPTION(field_registry_.find(field_spec)==field_registry_.end(),
      std::invalid_argument,"Requested field is not in the registry." + field_spec.get_name_label());
    return field_registry_.find(field_spec)->second;
  }

  /// Return a pointer to the field based on the name alone
  /// \param field_name The string name of the field
  std::pair<field_enums::Field_Spec,Teuchos::RCP<MultiField> > get_field(const std::string & field_name);

  /// this will actually do the import, create a new field that has the overlap values and return a pointer to that field
  /// \param field_spec the field_spec defines the field to get
  Teuchos::RCP<MultiField> get_overlap_field(const field_enums::Field_Spec & field_spec);

  /// Returns a vector of stings of the field names
  /// \param entity_rank The rank that filters which fields are returned
  /// \param field_type Differentiates between vector and scalar fields
  /// \param only_the_printable If this is true the field names are only returned for printable fields
  std::vector<std::string> get_field_names(const field_enums::Entity_Rank entity_rank,
    const field_enums::Field_Type field_type,
    const bool only_the_printable)const;

  /// Creates a new field
  /// \param field_spec The field_spec that defines the field
  void create_field(const field_enums::Field_Spec & field_spec);

  /// Re-register a field that already exists with changes
  /// \param existing_field_spec The field_spec that defines the existing field
  /// \param new_field_spec The field_spec that re-defines the field's attributes
  void re_register_field(const field_enums::Field_Spec & existing_field_spec,
    const field_enums::Field_Spec & new_field_spec);

  /// Returns true of the given field exists
  /// \param field_name The string name of the field to check for
  bool field_exists(const std::string & field_name);

  /// Import a field from the distributed fields
  /// \param field_spec Defines the field to import
  /// \param to_map The map that dictates how the field will be imported
  Teuchos::RCP<MultiField> field_import(const field_enums::Field_Spec & field_spec,
    Teuchos::RCP<MultiField_Map> to_map);

  /// Export the field to the overlap distribution
  /// \param from_field The field to be exported
  /// \param to_field_spec Defines the recieving field
  /// \param mode How the field should be combined when communicated (inserted, added, etc.)
  void field_overlap_export(const Teuchos::RCP<MultiField> from_field,
    const field_enums::Field_Spec & to_field_spec,
    const Combine_Mode mode=INSERT);

  /// Print verbose information about all existing fields
  void print_field_info();

  /// Print the max min avg and std dev of all fields to the screen
  void print_field_stats();

  /// compute the average value, min, max and std dev of a multifield
  /// \param field_spec the field to evaluate
  /// \param min the minimum value
  /// \param max the maximum value
  /// \param avg the average
  /// \param std_dev the standard deviation,
  /// \param comp the component if this is a vector field
  void field_stats(const field_enums::Field_Spec field_spec,
    scalar_t & min,
    scalar_t & max,
    scalar_t & avg,
    scalar_t & std_dev,
    const int_t comp);

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_proc_map(){
    return proc_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_all_own_all_proc_map(){
    return all_own_all_proc_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_scalar_node_overlap_map(){
    return scalar_node_overlap_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_vector_node_overlap_map(){
    return vector_node_overlap_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_mixed_vector_node_overlap_map(){
    return mixed_vector_node_overlap_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_scalar_node_dist_map(){
    return scalar_node_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_vector_node_dist_map(){
    return vector_node_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_mixed_vector_node_dist_map(){
    return mixed_vector_node_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_scalar_face_dist_map(){
    return scalar_face_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_vector_face_dist_map(){
    return vector_face_dist_map_;
  }

  //  /// Returns a pointer to the communication map
  //  Teuchos::RCP<MultiField_Map> get_scalar_boundary_face_dist_map(){
  //  return scalar_boundary_face_dist_map_;
  //}
  //
  //  /// Returns a pointer to the communication map
  //  Teuchos::RCP<MultiField_Map> get_vector_boundary_face_dist_map(){
  //  return vector_boundary_face_dist_map_;
  //}

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_scalar_elem_dist_map(){
    return scalar_elem_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_vector_elem_dist_map(){
    return vector_elem_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_scalar_subelem_dist_map(){
    return scalar_subelem_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_vector_subelem_dist_map(){
    return vector_subelem_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_scalar_internal_cell_dist_map(){
    return scalar_cell_dist_map_;
  }

  /// Returns a pointer to the communication map
  Teuchos::RCP<MultiField_Map> get_vector_internal_cell_dist_map(){
    return vector_cell_dist_map_;
  }

  /// Returns a pointer to the set of all elements on this processor
  Teuchos::RCP<element_set> get_element_set(){
    return element_set_;
  }

  /// Returns a pointer to the subelements on this processor
  Teuchos::RCP<subelement_set> get_subelement_set(){
    return subelement_set_;
  }

  /// Returns a pointer to the internal cells on this processor
  Teuchos::RCP<internal_cell_set> get_internal_cell_set(){
    return internal_cell_set_;
  }

  /// Returns a pointer to the internal faces or edges on this processor
  Teuchos::RCP<internal_face_edge_set> get_internal_face_edge_set(){
    return internal_face_edge_set_;
  }

  /// Returns a pointer to the internal faces or edges on this processor
  //Teuchos::RCP<external_face_edge_set> get_boundary_face_edge_set(){
  //return boundary_face_edge_set_;
  //}

  /// Returns a pointer to the edges on this processor
  Teuchos::RCP<edge_set> get_edge_set(){
    return edge_set_;
  }

  /// Returns a pointer to the bonds on this processor
  Teuchos::RCP<bond_set> get_bond_set(){
    return bond_set_;
  }

  /// Returns a pointer to the nodes on this processor
  Teuchos::RCP<node_set> get_node_set(){
    return node_set_;
  }

  /// Returns elements on this processor sorted by block
  /// \param block_id The requested block
  Teuchos::RCP<element_set> get_element_set(const int_t block_id){
    TEUCHOS_TEST_FOR_EXCEPTION(block_type_map_.find(block_id)==block_type_map_.end(),
      std::invalid_argument,"Requested block does not exist" << block_id);
    return element_sets_by_block_[block_id];
  }

  /// Returns bonds on this processor sorted by block
  /// \param block_id The requested block
  Teuchos::RCP<bond_set> get_bond_set(const int_t block_id){
    TEUCHOS_TEST_FOR_EXCEPTION(block_type_map_.find(block_id)==block_type_map_.end(),
      std::invalid_argument,"Requested block does not exist" << block_id);
    return bond_sets_by_block_[block_id];
  }

  /// Returns subelements on this processor sorted by block
  /// \param block_id The requested block
  Teuchos::RCP<subelement_set> get_subelement_set(const int_t block_id){
    TEUCHOS_TEST_FOR_EXCEPTION(block_type_map_.find(block_id)==block_type_map_.end(),
      std::invalid_argument,"Requested block does not exist" << block_id);
    return subelement_sets_by_block_[block_id];
  }

  /// Returns nodes on this processor sorted by block
  /// \param block_id The requested block
  Teuchos::RCP<node_set> get_node_set(const int_t block_id){
    TEUCHOS_TEST_FOR_EXCEPTION(block_type_map_.find(block_id)==block_type_map_.end(),
      std::invalid_argument,"Requested block does not exist" << block_id);
    return node_sets_by_block_[block_id];
  }

  /// Returns internal faces on this processor sorted by block
  /// \param block_id The requested block
  Teuchos::RCP<internal_face_edge_set> get_internal_face_set(const int_t block_id){
    TEUCHOS_TEST_FOR_EXCEPTION(block_type_map_.find(block_id)==block_type_map_.end(),
      std::invalid_argument,"Requested block does not exist" << block_id);
    return internal_face_edge_sets_by_block_[block_id];
  }

  /// Returns the whole map of elements by block
  std::map<int_t,Teuchos::RCP<element_set> > * get_element_sets_by_block(){
    return &element_sets_by_block_;
  }

  /// Returns the whole map of bonds by block
  std::map<int_t,Teuchos::RCP<bond_set> > * get_bond_sets_by_block(){
    return &bond_sets_by_block_;
  }

  /// Returns the whole map of subelements by block
  std::map<int_t,Teuchos::RCP<subelement_set> > * get_subelement_sets_by_block(){
    return &subelement_sets_by_block_;
  }

  /// Returns the whole map of nodes by block
  std::map<int_t,Teuchos::RCP<node_set> > * get_node_sets_by_block(){
    return &node_sets_by_block_;
  }

  /// Returns the whole map of internal faces or edges by block
  std::map<int_t,Teuchos::RCP<internal_face_edge_set> > * get_internal_face_sets_by_block(){
    return &internal_face_edge_sets_by_block_;
  }

  /// Returns the number of elements in the mesh
  size_t num_elem()const{
    return element_set_->size();
  }

  /// Returns the number of subelements in the mesh
  size_t num_subelem()const{
    return subelement_set_->size();
  }

  /// Returns the number of internal faces or edges in the mehs
  size_t num_internal_faces()const{
    return internal_face_edge_set_->size();
  }

//  /// Returns the number of internal faces or edges in the mehs
//  const int_t num_boundary_faces()const{return boundary_face_edge_set_->size();}

  /// Returns the number of edges in the mesh
  size_t num_edges()const{
    return edge_set_->size();
  }

  /// Returns the number of bonds in the mesh
  size_t num_bonds()const{
    return (int_t)bond_set_->size();
  }

  /// Returns the number of nodes in the mesh
  size_t num_nodes()const{
    return (int_t)node_set_->size();
  }

  /// Returns true if the mesh has been initialized
  bool is_initialized()const{
    return is_initialized_;
  }

  /// Sets the initialized flag for the mesh
  void set_initialized(){
    is_initialized_=true;
  }

  /// Returns true if the control volumes have been initialized
  bool control_volumes_are_initialized()const{
    return control_volumes_are_initialized_;
  }

  /// Sets the control volumes are initialized flag (requied for CVFEM)
  void set_control_volumes_are_initialzed(){
    control_volumes_are_initialized_=true;
  }

  /// Returns true if the control volumes have been initialized
  bool cell_sizes_are_initialized()const{
    return cell_sizes_are_initialized_;
  }

  /// Sets the control volumes are initialized flag (requied for CVFEM)
  void set_cell_sizes_are_initialized(){
    cell_sizes_are_initialized_=true;
  }

  /// Returns the maximum number of element relations
  int_t max_num_elem_relations(){
    return max_num_elem_relations_;
  }

  /// Returns the mean number of element relations
  int_t mean_num_elem_relations(){
    return mean_num_elem_relations_;
  }

  /// Returns the minimum number of element relations
  int_t min_num_elem_relations(){
    return min_num_elem_relations_;
  }

  /// Sets the maximum number of element relations
  void set_max_num_elem_relations(const int_t & max_rel){
    max_num_elem_relations_ = max_rel;
  }

  /// Sets the minimum number of element relations
  void set_min_num_elem_relations(const int_t & min_rel){
    max_num_elem_relations_ = min_rel;
  }

  /// Sets the meam number of element relations
  void set_mean_num_elem_relations(const int_t & mean_rel){
    max_num_elem_relations_ = mean_rel;
  }

  /// Returns the maximum number of node relations
  int_t max_num_node_relations(){
    return max_num_node_relations_;
  }

  /// Returns the mean number of node relations
  int_t mean_num_node_relations(){
    return mean_num_node_relations_;
  }

  /// Returns the minimum number of node relations
  int_t min_num_node_relations(){
    return min_num_node_relations_;
  }

  /// Sets the maximum number of node relations
  void set_max_num_node_relations(const int_t & max_rel){
    max_num_node_relations_ = max_rel;
  }

  /// Sets the minimum number of node relations
  void set_min_num_node_relations(const int_t & min_rel){
    max_num_node_relations_ = min_rel;
  }

  /// Sets the meam number of node relations
  void set_mean_num_node_relations(const int_t & mean_rel){
    max_num_node_relations_ = mean_rel;
  }

  /// Sets the maxium number of entries per row in the stiffness matrix
  void set_max_num_entries_per_row(const int_t & num_entries){
    max_num_entries_per_row_ = num_entries;
  }

  /// Returns the maximum number of subelem face relations
  int_t max_num_subelem_face_relations(){
    return max_num_subelem_face_relations_;
  }

  /// Returns the mean number of subelem face relations
  int_t mean_num_subelem_face_relations(){
    return mean_num_subelem_face_relations_;
  }

  /// Returns the minimum number of subelem face relations
  int_t min_num_subelem_face_relations(){
    return min_num_subelem_face_relations_;
  }

  /// Returns the max number of entries per row in the stiffness matrix
  int_t max_num_entries_per_row(){
    return max_num_entries_per_row_;
  }

  /// Sets the maximum number of subelement face relations
  void set_max_num_subelem_face_relations(const int_t & max_rel){
    max_num_subelem_face_relations_ = max_rel;
  }

  /// Sets the minimum number of subelement face relations
  void set_min_num_subelem_face_relations(const int_t & min_rel){
    max_num_subelem_face_relations_ = min_rel;
  }

  /// Sets the mean number of subelement face relations
  void set_mean_num_subelem_face_relations(const int_t & mean_rel){
    max_num_subelem_face_relations_ = mean_rel;
  }

  /// Returns the number of blocks in the mesh
  size_t num_blocks()const{
    return block_type_map_.size();
  }

  /// Returns the number of elemnets in the given block
  /// \param block_id The id of the block to count the number of elements
  int_t num_elem_in_block(const int_t block_id)const;

  /// Convert the global id to local id
  /// \param global_id The globa id to search for
  int_t elem_global_to_local_id(const int_t global_id)
  {
    element_set::const_iterator it = element_set_->begin();
    element_set::const_iterator end = element_set_->end();
    for(;it!=end;++it)
    {
      if(it->get()->global_id()==global_id)
        return it->get()->local_id();
    }
    return -1;
  }

  /// Convert a node global id to local
  /// \param global_id The global id to search for
  int_t node_global_to_local_id(const int_t global_id)
  {
    if(node_set_->find(global_id)!=node_set_->end())
      return node_set_->find(global_id)->second->local_id();
    else
      return -1;
  }

  /// Note y and x values are reversed here
  std::vector<std::set<std::pair<int_t,int_t> > > * get_pixel_ownership(){
    return & pixel_ownership_;
  }
  /// Note y and x values are reversed here
  std::vector<std::set<std::pair<scalar_t,scalar_t> > > * get_integration_point_ownership(){
    return & integration_point_ownership_;
  }

  /// sets boundary condition defs
  /// \param bc_defs a vector of bc definitions
  void set_bc_defs(std::vector<Boundary_Condition_Def> & bc_defs){
    bc_defs_ = bc_defs;
  }

  /// returns a pointer to the boundary condition defs
  std::vector<Boundary_Condition_Def> * bc_defs(){
    return & bc_defs_;
  }

  /// return the initial condition value in x
  scalar_t ic_value_x()const{
    return ic_value_x_;
  }

  /// return the initial condition value in x
  scalar_t ic_value_y()const{
    return ic_value_y_;
  }

  /// set the value of the initial conditions
  void set_ic_values(const scalar_t & value_x,
    const scalar_t & value_y){
    ic_value_x_ = value_x;
    ic_value_y_ = value_y;
  }

  /// set the value of the is a regular grid flag
  void set_is_regular_grid(const bool is_regular_grid){
    is_regular_grid_ = is_regular_grid;
  }

  /// returns true if this mesh is a regular grid
  bool is_regular_grid()const{
    return is_regular_grid_;
  }

private:
  /// The parallel or serial communicator
  Teuchos::RCP<MultiField_Comm> comm_;
  /// Number of spatial dimensions in the mesh
  int_t spatial_dimension_;
  /// The set of all elements local to this processor
  Teuchos::RCP<element_set> element_set_;
  /// The set of all subelements local to this processor
  Teuchos::RCP<subelement_set> subelement_set_;
  /// The set of all nodes local to this processor
  Teuchos::RCP<node_set> node_set_;
  /// The set of internal faces or edges local to this processor
  Teuchos::RCP<internal_cell_set> internal_cell_set_;
  /// The set of internal faces or edges local to this processor
  Teuchos::RCP<internal_face_edge_set> internal_face_edge_set_;
  /// The set of boundary faces or edges local to this processor
  //Teuchos::RCP<external_face_edge_set> boundary_face_edge_set_;
  /// The set of edeges local to this processor
  Teuchos::RCP<edge_set> edge_set_;
  /// The set of bonds local to this processor
  Teuchos::RCP<bond_set> bond_set_;
  /// Map of elements sorted by block
  std::map<int_t,Teuchos::RCP<element_set> > element_sets_by_block_;
  /// Map of bonds sorted by block
  std::map<int_t,Teuchos::RCP<bond_set> > bond_sets_by_block_;
  /// Map of subelements sorted by block
  std::map<int_t,Teuchos::RCP<subelement_set> > subelement_sets_by_block_;
  /// Map of nodes sorted by block
  std::map<int_t,Teuchos::RCP<node_set> > node_sets_by_block_;
  /// Map of internal faces or edges sorted by block
  std::map<int_t,Teuchos::RCP<internal_face_edge_set> > internal_face_edge_sets_by_block_;
  /// Map of element types associated with block ids
  block_type_map block_type_map_;
  /// Side set information from the mesh file
  side_set_info side_set_info_;
  /// True when the mesh has been initialized properly
  bool is_initialized_;
  /// max number of element relations
  int_t max_num_elem_relations_;
  /// max number of node relations
  int_t max_num_node_relations_;
  /// max number of subelement relations
  int_t max_num_subelem_face_relations_;
  /// mean number of element relations
  int_t mean_num_elem_relations_;
  /// mean number of node relations
  int_t mean_num_node_relations_;
  /// mean number of subelement relations
  int_t mean_num_subelem_face_relations_;
  /// min number of element relations
  int_t min_num_elem_relations_;
  /// min number of node relations
  int_t min_num_node_relations_;
  /// min number of subelement relations
  int_t min_num_subelem_face_relations_;
  /// max number of entries per row in the stiffness matrix
  int_t max_num_entries_per_row_;
  /// input file exodus id
  int_t input_exoid_;
  /// output file exodus id
  int_t output_exoid_;
  /// face edge output file exodus id
  int_t face_edge_output_exoid_;
  /// input mesh file name (already decorated for multiple processors)
  std::string input_filename_;
  /// output mesh file name (already decorated for multiple processors)
  std::string output_filename_;
  /// output face edge file name
  std::string face_edge_output_filename_;
  /// Set of all the node bcs
  bc_set node_bc_sets_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> all_own_all_proc_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> proc_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> scalar_node_overlap_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> scalar_node_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> vector_node_overlap_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> vector_node_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> mixed_vector_node_overlap_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> mixed_vector_node_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> scalar_elem_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> vector_elem_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> scalar_face_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> vector_face_dist_map_;
  /// Comminication map
  //Teuchos::RCP<MultiField_Map> scalar_boundary_face_dist_map_;
  /// Comminication map
  //Teuchos::RCP<MultiField_Map> vector_boundary_face_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> scalar_cell_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> vector_cell_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> scalar_subelem_dist_map_;
  /// Comminication map
  Teuchos::RCP<MultiField_Map> vector_subelem_dist_map_;
  /// Registry that holds all the mesh fields
  field_registry field_registry_;
  /// True if the control volume fields have been initialized
  bool control_volumes_are_initialized_;
  /// True if the cell size field has been populated
  bool cell_sizes_are_initialized_;
  /// Vector of vectors for each element that provide the pixels each element owns.
  /// Note the ownership is unique and complete, multiple elements cannot own the same pixel
  /// and all pixels must be owned
  std::vector<std::set<std::pair<int_t,int_t> > > pixel_ownership_;
  /// Vector of vectors for each element that provide the gauss integration points each element owns.
  /// Note the ownership is unique and complete, multiple elements cannot own the same integration points
  /// and all integration points must be owned
  std::vector<std::set<std::pair<scalar_t,scalar_t> > > integration_point_ownership_;
  /// storage of the boundary condition definitions
  std::vector<Boundary_Condition_Def> bc_defs_;
  /// value to use for initial condition in x
  scalar_t ic_value_x_;
  /// value to use for initial condition in x
  scalar_t ic_value_y_;
  /// true if this mesh is a regular grid
  bool is_regular_grid_;
};

/// \class Shape_Function_Evaluator
/// \brief Base class for shape function evaluation
class Shape_Function_Evaluator
{
public:
  /// Constrtuctor
  /// \param num_functions The number of shape functions
  /// \param dimension The spatial dimension
  Shape_Function_Evaluator(const int_t num_functions,
    const int_t dimension):
  num_functions_(num_functions),
  dimension_(dimension)
  {};

  /// Destructor
  virtual ~Shape_Function_Evaluator(){};

  /// Returns the number of functions
  int_t num_functions()const{
    return num_functions_;
  }

  /// Returns the spatial dimension
  int_t dimension()const{
    return dimension_;
  }

  /// Evaluation of the shape functions
  /// \param nodal_coords The coordinates of the nodes
  /// \param point_coords The coordinates of the point to evaluate the shape functinos
  /// \param coefficient The weighting function value in the Gauss integration
  /// \param shape_function_values The values returned for this particular point
  virtual void evaluate_shape_functions(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_values)=0;

  /// Evaluation of the shape functions
  /// \param natural_coords The coordinates of the point to evaluate the shape functinos
  /// \param shape_function_values The values returned for this particular point
  virtual void evaluate_shape_functions(const scalar_t * natural_coords,
    scalar_t * shape_function_values)=0;

  /// Evaluation of the shape function derivatives
  /// \param nodal_coords The coordinates of the nodes
  /// \param coefficient The weighting function value in the Gauss integration
  /// \param shape_function_derivative_values The values returned for this particular point
  virtual void evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_derivative_values)=0;

  /// Evaluation of the shape function derivatives
  /// \param natural_coords the isoparamteric coords of the point
  /// \param shape_function_derivative_values the array that is returned with the shape function derivatives
  virtual void evaluate_shape_function_derivatives(const scalar_t * natural_coords,
    scalar_t * shape_function_derivative_values)=0;

  /// Determine if a given point is inside the element or external
  /// \param point_coords The test point location
  /// \param nodal_coords The corrdinates of the nodes for this element
  /// \param coefficient The weighting function value
  virtual bool is_in_element(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient)=0;

  /// Returns the integration points and weights in the natural coordinates (isoparametric, not physical)
  /// \param order The gauss integration order
  /// \param locations The natural coordinates listed by point in row and dim in column
  /// \param weights The integration weights for each point
  /// \param num_points the number of integration points
  virtual void get_natural_integration_points(const int_t order,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
    Teuchos::ArrayRCP<scalar_t> & weights,
    int_t & num_points)=0;

protected:
  /// Protect the default constructor
  Shape_Function_Evaluator(const Shape_Function_Evaluator&);
  /// Comparison operator
  Shape_Function_Evaluator& operator=(const Shape_Function_Evaluator&);
  /// Number of functions
  const int_t num_functions_;
  /// Number of spatial dimensions
  const int_t dimension_;
};

/// \class CVFEM_Linear_Tri3
/// \brief CVFEM Linear Tri element shape function evaluator
class CVFEM_Linear_Tri3 : public Shape_Function_Evaluator
{
public:
  /// Constructor
  CVFEM_Linear_Tri3():Shape_Function_Evaluator(3,2){};

  /// Destructor
  virtual ~CVFEM_Linear_Tri3(){};

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_values);

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * natural_coords,
    scalar_t * shape_function_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_derivative_values);

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * natural_coords,
    scalar_t * shape_function_derivative_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// See base class documentation
  virtual bool is_in_element(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient);

  /// See base class documentation
  virtual void get_natural_integration_points(const int_t order,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
    Teuchos::ArrayRCP<scalar_t> & weights, int_t & num_points);
};

/// \class CVFEM_Linear_Tet4
/// \brief CVFEM Linear Tet element shape function evaluator
class CVFEM_Linear_Tet4 : public Shape_Function_Evaluator
{
public:
  /// Constructor
  CVFEM_Linear_Tet4():Shape_Function_Evaluator(4,3){};

  /// Destructor
  virtual ~CVFEM_Linear_Tet4(){};

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_values);

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * natural_coords,
    scalar_t * shape_function_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_derivative_values);

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * natural_coords,
    scalar_t * shape_function_derivative_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// See base class documentation
  virtual bool is_in_element(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient);

  /// See base class documentation
  virtual void get_natural_integration_points(const int_t order,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
    Teuchos::ArrayRCP<scalar_t> & weights,
    int_t & num_points){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }
};

/// \class FEM_Linear_Hex8
/// \brief FEM Linear Tet element shape function evaluator
class FEM_Linear_Hex8 : public Shape_Function_Evaluator
{
public:
  /// Constructor
  FEM_Linear_Hex8():Shape_Function_Evaluator(8,3){};

  /// Destructor
  virtual ~FEM_Linear_Hex8(){};

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * natural_coords,
    scalar_t * shape_function_values);

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_derivative_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * natural_coords,
    scalar_t * shape_function_derivative_values);

  /// See base class documentation
  virtual bool is_in_element(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient);

  /// See base class documentation
  virtual void get_natural_integration_points(const int_t order,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
    Teuchos::ArrayRCP<scalar_t> & weights,
    int_t & num_points);
};

/// \class FEM_Linear_Quad4
/// \brief FEM Linear quad element shape function evaluator
class FEM_Linear_Quad4 : public Shape_Function_Evaluator
{
public:
  /// Constructor
  FEM_Linear_Quad4():Shape_Function_Evaluator(4,2){};

  /// Destructor
  virtual ~FEM_Linear_Quad4(){};

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * nodal_coords,
    const scalar_t * point_coords, const scalar_t & coefficient,
    scalar_t * shape_function_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * natural_coords,
    scalar_t * shape_function_values);

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_derivative_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * natural_coords,
    scalar_t * shape_function_derivative_values);

  /// See base class documentation
  virtual bool is_in_element(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient);

  /// See base class documentation
  virtual void get_natural_integration_points(const int_t order,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
    Teuchos::ArrayRCP<scalar_t> & weights,
    int_t & num_points);
};

/// \class FEM_Linear_Tri3
/// \brief FEM Linear triangle element shape function evaluator
class FEM_Linear_Tri3 : public Shape_Function_Evaluator
{
public:
  /// Constructor
  FEM_Linear_Tri3():Shape_Function_Evaluator(3,2){};

  /// Destructor
  virtual ~FEM_Linear_Tri3(){};

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * nodal_coords,
    const scalar_t * point_coords, const scalar_t & coefficient,
    scalar_t * shape_function_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * natural_coords,
    scalar_t * shape_function_values);

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_derivative_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * natural_coords,
    scalar_t * shape_function_derivative_values);

  /// See base class documentation
  virtual bool is_in_element(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient);

  /// See base class documentation
  virtual void get_natural_integration_points(const int_t order,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
    Teuchos::ArrayRCP<scalar_t> & weights,
    int_t & num_points);
};


/// \class FEM_Quadratic_Tri6
/// \brief FEM quadratic triangle element shape function evaluator
class FEM_Quadratic_Tri6 : public Shape_Function_Evaluator
{
public:
  /// Constructor
  FEM_Quadratic_Tri6():Shape_Function_Evaluator(6,2){};

  /// Destructor
  virtual ~FEM_Quadratic_Tri6(){};

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * nodal_coords,
    const scalar_t * point_coords, const scalar_t & coefficient,
    scalar_t * shape_function_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * natural_coords,
    scalar_t * shape_function_values);

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_derivative_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * natural_coords,
    scalar_t * shape_function_derivative_values);

  /// See base class documentation
  virtual bool is_in_element(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient);

  /// See base class documentation
  virtual void get_natural_integration_points(const int_t order,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
    Teuchos::ArrayRCP<scalar_t> & weights,
    int_t & num_points);
};

/// \class FEM_Barycentric_Tri6
/// \brief FEM Barycentric triangle element shape function evaluator
class FEM_Barycentric_Tri6 : public Shape_Function_Evaluator
{
public:
  /// Constructor
  FEM_Barycentric_Tri6():Shape_Function_Evaluator(6,2){};

  /// Destructor
  virtual ~FEM_Barycentric_Tri6(){};

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * nodal_coords,
    const scalar_t * point_coords, const scalar_t & coefficient,
    scalar_t * shape_function_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_functions(const scalar_t * natural_coords,
    scalar_t * shape_function_values);

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * nodal_coords,
    const scalar_t & coefficient,
    scalar_t * shape_function_derivative_values){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }

  /// see base class documentation
  virtual void evaluate_shape_function_derivatives(const scalar_t * natural_coords,
    scalar_t * shape_function_derivative_values);

  /// See base class documentation
  virtual bool is_in_element(const scalar_t * nodal_coords,
    const scalar_t * point_coords,
    const scalar_t & coefficient);

  /// See base class documentation
  virtual void get_natural_integration_points(const int_t order,
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
    Teuchos::ArrayRCP<scalar_t> & weights,
    int_t & num_points);
};


/// \class Shape_Function_Evaluator_Factory
/// \brief Factory class that creates shape function evaluators
class Shape_Function_Evaluator_Factory
{
public:
  /// Constructor
  Shape_Function_Evaluator_Factory(){};

  /// Destructor
  virtual ~Shape_Function_Evaluator_Factory(){}

  /// Create an evaluator
  /// \param elem_type The base element type that determines which evaluator to make
  virtual Teuchos::RCP<Shape_Function_Evaluator> create(const Base_Element_Type elem_type);

private:
  /// Protect the base default constructor
  Shape_Function_Evaluator_Factory(const Shape_Function_Evaluator_Factory&);

  /// Comparison operator
  Shape_Function_Evaluator_Factory& operator=(const Shape_Function_Evaluator_Factory&);
};

/// create a point or triangle mesh from scratch, (not read from an existing file)
/// \param elem_type the element type TRI6 or TRI3
/// \param node_coords_x x coordinates of the nodes
/// \param node_coords_y y coordinates of the nodes
/// \param connectivity the connectivity matrix (always 1...n based)
/// \param node_map converts local ids to global ids
/// \param elem_map converts local ids to global ids
/// \param dirichlet_boundary_nodes a set of nodes to mark as dirichlet boundary nodes
/// \param neumann_boundary_nodes a set of nodes to mark as a neumann boundary
/// \param lagrange_boundary_nodes a set of nodes to mark as the lagrange multiplier boundary nodes
/// \param serial_output_filename The output fiel name with no parallel decorations
Teuchos::RCP<Mesh> create_point_or_tri_mesh(const DICe::mesh::Base_Element_Type elem_type,
  Teuchos::ArrayRCP<scalar_t> node_coords_x,
  Teuchos::ArrayRCP<scalar_t> node_coords_y,
  Teuchos::ArrayRCP<int_t> connectivity,
  Teuchos::ArrayRCP<int_t> node_map,
  Teuchos::ArrayRCP<int_t> elem_map,
  std::vector<std::pair<int_t,int_t> > & dirichlet_boundary_nodes,
  std::set<int_t> & neumann_boundary_nodes,
  std::set<int_t> & lagrange_boundary_nodes,
  const std::string & serial_output_filename);

/// create a linear tri mesh from a quadratic one
/// \param tri6_mesh the parent tri6 mesh
/// \param serial_output_filename the name to use for the output
Teuchos::RCP<Mesh> create_tri3_mesh_from_tri6(Teuchos::RCP<Mesh> tri6_mesh,
  const std::string & serial_output_filename);

} //mesh

/// gather the natural integration points for a triangle
/// \param order the integration order
/// \param locations array returned with integration point locations
/// \param weights array returned with the weights associated with each point
/// \param num_points return value with the total number of integration points
void tri3d_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points);

/// gather the natural integration points for a triangle
/// \param order the integration order
/// \param locations array returned with integration point locations
/// \param weights array returned with the weights associated with each point
/// \param num_points return value with the total number of integration points
void tri2d_natural_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points);

/// gather the natural integration points for a triangle
/// these points are equally spaced and weighted so this isn't exact for any order
/// \param order the integration order
/// \param locations array returned with integration point locations
/// \param weights array returned with the weights associated with each point
/// \param num_points return value with the total number of integration points
void tri2d_nonexact_integration_points(const int_t order,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & locations,
  Teuchos::ArrayRCP<scalar_t> & weights,
  int_t & num_points);

/// compute the cross product of two vectors (ABxAC) and return the area of the parallelogram
/// \param A vector to point A
/// \param B vector to point B
/// \param C vector to point C
scalar_t cross(const scalar_t * A,
  const scalar_t * B,
  const scalar_t * C);

/// compute the cross product of two vectors (ABxAC) in 3d
/// \param A vector to point A
/// \param B vector to point B
/// \param C vector to point C
scalar_t cross3d(const scalar_t * A,
  const scalar_t * B,
  const scalar_t * C);

/// compute the cross product of two vectors (ABxAC) and return the area of the parallelogram also output the normal of the two
/// \param A vector to point A
/// \param B vector to point B
/// \param C vector to point C
/// \param normal [out] the vector normal to ABXAC
scalar_t cross3d_with_normal(const scalar_t * A,
  const scalar_t * B,
  const scalar_t * C,
  scalar_t * normal);


/// compute the cross product of two vectors (ABxAC) and return the area of the parallelogram also output the cross-product of the two
/// \param A vector to point A
/// \param B vector to point B
/// \param C vector to point C
/// \param cross_prod [out] the vector ABXAC (not normalized by the mag as the normal would be)
void cross3d_with_cross_prod(const scalar_t * A,
  const scalar_t * B,
  const scalar_t * C,
  scalar_t * cross_prod);

/// Compute the determinant of a 4 x 4
/// \param a The coefficients in an array format
scalar_t determinant_4x4(const scalar_t * a);

/// Compute the 1d gauss points
/// \param r the gauss points
/// \param w the wieghts
/// \param gauss_order the order of integration
void gauss_1D(Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & r,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & w,
  int_t gauss_order);

/// Compute the 2d gauss points
/// \param r the gauss points
/// \param w the wieghts
/// \param gauss_order the order of integration
void gauss_2D(Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & r,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & w,
  int_t gauss_order);

/// Compute the kronecker product of two matrices
/// \param A the A matrix
/// \param B the B matrix
Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > kronecker( Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & A,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > & B);

/// calculate the element jacobian
/// \param xcap the nodal coordinates
/// \param DN the derivative of the shape functions
/// \param jacobian the jacobian matrix
/// \param inv_jacobian inverse of the jacobian matrix
/// \param J the determinant of the jacobian
/// \param num_elem_nodes number of nodes per element
/// \param dim spatial dimension
void calc_jacobian(const scalar_t * xcap,
  const scalar_t * DN,
  scalar_t * jacobian,
  scalar_t * inv_jacobian,
  scalar_t & J,
  int_t num_elem_nodes,
  int_t dim );

/// compute the B matrix for linear elasticity
/// \param DN the derivative of the shape functions
/// \param inv_jacobian the inverse of the jacobian matrix
/// \param num_elem_nodes The number of nodes per element
/// \param dim spatial dimension
/// \param solid_B [out] the output B vector
void calc_B(const scalar_t * DN,
  const scalar_t * inv_jacobian,
  int_t num_elem_nodes,
  int_t dim,
  scalar_t * solid_B);

} // namespace DICe

#endif /* DICE_MESH_H_ */
