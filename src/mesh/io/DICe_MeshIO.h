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

#ifndef DICE_MESHIO_H_
#define DICE_MESHIO_H_

#include <DICe.h>
#include <DICe_Mesh.h>
#include <DICe_MeshEnums.h>

#include <Teuchos_RCP.hpp>

namespace DICe{
namespace mesh{

/// Read the exodus mesh given the input and output names
/// \param serial_input_filename The input file with no parallel decorations (for example mesh.g instead of mesh.g.4.0)
/// \param serial_output_filename The output fiel name with no parallel decorations
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Mesh> read_exodus_mesh(const std::string & serial_input_filename,
  const std::string & serial_output_filename);

/// Read the coordinates of the nodes from the mesh
/// \param mesh The mesh to use for this function
DICE_LIB_DLL_EXPORT
void read_exodus_coordinates(Teuchos::RCP<Mesh> mesh);

/// Read the coordinates of the nodes from the mesh
/// \param coords_x vector that will be populated with the x values
/// \param coords_y vector that will be populated with the y values
/// \param coords_z vector that will be populated with the z values
DICE_LIB_DLL_EXPORT
void read_exodus_coordinates(const std::string & exo_file,
  std::vector<scalar_t> & coords_x,
  std::vector<scalar_t> & coords_y,
  std::vector<scalar_t> & coords_z);

/// Read the nodal field names of from the mesh
/// \param mesh The mesh to use for this function
/// returns a vector of the field names
DICE_LIB_DLL_EXPORT
std::vector<std::string>
read_exodus_field_names(Teuchos::RCP<Mesh> mesh);

/// Read the nodal field names of from the mesh
/// \param file_name The file to read for this function
/// returns a vector of the field names
DICE_LIB_DLL_EXPORT
std::vector<std::string>
read_exodus_field_names(const std::string & file_name);

/// returns the number of steps in the file
/// \param mesh The mesh to use for this function
DICE_LIB_DLL_EXPORT
int_t read_exodus_num_steps(Teuchos::RCP<Mesh> mesh);

/// returns the number of steps in the file
/// \param file_name the file to read for this function
DICE_LIB_DLL_EXPORT
int_t read_exodus_num_steps(const std::string & file_name);

/// Returns a vector with nodal field values
/// \param mesh the mesh to use
/// \param var_index the variable index for the requested field
/// \param step the step number
DICE_LIB_DLL_EXPORT
std::vector<scalar_t>
read_exodus_field(
Teuchos::RCP<Mesh> mesh,
  const int_t var_index,
  const int_t step);

/// Returns a vector with nodal field values
/// \param file_name the exodus file to read
/// \param var_index the variable index for the requested field
/// \param step the step number
DICE_LIB_DLL_EXPORT
std::vector<scalar_t>
read_exodus_field(
  const std::string & file_name,
  const int_t var_index,
  const int_t step);

/// Returns a vector with nodal field values
/// \param file_name the name of the exodus file
/// \param field_name string name of the requested field
/// \param step the step number
DICE_LIB_DLL_EXPORT
std::vector<scalar_t>
read_exodus_field(
  const std::string & file_name,
  const std::string & field_name,
  const int_t step);

/// Create an exodus output file
/// \param mesh The mesh to use for this function
/// \param output_folder The name of the output folder
DICE_LIB_DLL_EXPORT
void create_output_exodus_file(Teuchos::RCP<Mesh> mesh,
  const std::string & output_folder);

/// Create an output file with the face/edge information
/// \param mesh The mesh to use for this function
/// \param output_folder The name of the output folder
DICE_LIB_DLL_EXPORT
void create_face_edge_output_exodus_file(Teuchos::RCP<Mesh> mesh,
  const std::string & output_folder);


/// Get the variable index for writing this varaible to an exodus file
/// \param mesh The mesh to use for this function
/// \param name The name of the field
/// \param component The field component
/// \param rank The type of field
/// \param ignore_dimension Ignore the dimension argument
/// \param only_printable Return the index of only printable fields
DICE_LIB_DLL_EXPORT
int_t get_var_index(Teuchos::RCP<Mesh> mesh,
  const std::string & name,
  const std::string & component,
  const field_enums::Entity_Rank rank,
  const bool ignore_dimension=false,
  const bool only_printable=true);

/// Write a time step to the exodus file
/// \param mesh The mesh to use for this function
/// \param time_step_num The time step number NOTE: has to be 1 or greater or exodus will throw an error
/// \param time_value The time for this step
DICE_LIB_DLL_EXPORT
void exodus_output_dump(Teuchos::RCP<Mesh> mesh,
  const int_t & time_step_num,
  const float & time_value);

/// Write a time step to the exodus file
/// \param mesh The mesh to use for this function
/// \param time_step_num The time step number NOTE: has to be 1 or greater or exodus will throw an error
/// \param time_value The time for this step
DICE_LIB_DLL_EXPORT
void exodus_face_edge_output_dump(Teuchos::RCP<Mesh> mesh,
  const int_t & time_step_num,
  const float & time_value);

/// Close the exodus file
/// \param mesh The mesh to use for this function
DICE_LIB_DLL_EXPORT
void close_exodus_output(Teuchos::RCP<Mesh> mesh);

/// Close the face/edge exodus file
/// \param mesh The mesh to use for this function
DICE_LIB_DLL_EXPORT
void close_face_edge_exodus_output(Teuchos::RCP<Mesh> mesh);

/// Create the variable names in the exodus file
/// \param mesh The mesh to use for this function
DICE_LIB_DLL_EXPORT
void create_exodus_output_variable_names(Teuchos::RCP<Mesh> mesh);

/// Create the variable names in the exodus file
/// \param mesh The mesh to use for this function
DICE_LIB_DLL_EXPORT
void create_face_edge_output_variable_names(Teuchos::RCP<Mesh> mesh);

/// Create the necessary fields for CVFEM
DICE_LIB_DLL_EXPORT
void initialize_control_volumes(Teuchos::RCP<Mesh> mesh);

/// Compute the edge lengths and normals for a subelement derived from a tri 3 parent element
/// \param mesh The computational mesh
/// \param coords_values An array containing the coordinate values
/// \param cv_node the control volume node
/// \param subelement The control volume subelement
DICE_LIB_DLL_EXPORT
void tri3_sub_elem_edge_legths_and_normals(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> cv_node,
  Teuchos::RCP<DICe::mesh::Subelement> subelement);

/// Compute the edge lengths and normals for a subelement derived from a tetra 4 parent element
/// \param mesh The computational mesh
/// \param coords_values An array containing the coordinate values
/// \param cv_node the control volume node
/// \param subelement The control volume subelement
DICE_LIB_DLL_EXPORT
void tetra4_sub_elem_edge_legths_and_normals(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> cv_node,
  Teuchos::RCP<DICe::mesh::Subelement> subelement);

/// Compute the submesh object from a tri 3 parent element
/// \param coords_values An array with the  coordinates values
/// \param node_A Node A in the diagram for subelements
/// \param node_B Node B in the diagram for subelements
/// \param node_I Node I in the diagram for subelements
/// \param parent_centroid Pointer to an array with the parent element centroid coords
/// \param edge_length The length for this particular edge
/// \param normal Pointer to an array with the normal vector components
/// \param subtri_centroid Pointer to an array with the centroid of this subelement
/// \param subtri_edge_centroid Pointer to an array with the centroid of the edge
DICE_LIB_DLL_EXPORT
void compute_submesh_obj_from_tri3(
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_I,
  const scalar_t * parent_centroid,
  scalar_t & edge_length,
  scalar_t * normal,
  scalar_t * subtri_centroid,
  scalar_t * subtri_edge_centroid);

/// Compute the submesh object from a tet4 parent element
/// \param coords_values An array with the coordinates values
/// \param node_A Node A in the diagram for subelements
/// \param node_B Node B in the diagram for subelements
/// \param node_C Node C in the diagram for subelements
/// \param node_I Node I in the diagram for subelements
/// \param parent_centroid Pointer to an array with the parent element centroid coords
/// \param face_area [out] The area for this particular face
/// \param normal [out] Pointer to an array with the normal vector components
/// \param subtet_centroid [out] Pointer to an array with the centroid of this subelement
/// \param subtet_face_centroid [out] Pointer to an array with the centroid of the face
DICE_LIB_DLL_EXPORT
void compute_submesh_obj_from_tet4(
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_C,
  Teuchos::RCP<DICe::mesh::Node> node_I,
  const scalar_t * parent_centroid,
  scalar_t & face_area,
  scalar_t * normal,
  scalar_t * subtet_centroid,
  scalar_t * subtet_face_centroid);

/// Compute the centroid of a triangle
/// \param coords_values An array with the coordinates values
/// \param node_A Node A in the diagram for subelements
/// \param node_B Node B in the diagram for subelements
/// \param node_I Node I in the diagram for subelements
/// \param cntrd [out] Pointer to the array with the centroid coordinates
DICE_LIB_DLL_EXPORT
void compute_centroid_of_tri(
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_I,
  scalar_t * cntrd);

/// Compute the centroid of a tetrahedron
/// \param coords_values An array with the coordinates values
/// \param node_A Node A in the diagram for subelements
/// \param node_B Node B in the diagram for subelements
/// \param node_C Node C in the diagram for subelements
/// \param node_I Node I in the diagram for subelements
/// \param cntrd [out] Pointer to the array with the centroid coordinates
DICE_LIB_DLL_EXPORT
void compute_centroid_of_tet(
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_C,
  Teuchos::RCP<DICe::mesh::Node> node_I,
  scalar_t * cntrd);

/// Compute the area of a triangle
/// \param coords_values An array with the coordinate values
/// \param node_A Node A from the diagram for subelements
/// \param node_B Node B from the diagram for subelements
/// \param node_C Node C from the diagram for subelements
DICE_LIB_DLL_EXPORT
scalar_t tri3_area(Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_C);

/// Compute the volume of a tetrahedron
/// Compute the area of a triangle
/// \param coords_values An array with the coordinate values
/// \param node_A Node A from the diagram for subelements
/// \param node_B Node B from the diagram for subelements
/// \param node_C Node C from the diagram for subelements
/// \param node_D Node D from the diagram for subelements
DICE_LIB_DLL_EXPORT
scalar_t tetra4_volume(Teuchos::ArrayRCP<const scalar_t> coords_values,
  Teuchos::RCP<DICe::mesh::Node> node_A,
  Teuchos::RCP<DICe::mesh::Node> node_B,
  Teuchos::RCP<DICe::mesh::Node> node_C,
  Teuchos::RCP<DICe::mesh::Node> node_D);

/// Compute the volume and radius of each element
/// \param mesh The computational mesh
DICE_LIB_DLL_EXPORT
void create_cell_size_and_radius(Teuchos::RCP<Mesh> mesh);

/// Compute a specific elemnet type size
/// \param mesh the mesh
/// \param coords_values Array with the coordinate values
/// \param element Pointer to the particular element
DICE_LIB_DLL_EXPORT
void hex8_volume_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element>  element);

/// Compute a specific elemnet type size
/// \param mesh the mesh
/// \param coords_values Array with the coordinate values
/// \param element Pointer to the particular element
DICE_LIB_DLL_EXPORT
void tetra4_volume_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element>  element);

/// Compute a specific elemnet type size
/// \param mesh the mesh
/// \param coords_values Array with the coordinate values
/// \param element Pointer to the particular element
DICE_LIB_DLL_EXPORT
void quad4_area_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element> element);

/// Compute a specific elemnet type size
/// \param mesh the mesh
/// \param coords_values Array with the  coordinate values
/// \param element Pointer to the particular element
DICE_LIB_DLL_EXPORT
void tri3_area_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element> element);

/// Compute a specific elemnet type size
/// \param mesh the mesh
/// \param coords_values Array with the coordinate values
/// \param element Pointer to the particular element
DICE_LIB_DLL_EXPORT
void pyramid5_volume_radius(Teuchos::RCP<Mesh> mesh,
  Teuchos::ArrayRCP<const scalar_t> coords_values,
  const Teuchos::RCP<DICe::mesh::Element> element);

} //mesh
} //DICe

#endif /* DICE_MESHIO_H_ */
