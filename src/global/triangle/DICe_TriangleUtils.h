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
#ifndef DICE_TRIANGLEUTILS_H
#define DICE_TRIANGLEUTILS_H

#include <DICe.h>
#include <DICe_Mesh.h>

#include <Teuchos_RCP.hpp>

namespace DICe {

DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::mesh::Mesh> generate_tri_mesh(const DICe::mesh::Base_Element_Type elem_type,
  Teuchos::ArrayRCP<scalar_t> points_x,
  Teuchos::ArrayRCP<scalar_t> points_y,
  const scalar_t & max_size_constraint,
  const std::string & output_file_name,
  const bool enforce_lagrange_bc=true,
  const bool use_regular_grid=false);

DICE_LIB_DLL_EXPORT
Teuchos::RCP<DICe::mesh::Mesh> generate_tri_mesh(const DICe::mesh::Base_Element_Type elem_type,
  const std::string & roi_file_name,
  const scalar_t & max_size_constraint,
  const std::string & output_file_name);

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
  const bool enforce_lagrange_bc=false,
  const bool use_regular_grid=false);

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
  const bool enforce_lagrange_bc=false);


}// End DICe Namespace

#endif
