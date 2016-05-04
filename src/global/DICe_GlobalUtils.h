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
#ifndef DICE_GLOBALUTILS_H
#define DICE_GLOBALUTILS_H

#include <DICe_Schema.h>

namespace DICe {

namespace global{

void initialize_exodus_output(Schema * schema,
  const std::string & output_folder);

Status_Flag execute_global_step(Schema * schema);

void calc_jacobian(const scalar_t * xcap,
  const scalar_t * DN,
  scalar_t * jacobian,
  scalar_t * inv_jacobian,
  scalar_t & J,
  int_t num_elem_nodes,
  int_t dim );

void calc_B(const scalar_t * DN,
  const scalar_t * inv_jacobian,
  const int_t num_elem_nodes,
  const int_t dim,
  scalar_t * solid_B);

void calc_mms_force_elasticity(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & alpha,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & force_x,
  scalar_t & force_y);

void calc_mms_bc_elasticity(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  const scalar_t & m,
  scalar_t & b_x,
  scalar_t & b_y);

void calc_mms_bc_simple(const scalar_t & x,
  const scalar_t & y,
  scalar_t & b_x,
  scalar_t & b_y);

void calc_mms_force_simple(const scalar_t & alpha,
  scalar_t & force_x,
  scalar_t & force_y);

void calc_mms_bc_2(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & L,
  scalar_t & b_x,
  scalar_t & b_y);

}// end global namespace

}// End DICe Namespace

#endif
