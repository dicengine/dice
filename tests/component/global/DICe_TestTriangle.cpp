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

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <iostream>

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

  const scalar_t max_size_constraint = 75.0;

  Teuchos::ArrayRCP<scalar_t> points_x(12);
  Teuchos::ArrayRCP<scalar_t> points_y(12);
  // outer boundary points
  points_x[0] = 0.0;   points_y[0] = 0.0;
  points_x[1] = 100.0; points_y[1] = 0.0;
  points_x[2] = 100.0; points_y[2] = 50.0;
  points_x[3] = 0.0;   points_y[3] = 50.0;
  // interior boundary points
  points_x[4] = 40.0; points_y[4] = 15.0;
  points_x[5] = 60.0; points_y[5] = 15.0;
  points_x[6] = 60.0; points_y[6] = 35.0;
  points_x[7] = 40.0; points_y[7] = 35.0;
  points_x[8] = 5.0;  points_y[8] = 40.0;
  points_x[9] = 10.0; points_y[9] = 40.0;
  points_x[10] = 10.0;points_y[10] = 45.0;
  points_x[11] = 5.0; points_y[11] = 45.0;

  // define the boundary edges:
  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_left(2);
  Teuchos::ArrayRCP<int_t> dirichlet_boundary_segments_right(2);
  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_left(10);
  Teuchos::ArrayRCP<int_t> neumann_boundary_segments_right(10);
  dirichlet_boundary_segments_left[0] = 1; dirichlet_boundary_segments_right[0] = 4;
  dirichlet_boundary_segments_left[1] = 2; dirichlet_boundary_segments_right[1] = 3;
  neumann_boundary_segments_left[0] = 5; neumann_boundary_segments_right[0] = 6;
  neumann_boundary_segments_left[1] = 6; neumann_boundary_segments_right[1] = 7;
  neumann_boundary_segments_left[2] = 7; neumann_boundary_segments_right[2] = 8;
  neumann_boundary_segments_left[3] = 8; neumann_boundary_segments_right[3] = 5;
  neumann_boundary_segments_left[4] = 9; neumann_boundary_segments_right[4] = 10;
  neumann_boundary_segments_left[5] = 10; neumann_boundary_segments_right[5] = 11;
  neumann_boundary_segments_left[6] = 11; neumann_boundary_segments_right[6] = 12;
  neumann_boundary_segments_left[7] = 12; neumann_boundary_segments_right[7] = 9;
  neumann_boundary_segments_left[8] = 1; neumann_boundary_segments_right[8] = 2;
  neumann_boundary_segments_left[9] = 4; neumann_boundary_segments_right[9] = 3;

  // holes in the mesh
  Teuchos::ArrayRCP<scalar_t> holes_x(2);
  Teuchos::ArrayRCP<scalar_t> holes_y(2);
  holes_x[0] = 50.0; holes_y[0] = 25.0;
  holes_x[1] = 7.5;  holes_y[1] = 42.5;

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
    "triangle_test_mesh.e");
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

