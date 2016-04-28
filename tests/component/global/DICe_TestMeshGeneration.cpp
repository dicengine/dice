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
#include <DICe_Mesh.h>
#include <DICe_MeshIO.h>
#include <DICe_MeshEnums.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

#include <triangle.h>

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

  // define the veritces
  struct triangulateio in, out;
  in.numberofpoints = 4;
  in.numberofpointattributes = 1;
  in.pointattributelist = new REAL[in.numberofpoints*in.numberofpointattributes];
  in.pointlist = new REAL[in.numberofpoints*2];
  in.pointlist[0] = 0.0;
  in.pointlist[1] = 0.0;
  in.pointlist[2] = 1.0;
  in.pointlist[3] = 0.0;
  in.pointlist[4] = 1.0;
  in.pointlist[5] = 10.0;
  in.pointlist[6] = 0.0;
  in.pointlist[7] = 10.0;

  in.pointattributelist[0] = 0.0;
  in.pointattributelist[1] = 1.0;
  in.pointattributelist[2] = 11.0;
  in.pointattributelist[3] = 10.0;
  in.pointmarkerlist = new int[in.numberofpoints];
  in.pointmarkerlist[0] = 0;
  in.pointmarkerlist[1] = 2;
  in.pointmarkerlist[2] = 0;
  in.pointmarkerlist[3] = 0;

  in.numberofsegments = 0;
  in.numberofholes = 0;
  in.numberofregions = 1;
  in.regionlist = new REAL[in.numberofregions * 4];
  in.regionlist[0] = 0.5; // x coordinate of the constraint
  in.regionlist[1] = 5.0; // y coordinate of the constraint
  in.regionlist[2] = 7.0; // regional atribute
  in.regionlist[3] = 0.1; // maximum area

  out.pointlist = (REAL *) NULL;
  out.trianglelist = (int *) NULL;
  out.triangleattributelist = (REAL *) NULL;
  out.pointmarkerlist = (int *) NULL;
  out.segmentlist = (int *) NULL;
  out.segmentmarkerlist = (int *) NULL;
  out.edgelist = (int *) NULL;
  out.edgemarkerlist = (int *) NULL;

  // using the o2 flag here to get quadratic tris with 6 nodes each

  char args[] = "pcAeo2";
  triangulate(args,&in,&out,NULL);

  // grab the node location values and connectivity
  Teuchos::ArrayRCP<int_t> connectivity(out.numberoftriangles*out.numberofcorners); // numberofcorners is num nodes per elem
  for(int_t i=0;i<out.numberoftriangles;++i){
    for (int_t j = 0; j < out.numberofcorners; j++) {
      connectivity[i*out.numberofcorners + j] = out.trianglelist[i * out.numberofcorners + j];
    }
  }
  Teuchos::ArrayRCP<scalar_t> node_coords_x(out.numberofpoints); // numberofcorners is num nodes per elem
  Teuchos::ArrayRCP<scalar_t> node_coords_y(out.numberofpoints); // numberofcorners is num nodes per elem
  for(int_t i=0;i<out.numberofpoints;++i){
      node_coords_x[i] = out.pointlist[i*2+0];
      node_coords_y[i] = out.pointlist[i*2+1];
  }

  Teuchos::RCP<DICe::mesh::Mesh> mesh = DICe::mesh::create_tri6_exodus_mesh(node_coords_x,node_coords_y,connectivity,"scratch_mesh.e");
  *outStream << "creating some fields on the mesh" << std::endl;
  mesh->create_field(mesh::field_enums::CVFEM_AD_PHI_FS);
  mesh->create_field(mesh::field_enums::CVFEM_AD_IMAGE_PHI_FS);
  mesh->create_field(mesh::field_enums::CVFEM_AD_LAMBDA_FS);
  *outStream << "populating values for phi field" << std::endl;
  MultiField & phi = *mesh->get_field(mesh::field_enums::CVFEM_AD_PHI_FS);
  MultiField & coords = *mesh->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  Teuchos::ArrayRCP<const scalar_t> coords_values = coords.get_1d_view();
  const int_t spa_dim = mesh->spatial_dimension();
  for(size_t i=0;i<mesh->num_nodes();++i){
    phi.local_value(i) = coords_values[i*spa_dim]*10.0;
  }
  *outStream << "fields have been created" << std::endl;
  *outStream << "creating the output meshes" << std::endl;
  DICe::mesh::create_output_exodus_file(mesh,"./");
  DICe::mesh::create_exodus_output_variable_names(mesh);
  *outStream << "writing an output step" << std::endl;
  scalar_t time = 0.0;
  DICe::mesh::exodus_output_dump(mesh,1,time);
  *outStream << "closing the exodus output files" << std::endl;
  DICe::mesh::close_exodus_output(mesh);

  delete[] in.pointlist;
  delete[] in.pointattributelist;
  delete[] in.regionlist;
  delete[] in.pointmarkerlist;

  delete[] out.pointlist;
  delete[] out.pointmarkerlist;
  delete[] out.trianglelist;
  delete[] out.triangleattributelist;
  delete[] out.segmentlist;
  delete[] out.segmentmarkerlist;
  delete[] out.edgelist;
  delete[] out.edgemarkerlist;

  // TODO valgrind the final test

  // TODO generate a mesh with a given set of boundary vertices

  // TODO add holes

  // TODO create an exodus mesh given the triangulation

  // TODO export an exodus mesh


  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

