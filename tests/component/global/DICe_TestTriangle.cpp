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
  if(mesh->num_node_sets()!=3){
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

