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
#include <DICe_Triangulation.h>

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
  scalar_t errorTol = 1.0E-2;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;


  std::vector<std::vector<scalar_t> > intrinsic_gold =
    {{638.913,407.295,2468.53,2468.25,-0.171198,0.0638413,0,0},
     {628.607,394.571,2377.11,2376.92,0.0897842,0.0619845,0,0}};
  std::vector<std::vector<scalar_t> > T_mat_gold =
    {{0.950892,0.00104338,-0.30952,130.755},
     {-0.00145487,0.999998,-0.00109863,-0.610487},
     {0.309519,0.00149499,0.950892,17.1329},
     {0,0,0,1}};
  std::vector<std::vector<scalar_t> > zero_to_world_xml_gold =
     {{0.987647,0.000580617,-0.156696,65.3774},
      {0.000684129,-1.0,0.00060666,-0.305243},
      {-0.156695,-0.000706366,-0.987647,8.56645},
      {0.0,0.0,0.0,1.0}};
  std::vector<std::vector<scalar_t> > zero_to_world_txt_gold =
    {{1.0,0.0,0.0,0.0},
     {0.0,1.0,0.0,0.0},
     {0.0,0.0,1.0,0.0},
     {0.0,0.0,0.0,1.0}};

  *outStream << "reading calibration parameters from vic3d format" << std::endl;

  Teuchos::RCP<Triangulation> triangulation_xml = Teuchos::rcp(new Triangulation());
  triangulation_xml->load_calibration_parameters("./cal/cal_a.xml");
  std::vector<std::vector<scalar_t> > & calibration_intrinsics_xml = *triangulation_xml->cal_intrinsics();
  std::vector<std::vector<scalar_t> > & calibration_T_mat_xml = * triangulation_xml->cal_extrinsics();
  std::vector<std::vector<scalar_t> > & zero_to_world_xml = * triangulation_xml->trans_extrinsics();

  *outStream << "testing intrinsics from vic3d format" << std::endl;

  if(calibration_intrinsics_xml.size()!=2){
    errorFlag++;
    *outStream << "Error, intrinsics array is the wrong length, should be 2 and is " << calibration_intrinsics_xml.size() << std::endl;
  }
  else{
    if(calibration_intrinsics_xml[0].size()!=8){
      errorFlag++;
      *outStream << "Error, intrinsics array is the wrong width, should be 8 and is " << calibration_intrinsics_xml[0].size() << std::endl;
    }
    else{
      for(size_t i=0;i<intrinsic_gold.size();++i){
        for(size_t j=0;j<intrinsic_gold[0].size();++j){
          if(std::abs(calibration_intrinsics_xml[i][j]-intrinsic_gold[i][j])>errorTol){
            *outStream << "Error, intrinsic value " << i << " " << j << " is not correct. Should be " << intrinsic_gold[i][j] << " is " << calibration_intrinsics_xml[i][j] << std::endl;
            errorFlag++;
          }
        }
      }
    }
  }

  *outStream << "testing T_mat from vic3d format" << std::endl;

  if(calibration_T_mat_xml.size()!=4){
    errorFlag++;
    *outStream << "Error, T_mat array is the wrong length, should be 4 and is " << calibration_T_mat_xml.size() << std::endl;
  }
  else{
    if(calibration_T_mat_xml[0].size()!=4){
      errorFlag++;
      *outStream << "Error, T_mat array is the wrong width, should be 4 and is " << calibration_T_mat_xml[0].size() << std::endl;
    }
    else{
      for(size_t i=0;i<T_mat_gold.size();++i){
        for(size_t j=0;j<T_mat_gold[0].size();++j){
          if(std::abs(calibration_T_mat_xml[i][j]-T_mat_gold[i][j])>errorTol){
            *outStream << "Error, T_mat value " << i << " " << j << " is not correct. Should be " << T_mat_gold[i][j] << " is " << calibration_T_mat_xml[i][j] << std::endl;
            errorFlag++;
          }
        }
      }
    }
  }

  *outStream << "testing camera 0 to world transform from vic3d format" << std::endl;

  if(zero_to_world_xml.size()!=4){
    errorFlag++;
    *outStream << "Error, zero_to_world array is the wrong length, should be 4 and is " << zero_to_world_xml.size() << std::endl;
  }
  else{
    if(zero_to_world_xml[0].size()!=4){
      errorFlag++;
      *outStream << "Error, zero_to_world array is the wrong width, should be 4 and is " << zero_to_world_xml[0].size() << std::endl;
    }
    else{
      for(size_t i=0;i<zero_to_world_xml_gold.size();++i){
        for(size_t j=0;j<zero_to_world_xml_gold[0].size();++j){
          if(std::abs(zero_to_world_xml[i][j]-zero_to_world_xml_gold[i][j])>errorTol){
            *outStream << "Error, zero_to_world value " << i << " " << j << " is not correct. Should be " << zero_to_world_xml_gold[i][j] << " is " << zero_to_world_xml[i][j] << std::endl;
            errorFlag++;
          }
        }
      }
    }
  }

  *outStream << "calibration parameters from vic3d format have been checked" << std::endl;

  *outStream << "reading calibration parameters from text format" << std::endl;

  Teuchos::RCP<Triangulation> triangulation_txt = Teuchos::rcp(new Triangulation());
  triangulation_txt->load_calibration_parameters("./cal/cal_a.txt");
  std::vector<std::vector<scalar_t> > & calibration_intrinsics_txt = *triangulation_txt->cal_intrinsics();
  std::vector<std::vector<scalar_t> > & calibration_T_mat_txt = * triangulation_txt->cal_extrinsics();
  std::vector<std::vector<scalar_t> > & zero_to_world_txt = * triangulation_txt->trans_extrinsics();

  *outStream << "testing intrinsics from txt format" << std::endl;

  if(calibration_intrinsics_txt.size()!=2){
    errorFlag++;
    *outStream << "Error, intrinsics array is the wrong length, should be 2 and is " << calibration_intrinsics_txt.size() << std::endl;
  }
  else{
    if(calibration_intrinsics_txt[0].size()!=8){
      errorFlag++;
      *outStream << "Error, intrinsics array is the wrong width, should be 8 and is " << calibration_intrinsics_txt[0].size() << std::endl;
    }
    else{
      for(size_t i=0;i<intrinsic_gold.size();++i){
        for(size_t j=0;j<intrinsic_gold[0].size();++j){
          if(std::abs(calibration_intrinsics_txt[i][j]-intrinsic_gold[i][j])>errorTol){
            *outStream << "Error, intrinsic value " << i << " " << j << " is not correct. Should be " << intrinsic_gold[i][j] << " is " << calibration_intrinsics_txt[i][j] << std::endl;
            errorFlag++;
          }
        }
      }
    }
  }

  *outStream << "testing T_mat from txt format" << std::endl;

  if(calibration_T_mat_txt.size()!=4){
    errorFlag++;
    *outStream << "Error, T_mat array is the wrong length, should be 4 and is " << calibration_T_mat_txt.size() << std::endl;
  }
  else{
    if(calibration_T_mat_txt[0].size()!=4){
      errorFlag++;
      *outStream << "Error, T_mat array is the wrong width, should be 4 and is " << calibration_T_mat_txt[0].size() << std::endl;
    }
    else{
      for(size_t i=0;i<T_mat_gold.size();++i){
        for(size_t j=0;j<T_mat_gold[0].size();++j){
          if(std::abs(calibration_T_mat_txt[i][j]-T_mat_gold[i][j])>errorTol){
            *outStream << "Error, T_mat value " << i << " " << j << " is not correct. Should be " << T_mat_gold[i][j] << " is " << calibration_T_mat_txt[i][j] << std::endl;
            errorFlag++;
          }
        }
      }
    }
  }

  *outStream << "testing camera 0 to world transform from txt format" << std::endl;

  if(zero_to_world_txt.size()!=4){
    errorFlag++;
    *outStream << "Error, zero_to_world array is the wrong length, should be 4 and is " << zero_to_world_txt.size() << std::endl;
  }
  else{
    if(zero_to_world_txt[0].size()!=4){
      errorFlag++;
      *outStream << "Error, zero_to_world array is the wrong width, should be 4 and is " << zero_to_world_txt[0].size() << std::endl;
    }
    else{
      for(size_t i=0;i<zero_to_world_txt_gold.size();++i){
        for(size_t j=0;j<zero_to_world_txt_gold[0].size();++j){
          if(std::abs(zero_to_world_txt[i][j]-zero_to_world_txt_gold[i][j])>errorTol){
            *outStream << "Error, zero_to_world value " << i << " " << j << " is not correct. Should be " << zero_to_world_txt_gold[i][j] << " is " << zero_to_world_txt[i][j] << std::endl;
            errorFlag++;
          }
        }
      }
    }
  }

  *outStream << "calibration parameters from txt format have been checked" << std::endl;

  *outStream << "testing calibration txt file with custom transform" << std::endl;

  Teuchos::RCP<Triangulation> tri_custom = Teuchos::rcp(new Triangulation());
  tri_custom->load_calibration_parameters("./cal/cal_a_with_transform.txt");
  std::vector<std::vector<scalar_t> > & custom_zero_to_world = * tri_custom->trans_extrinsics();
  *outStream << "testing camera 0 to world transform from txt format with custom transform" << std::endl;

  if(custom_zero_to_world.size()!=4){
    errorFlag++;
    *outStream << "Error, zero_to_world array is the wrong length, should be 4 and is " << custom_zero_to_world.size() << std::endl;
  }
  else{
    if(custom_zero_to_world[0].size()!=4){
      errorFlag++;
      *outStream << "Error, zero_to_world array is the wrong width, should be 4 and is " << custom_zero_to_world[0].size() << std::endl;
    }
    else{
      for(size_t i=0;i<zero_to_world_xml_gold.size();++i){
        for(size_t j=0;j<zero_to_world_xml_gold[0].size();++j){
          if(std::abs(custom_zero_to_world[i][j]-zero_to_world_xml_gold[i][j])>errorTol){
            *outStream << "Error, zero_to_world value " << i << " " << j << " is not correct. Should be " << zero_to_world_xml_gold[i][j] << " is " << custom_zero_to_world[i][j] << std::endl;
            errorFlag++;
          }
        }
      }
    }
  }

  *outStream << "calibration parameters from txt format with custom transform have been checked" << std::endl;

  *outStream << "testing triangulation of 3d points" << std::endl;

  Teuchos::RCP<Triangulation> tri = Teuchos::rcp(new Triangulation());
  tri->load_calibration_parameters("./cal/cal_b.xml");

  scalar_t x_out=0.0,y_out=0.0,z_out=0.0;
  scalar_t x_0 = 190; scalar_t y_0 = 187;
  scalar_t x_1 = 193.8777; scalar_t y_1 = 186.0944;
  tri->triangulate(x_0,y_0,x_1,y_1,x_out,y_out,z_out);
  // TODO check output points
  scalar_t global_x_gold = 46.1199;
  scalar_t global_y_gold = -25.5283;
  scalar_t global_z_gold = -6543.5;
  if(std::abs(global_x_gold - x_out) > errorTol){
    errorFlag++;
    *outStream << "Error, triangulation x coord is wrong. Should be " << global_x_gold << " is " << x_out << std::endl;
  }
  if(std::abs(global_y_gold - y_out) > errorTol){
    errorFlag++;
    *outStream << "Error, triangulation y coord is wrong. Should be " << global_y_gold << " is " << y_out << std::endl;
  }
  if(std::abs(global_z_gold - z_out) > errorTol){
    errorFlag++;
    *outStream << "Error, triangulation z coord is wrong. Should be " << global_z_gold << " is " << z_out << std::endl;
  }

  *outStream << "triangulation of 3d points completed and tested" << std::endl;


  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

