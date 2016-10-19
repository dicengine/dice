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

#include <DICe_Triangulation.h>
#include <DICe_Parser.h>

#include <Teuchos_LAPACK.hpp>

namespace DICe {

void
Triangulation::convert_CB_angles_to_T(const scalar_t & alpha,
  const scalar_t & beta,
  const scalar_t & gamma,
  const scalar_t & tx,
  const scalar_t & ty,
  const scalar_t & tz,
  Teuchos::SerialDenseMatrix<int_t,double> & T_out){
  T_out.reshape(4,4);
  T_out.putScalar(0.0);
  scalar_t cx = std::cos(alpha*DICE_PI/180.0); // input as degrees, need radians
  scalar_t sx = std::sin(alpha*DICE_PI/180.0);
  scalar_t cy = std::cos(beta*DICE_PI/180.0);
  scalar_t sy = std::sin(beta*DICE_PI/180.0);
  scalar_t cz = std::cos(gamma*DICE_PI/180.0);
  scalar_t sz = std::sin(gamma*DICE_PI/180.0);
  T_out(0,0) = cy*cz;
  T_out(0,1) = sx*sy*cz-cx*sz;
  T_out(0,2) = cx*sy*cz+sx*sz;
  T_out(1,0) = cy*sz;
  T_out(1,1) = sx*sy*sz+cx*cz;
  T_out(1,2) = cx*sy*sz-sx*cz;
  T_out(2,0) = -sy;
  T_out(2,1) = sx*cy;
  T_out(2,2) = cx*cy;
  T_out(0,3) = tx;
  T_out(1,3) = ty;
  T_out(2,3) = tz;
  T_out(3,3) = 1.0;
}

void
Triangulation::invert_transform(Teuchos::SerialDenseMatrix<int_t,double> & T_out){
  Teuchos::LAPACK<int_t,double> lapack;
  int *IPIV = new int[5];
  int LWORK = 16;
  int INFO = 0;
  double *WORK = new double[LWORK];
  lapack.GETRF(4,4,T_out.values(),4,IPIV,&INFO);
  for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
  try
  {
    lapack.GETRI(4,T_out.values(),4,IPIV,WORK,LWORK,&INFO);
  }
  catch(std::exception &e){
    DEBUG_MSG( e.what() << '\n');
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
      "Error, could not invert the transformation matrix from camera 0");
  }
  delete [] IPIV;
  delete [] WORK;
}

void
Triangulation::load_calibration_parameters(const std::string & param_file_name){
  DEBUG_MSG("Triangulation::load_calibration_parameters(): begin");
  TEUCHOS_TEST_FOR_EXCEPTION(has_cal_params_,std::runtime_error,
    "Error, calibration parameters have already been loaded, repeat call to load_calibration_parameters()");

  DEBUG_MSG("Triangulation::load_calibration_parameters(): Parsing calibration parameters from file: " << param_file_name);
  std::fstream dataFile(param_file_name.c_str(), std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(), std::runtime_error,
    "Error, the calibration xml file does not exist or is corrupt: " << param_file_name);

  cal_extrinsics_.clear();
  for(int_t i=0;i<4;++i)
    cal_extrinsics_.push_back(std::vector<scalar_t>(4,0.0));

  trans_extrinsics_.clear();
  for(int_t i=0;i<4;++i){
    trans_extrinsics_.push_back(std::vector<scalar_t>(4,0.0));
    trans_extrinsics_[i][i] = 1.0; // initialize to the identity tensor (no transformation)
  }

  // intrinsic parameters from both vic3d and the generic text reader are the same and in this order
  // cx cy fx fy fs k0 k1 k2
  cal_intrinsics_.clear();
  for(int_t i=0;i<2;++i)
    cal_intrinsics_.push_back(std::vector<scalar_t>(8,0.0));

//  Teuchos::LAPACK<int_t,double> lapack;

  const std::string xml("xml");
  const std::string txt("txt");
  if(param_file_name.find(xml)!=std::string::npos){
    DEBUG_MSG("Triangulation::load_calibration_parameters(): calibration file is vic3D xml format");
    // cal.xml file can't be ready by Teuchos parser because it has a !DOCTYPE
    // have to manually read the file here, lots of assumptions in how the file is formatted
    // FIXME do this a little more robustly
    // camera orientation for each camera in vic3d is in terms of the world to camera
    // orientation and the order of variables is alpha beta gamma tx ty tz (the Cardan Bryant angles + translations)
    std::vector<std::vector<scalar_t> > cam_orientation(2,std::vector<scalar_t>(6,0.0));
    // has to be double for lapack calls
    std::vector<Teuchos::SerialDenseMatrix<int_t,double> > vec_of_Ts(2,Teuchos::SerialDenseMatrix<int_t,double>(4,4,true));
    int_t camera_index = 0;
    // read each line of the file
    while (!dataFile.eof())
    {
      Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile," \t<>");
      if(tokens.size()==0) continue;
      if(tokens[0]!="CAMERA") continue;
      assert(camera_index<2);
      assert(tokens.size()>17);
      int_t coeff_index = 0;
      for(int_t i=2;i<=9;++i)
        cal_intrinsics_[camera_index][coeff_index++] = strtod(tokens[i].c_str(),NULL);
      coeff_index = 0;
      for(int_t i=11;i<=16;++i){
        cam_orientation[camera_index][coeff_index] = strtod(tokens[i].c_str(),NULL);
        DEBUG_MSG("Triangulation::load_calibration_parameters(): camera " << camera_index << " orientation " <<
          coeff_index << " " << cam_orientation[camera_index][coeff_index]);
        coeff_index++;
      }
      // convert the Cardan-Bryant angles to the rotation matrix for each camera
      convert_CB_angles_to_T(cam_orientation[camera_index][0],
        cam_orientation[camera_index][1],
        cam_orientation[camera_index][2],
        cam_orientation[camera_index][3],
        cam_orientation[camera_index][4],
        cam_orientation[camera_index][5],
        vec_of_Ts[camera_index]);
      camera_index++;
    } // end file read
//    std::cout << " R0 matrix " << std::endl;
//    for(int_t i=0;i<4;++i){
//      for(int_t j=0;j<4;++j){
//        std::cout << vec_of_Ts[0](i,j) << " ";
//      }
//      std::cout << std::endl;
//    }
//    std::cout << " R1 matrix " << std::endl;
//    for(int_t i=0;i<4;++i){
//      for(int_t j=0;j<4;++j){
//        std::cout << vec_of_Ts[1](i,j) << " ";
//      }
//      std::cout << std::endl;
//    }
    // compute the inverse of camera 0
    invert_transform(vec_of_Ts[0]);
    // store the T matrix from camera 0 post-inversion as the trans_extrinsics (the camera_0 to world coordinates transform)
    for(int_t i=0;i<4;++i){
      for(int_t j=0;j<4;++j){
        trans_extrinsics_[i][j] = vec_of_Ts[0](i,j);
      }
    }
    // multiply the two tranformation matrices to get the left to right transform
    for(int_t i=0;i<4;++i){
      for(int_t j=0;j<4;++j){
        for(int_t k=0;k<4;++k){
          cal_extrinsics_[i][j] += vec_of_Ts[1](i,k)*vec_of_Ts[0](k,j);
        }
      }
    }
  }
  else if(param_file_name.find(txt)!=std::string::npos){
    DEBUG_MSG("Triangulation::load_calibration_parameters(): calibration file is generic txt format");
    const int_t num_values_expected = 22;
    const int_t num_values_with_custom_transform = 28;
    int_t num_values = 0;
    for(int_t i=0;i<4;++i){
      trans_extrinsics_[i][i] = 1.0; // default transformation is the identity tensor
    }
    std::vector<scalar_t> extrinsics(6,0.0);
    std::vector<scalar_t> trans_extrinsics(6,0.0);
    while (!dataFile.eof())
    {
      Teuchos::ArrayRCP<std::string> tokens = tokenize_line(dataFile," \t<>");
      if(tokens.size()==0) continue;
      if(tokens[0]=="#") continue;
      if(tokens.size() > 1){
        assert(tokens[1]=="#"); // only one entry per line plus comments
      }
      const int_t camera_index = num_values >= 8 ? 1 : 0;
      if(num_values < 16)
        cal_intrinsics_[camera_index][num_values - camera_index*8] = strtod(tokens[0].c_str(),NULL);
      else if(num_values < 22)
        extrinsics[num_values - 16] = strtod(tokens[0].c_str(),NULL);
      else
        trans_extrinsics[num_values - 22] = strtod(tokens[0].c_str(),NULL);
      num_values++;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(num_values!=num_values_expected&&num_values!=num_values_with_custom_transform,std::runtime_error,
      "Error reading calibration text file " << param_file_name);
    Teuchos::SerialDenseMatrix<int_t,double> converted_extrinsics(4,4,true);
    convert_CB_angles_to_T(extrinsics[0],
      extrinsics[1],
      extrinsics[2],
      extrinsics[3],
      extrinsics[4],
      extrinsics[5],
      converted_extrinsics);
    for(int_t i=0;i<4;++i)
      for(int_t j=0;j<4;++j)
        cal_extrinsics_[i][j] = converted_extrinsics(i,j);

    if(num_values==num_values_with_custom_transform){
      DEBUG_MSG("Triangulation::load_calibration_parameters(): loading custom transform from camera 0 to world coordinates");
      Teuchos::SerialDenseMatrix<int_t,double> converted_extrinsics(4,4,true);
      convert_CB_angles_to_T(trans_extrinsics[0],
        trans_extrinsics[1],
        trans_extrinsics[2],
        trans_extrinsics[3],
        trans_extrinsics[4],
        trans_extrinsics[5],
        converted_extrinsics);
      invert_transform(converted_extrinsics);
      for(int_t i=0;i<4;++i)
        for(int_t j=0;j<4;++j)
          trans_extrinsics_[i][j] = converted_extrinsics(i,j);
    }
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
      "Error, unrecognized calibration parameters file format: " << param_file_name);
  }
  for(int_t i=0;i<2;++i){
    DEBUG_MSG("Triangulation::load_calibration_parameters(): camera " << i << " intrinsic parameters");
    DEBUG_MSG("Triangulation::load_calibration_parameters(): cx " << cal_intrinsics_[i][0]);
    DEBUG_MSG("Triangulation::load_calibration_parameters(): cy " << cal_intrinsics_[i][1]);
    DEBUG_MSG("Triangulation::load_calibration_parameters(): fx " << cal_intrinsics_[i][2]);
    DEBUG_MSG("Triangulation::load_calibration_parameters(): fy " << cal_intrinsics_[i][3]);
    DEBUG_MSG("Triangulation::load_calibration_parameters(): fs " << cal_intrinsics_[i][4]);
    DEBUG_MSG("Triangulation::load_calibration_parameters(): k1 " << cal_intrinsics_[i][5]);
    DEBUG_MSG("Triangulation::load_calibration_parameters(): k2 " << cal_intrinsics_[i][6]);
    DEBUG_MSG("Triangulation::load_calibration_parameters(): k3 " << cal_intrinsics_[i][7]);
  }
  DEBUG_MSG("Triangulation::load_calibration_parameters(): extrinsic T mat from camera 0 to camera 1");
  for(int_t i=0;i<4;++i){
    DEBUG_MSG("Triangulation::load_calibration_parameters(): " << cal_extrinsics_[i][0] <<
      " " << cal_extrinsics_[i][1] << " " << cal_extrinsics_[i][2] << " " << cal_extrinsics_[i][3]);
  }
  DEBUG_MSG("Triangulation::load_calibration_parameters(): transform mat from camera 0 to world");
  for(int_t i=0;i<4;++i){
    DEBUG_MSG("Triangulation::load_calibration_parameters(): " << trans_extrinsics_[i][0] <<
      " " << trans_extrinsics_[i][1] << " " << trans_extrinsics_[i][2] << " " << trans_extrinsics_[i][3]);
  }
  has_cal_params_ = true;
  DEBUG_MSG("Triangulation::load_calibration_parameters(): end");
}

void Triangulation::triangulate(const scalar_t & x0,
  const scalar_t & y0,
  const scalar_t & x1,
  const scalar_t & y1,
  scalar_t & x_out,
  scalar_t & y_out,
  scalar_t & z_out){
  DEBUG_MSG("Triangulation::triangulate(): camera 0 sensor coords " << x0 << " " << y0 << " camera 1 sensor coords " << x1 << " " << y1);
  static Teuchos::SerialDenseMatrix<int_t,double> M(4,3,true);
  static Teuchos::SerialDenseMatrix<int_t,double> MTM(3,3,true);
  static Teuchos::SerialDenseMatrix<int_t,double> MTMMT(3,4,true);
  static Teuchos::LAPACK<int_t,double> lapack;
  static std::vector<scalar_t> r(4,0.0);
  static std::vector<scalar_t> XYZc0(4,0.0); // camera 0 coords
  static std::vector<scalar_t> XYZ(4,0.0); // world coords
  static scalar_t cmx = 0.0;
  static scalar_t cmy = 0.0;
  static std::vector<int> IPIV(4,0.0);
  int * IPIV_ptr = &IPIV[0];
  static int LWORK = 9;
  static std::vector<double> WORK(LWORK,0.0);
  double * WORK_ptr = &WORK[0];
  static int INFO = 0;

  // clear the storage
  for(int_t i=0;i<3;++i){
    for(int_t j=0;j<4;++j){
      M(j,i) = 0.0;
      MTMMT(i,j) = 0.0;
    }
    for(int_t j=0;j<3;++j)
      MTM(j,i) = 0.0;
  }
  for(int_t i=0;i<4;++i){
    XYZc0[i] = 0.0;
    XYZ[i] = 0.0;
    IPIV[i] = 0;
    r[i] = 0.0;
  }
  for(int_t i=0;i<LWORK;++i)
    WORK[i] = 0.0;

  // calculate the M matrix
  M(0,0) = cal_intrinsics_[0][2]; // fx0
  M(0,1) = cal_intrinsics_[0][4]; // fs0
  M(0,2) = cal_intrinsics_[0][0] - x0; // cx1 - xs1
  M(1,1) = cal_intrinsics_[0][3]; // fy1
  M(1,2) = cal_intrinsics_[0][1] - y0; // cy1 - ys1
  cmx = cal_intrinsics_[1][0] - x1; // cx2 - xs2
  cmy = cal_intrinsics_[1][1] - y1; // cy2 - ys2
  // (cx2-xs2)*R31 + fx2*R11 + fs2*R21
  M(2,0) = cmx*cal_extrinsics_[2][0] + cal_intrinsics_[1][2]*cal_extrinsics_[0][0] + cal_intrinsics_[1][4]*cal_extrinsics_[1][0];
  // (cx2-xs2)*R32 + fx2*R12 + fs2*R22
  M(2,1) = cmx*cal_extrinsics_[2][1] + cal_intrinsics_[1][2]*cal_extrinsics_[0][1] + cal_intrinsics_[1][4]*cal_extrinsics_[1][1];
  // (cx2-xs2)*R33 + fx2*R13 + fs2*R23
  M(2,2) = cmx*cal_extrinsics_[2][2] + cal_intrinsics_[1][2]*cal_extrinsics_[0][2] + cal_intrinsics_[1][4]*cal_extrinsics_[1][2];
  // (cy2-ys2)*R31 + fy2*R21
  M(3,0) = cmy*cal_extrinsics_[2][0] + cal_intrinsics_[1][3]*cal_extrinsics_[1][0];
  // (cy2-ys2)*R32 + fy2*R22
  M(3,1) = cmy*cal_extrinsics_[2][1] + cal_intrinsics_[1][3]*cal_extrinsics_[1][1];
  // (cy2-ys2)*R33 + fy2*R23
  M(3,2) = cmy*cal_extrinsics_[2][2] + cal_intrinsics_[1][3]*cal_extrinsics_[1][2];
  //-fx2tx - fs2ty -(cx2-xs2)*tz
  r[2] = -cal_intrinsics_[1][2]*cal_extrinsics_[0][3] - cal_intrinsics_[1][4]*cal_extrinsics_[1][3] - cmx*cal_extrinsics_[2][3];
  //-fy2ty -(cy2-ys2)*tz
  r[3] = -cal_intrinsics_[1][3]*cal_extrinsics_[1][3] - cmy*cal_extrinsics_[2][3];

//  std::cout << " M matrix: " << std::endl;
//  for(int_t i=0;i<4;++i){
//    for(int_t j=0;j<3;++j){
//      std::cout << M(i,j) << " ";
//    }
//    std::cout << std::endl;
//  }

  // compute M^TM
  for(int_t i=0;i<3;++i){
    for(int_t j=0;j<3;++j){
      for(int_t k=0;k<4;++k){
        MTM(i,j) += M(k,i)*M(k,j);
      }
    }
  }
  // compute the inverse of M^TM
  lapack.GETRF(3,3,MTM.values(),3,IPIV_ptr,&INFO);
  try
  {
    lapack.GETRI(3,MTM.values(),3,IPIV_ptr,WORK_ptr,LWORK,&INFO);
  }
  catch(std::exception &e){
    DEBUG_MSG( e.what() << '\n');
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
      "Error, could not invert the M matrix in triangulation");
  }
  // now MTM is inverted
  // compute MTM*MT
  for(int_t i=0;i<3;++i){
    for(int_t j=0;j<4;++j){
      for(int_t k=0;k<3;++k){
        MTMMT(i,j) += MTM(i,k)*M(j,k);
      }
    }
  }

  // compute the 3d point
  for(int_t i=0;i<3;++i){
    for(int_t j=0;j<4;++j){
      XYZc0[i] += MTMMT(i,j)*r[j];
    }
  }
  XYZc0[3] = 1.0;
  DEBUG_MSG("Triangulation::triangulate(): camera 0 coordinates X " << XYZc0[0] << " Y " << XYZc0[1] << " Z "  << XYZc0[2]);

  // apply the camera 0 to world coord transform
  for(int_t i=0;i<4;++i){
    for(int_t j=0;j<4;++j){
      XYZ[i] += trans_extrinsics_[i][j]*XYZc0[j];
    }
  }
  x_out = XYZ[0];
  y_out = XYZ[1];
  z_out = XYZ[2];
  DEBUG_MSG("Triangulation::triangulate(): world coordinates X " << x_out << " Y " << y_out << " Z "  << z_out);
}



}// End DICe Namespace
