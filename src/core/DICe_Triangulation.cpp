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
#include <DICe_Simplex.h>
#include <DICe_Parser.h>

#include <Teuchos_LAPACK.hpp>

#include <fstream>

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
  TEUCHOS_TEST_FOR_EXCEPTION(cal_intrinsics_[0][0]<=0.0,std::runtime_error,"Error, invalid cx for camera 0" << cal_intrinsics_[0][0]);
  TEUCHOS_TEST_FOR_EXCEPTION(cal_intrinsics_[0][1]<=0.0,std::runtime_error,"Error, invalid cy for camera 0" << cal_intrinsics_[0][1]);
  TEUCHOS_TEST_FOR_EXCEPTION(cal_intrinsics_[1][0]<=0.0,std::runtime_error,"Error, invalid cx for camera 1" << cal_intrinsics_[1][0]);
  TEUCHOS_TEST_FOR_EXCEPTION(cal_intrinsics_[1][1]<=0.0,std::runtime_error,"Error, invalid cy for camera 1" << cal_intrinsics_[1][1]);

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
  DEBUG_MSG("Triangulation::load_calibration_parameters(): end");
}


void
// note these fields should not be the overlap fields! but the distributed ones (1-to-1 map)
Triangulation::best_fit_plane(Teuchos::RCP<MultiField> & cx,
  Teuchos::RCP<MultiField> & cy,
  Teuchos::RCP<MultiField> & cz,
  Teuchos::RCP<MultiField> & sigma){

  // create an all on all map for the K and F coeffs:
  // create all on zero map
  MultiField_Comm comm = cx->get_map()->get_comm();
  const int_t num_entries = 9; // k11 k12 k13 k22 k23 k33 f1 f2 f3
  const int_t num_coeffs = 12; // R11 R12 R13 R21 R22 R23 R31 R32 R33 tx ty tz
  Teuchos::Array<int_t> all_on_all_ids(num_entries);
  for(int_t i=0;i<num_entries;++i)
    all_on_all_ids[i] = i;
  Teuchos::RCP<MultiField_Map> all_on_all_map = Teuchos::rcp (new MultiField_Map(-1, all_on_all_ids,0,comm));
  Teuchos::RCP<MultiField> all_entries = Teuchos::rcp(new MultiField(all_on_all_map,1,true));

  Teuchos::Array<int_t> all_on_all_coeff_ids(num_coeffs);
  for(int_t i=0;i<num_coeffs;++i)
    all_on_all_coeff_ids[i] = i;
  Teuchos::RCP<MultiField_Map> all_on_all_coeff_map = Teuchos::rcp (new MultiField_Map(-1, all_on_all_coeff_ids,0,comm));
  Teuchos::RCP<MultiField> all_coeffs = Teuchos::rcp(new MultiField(all_on_all_coeff_map,1,true));

  const int_t num_local_points = cx->get_map()->get_num_local_elements();
  for(int_t i=0;i<num_local_points;++i){
    if(sigma->local_value(i)<0.0) continue;
    const scalar_t x = cx->local_value(i);
    const scalar_t y = cy->local_value(i);
    const scalar_t z = cz->local_value(i);
    all_entries->local_value(0) += x*x;
    all_entries->local_value(1) += x*y;
    all_entries->local_value(2) += x;
    all_entries->local_value(3) += y*y;
    all_entries->local_value(4) += y;
    all_entries->local_value(5) += 1.0;
    all_entries->local_value(6) += -1.0*x*z;
    all_entries->local_value(7) += -1.0*y*z;
    all_entries->local_value(8) += -1.0*z;
  }
  // broadcast the values to processor 0
  Teuchos::Array<int_t> all_on_zero_ids;
  Teuchos::Array<int_t> all_on_zero_coeff_ids;
  if(comm.get_rank()==0){
    all_on_zero_ids.resize(num_entries);
    all_on_zero_coeff_ids.resize(num_coeffs);
    for(int_t i=0;i<num_entries;++i)
       all_on_zero_ids[i] = i;
    for(int_t i=0;i<num_coeffs;++i)
       all_on_zero_coeff_ids[i] = i;
  }
  Teuchos::RCP<MultiField_Map> all_on_zero_map = Teuchos::rcp (new MultiField_Map(-1, all_on_zero_ids, 0, comm));
  Teuchos::RCP<MultiField> all_on_zero_entries = Teuchos::rcp( new MultiField(all_on_zero_map,1,true));
  Teuchos::RCP<MultiField_Map> all_on_zero_coeff_map = Teuchos::rcp (new MultiField_Map(-1, all_on_zero_coeff_ids, 0, comm));
  Teuchos::RCP<MultiField> all_on_zero_coeffs = Teuchos::rcp( new MultiField(all_on_zero_coeff_map,1,true));
  MultiField_Exporter exporter(*all_on_all_map,*all_on_zero_map);
  // export the field to zero
  all_on_zero_entries->do_export(all_entries,exporter,ADD);
  // compute the plane coefficients on process 0 only, then broadcast
  if(comm.get_rank()==0){
    Teuchos::SerialDenseMatrix<int_t,double> K(3,3,true);
    std::vector<scalar_t> F(3,0.0);
    std::vector<scalar_t> u(3,0.0);
    K(0,0) = all_on_zero_entries->local_value(0);
    K(0,1) = all_on_zero_entries->local_value(1);
    K(0,2) = all_on_zero_entries->local_value(2);
    K(1,0) = all_on_zero_entries->local_value(1);
    K(1,1) = all_on_zero_entries->local_value(3);
    K(1,2) = all_on_zero_entries->local_value(4);
    K(2,0) = all_on_zero_entries->local_value(2);
    K(2,1) = all_on_zero_entries->local_value(4);
    K(2,2) = all_on_zero_entries->local_value(5);
    F[0] = all_on_zero_entries->local_value(6);
    F[1] = all_on_zero_entries->local_value(7);
    F[2] = all_on_zero_entries->local_value(8);
    Teuchos::LAPACK<int_t,double> lapack;
    int *IPIV = new int[4];
    int LWORK = 9;
    int INFO = 0;
    double *WORK = new double[LWORK];
    lapack.GETRF(3,3,K.values(),3,IPIV,&INFO);
    for(int_t i=0;i<LWORK;++i) WORK[i] = 0.0;
    try
    {
      lapack.GETRI(3,K.values(),3,IPIV,WORK,LWORK,&INFO);
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
        "Error, could not invert the K matrix in best fit plane");
    }
    delete [] IPIV;
    delete [] WORK;
    // determine the coefficients
    // z = -1*(u[0]*x + u[1]*y + u[3]), equation of a plane
    for(int_t i=0;i<3;++i){
      for(int_t j=0;j<3;++j){
        u[i] += K(i,j)*F[j];
      }
    }
    for(int_t i=0;i<3;++i){
      DEBUG_MSG("[proc " << comm.get_rank() << "] best_fit_plane coeff " << i << " " << u[i]);
    }
    std::FILE * filePtr = fopen("best_fit_plane_out.dat","w");
    fprintf(filePtr,"# Best fit plane equation coefficients (z = -1*(c[0]*x + c[1]*y + c[2]) ): \n");
    fprintf(filePtr,"%e\n",u[0]);
    fprintf(filePtr,"%e\n",u[1]);
    fprintf(filePtr,"%e\n",u[2]);

    // read in origin in image left coordinates
    std::vector<int_t> fit_def_x_left(2,0);
    std::vector<int_t> fit_def_y_left(2,0);
    std::vector<scalar_t> fit_def_x_right(2,0.0);
    std::vector<scalar_t> fit_def_y_right(2,0.0);
    std::fstream bestFitDataFile("best_fit_plane.dat", std::ios_base::in);
    TEUCHOS_TEST_FOR_EXCEPTION(!bestFitDataFile.good(),std::runtime_error,
      "Error, could not open file best_fit_plane.dat (required to project output to best fit plane)");
    int_t line = 0;
    while(!bestFitDataFile.eof()){
      Teuchos::ArrayRCP<std::string> tokens = tokenize_line(bestFitDataFile);
      if(tokens.size()==0) continue;
      TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=2,std::runtime_error,
        "Error reading best_fit_plane.dat, should be 2 values per line (x_left y_left for origin and point on x axis),"
          " but found " << tokens.size() << " values on one line");
      assert(line<(int_t)fit_def_x_left.size());
      fit_def_x_left[line] = atoi(tokens[0].c_str());
      fit_def_y_left[line] = atoi(tokens[1].c_str());
      line++;
    }
    DEBUG_MSG("Best fit plane origin (left sensor coords):          " << fit_def_x_left[0] << " " << fit_def_y_left[0]);
    DEBUG_MSG("Best fit plane point on x axis (left sensor coords): " << fit_def_x_left[1] << " " << fit_def_y_left[1]);
    scalar_t xr = 0.0;
    scalar_t yr = 0.0;
    for(int_t i=0;i<2;++i){
      project_left_to_right_sensor_coords(fit_def_x_left[i],fit_def_y_left[i],xr,yr);
      fit_def_x_right[i] = xr;
      fit_def_y_right[i] = yr;
    }
    DEBUG_MSG("Best fit plane origin (right sensor coords):          " << fit_def_x_right[0] << " " << fit_def_y_right[0]);
    DEBUG_MSG("Best fit plane point on x axis (right sensor coords): " << fit_def_x_right[1] << " " << fit_def_y_right[1]);
    // triangulate the two points to get the origin and x-axis in model coordinates
    scalar_t oX = 0.0; // origin point
    scalar_t oY = 0.0;
    scalar_t oZ = 0.0;
    scalar_t oXw = 0.0;
    scalar_t oYw = 0.0;
    scalar_t oZw = 0.0;
    triangulate(fit_def_x_left[0],fit_def_y_left[0],fit_def_x_right[0],fit_def_y_right[0],oX,oY,oZ,oXw,oYw,oZw);
    // convert to planar coordinates
    oZw = u[0]*oXw + u[1]*oYw + u[2];
    oZw *=-1.0;
    scalar_t xaX = 0.0; // origin point
    scalar_t xaY = 0.0;
    scalar_t xaZ = 0.0;
    scalar_t xaXw = 0.0;
    scalar_t xaYw = 0.0;
    scalar_t xaZw = 0.0;
    triangulate(fit_def_x_left[1],fit_def_y_left[1],fit_def_x_right[1],fit_def_y_right[1],xaX,xaY,xaZ,xaXw,xaYw,xaZw);
    // convert to planar coordinates
    xaZw = u[0]*xaXw + u[1]*xaYw + u[2];
    xaZw *= -1.0;

    fprintf(filePtr,"# Best fit origin in model coordinates \n");
    fprintf(filePtr,"%e\n",oXw);
    fprintf(filePtr,"%e\n",oYw);
    fprintf(filePtr,"%e\n",oZw);
    fprintf(filePtr,"# Best fit point along x axis in model coordinates \n");
    fprintf(filePtr,"%e\n",xaXw);
    fprintf(filePtr,"%e\n",xaYw);
    fprintf(filePtr,"%e\n",xaZw);

    // the world coordinates are obtained from the standard basis vectors e1 = 1 0 0, e2 = 0 1 0, e3 = 0 0 1
    std::vector<scalar_t> e1(3,0.0);
    e1[0] = 1.0;
    std::vector<scalar_t> e2(3,0.0);
    e2[1] = 1.0;
    std::vector<scalar_t> e3(3,0.0);
    e3[2] = 1.0;

    // the best fit plane coordinates are determined by the basis vectors g1 g2 g3
    // g1 is from the origin to the point on the x-axis
    std::vector<scalar_t> g1(3,0.0);
    g1[0] = xaXw - oXw;
    g1[1] = xaYw - oYw;
    g1[2] = xaZw - oZw;
    // g3 is the normal vector on the plane
    std::vector<scalar_t> g3(3,0.0);
    g3[0] = u[0];
    g3[1] = u[1];
    g3[2] = 1.0;
    // g2 is obtained by the cross product of g3 with g1
    std::vector<scalar_t> g2(3,0.0);
    g2[0] = g3[1]*g1[2] - g1[1]*g3[2];
    g2[1] = g3[2]*g1[0] - g1[2]*g3[0];
    g2[2] = g3[0]*g1[1] - g1[0]*g3[1];

    // the rotation matrix components are the cosines of the angles between the basis vectors
    all_on_zero_coeffs->local_value(0) = cosine_of_two_vectors(e1,g1);
    all_on_zero_coeffs->local_value(1) = cosine_of_two_vectors(e1,g2);
    all_on_zero_coeffs->local_value(2) = cosine_of_two_vectors(e1,g3);
    all_on_zero_coeffs->local_value(3) = cosine_of_two_vectors(e2,g1);
    all_on_zero_coeffs->local_value(4) = cosine_of_two_vectors(e2,g2);
    all_on_zero_coeffs->local_value(5) = cosine_of_two_vectors(e2,g3);
    all_on_zero_coeffs->local_value(6) = cosine_of_two_vectors(e3,g1);
    all_on_zero_coeffs->local_value(7) = cosine_of_two_vectors(e3,g2);
    all_on_zero_coeffs->local_value(8) = cosine_of_two_vectors(e3,g3);
    // the translation is simply given by the origin of the transform;
    all_on_zero_coeffs->local_value(9) = oXw;
    all_on_zero_coeffs->local_value(10) = oYw;
    all_on_zero_coeffs->local_value(11) = oZw;
    fprintf(filePtr,"# Transform from current world coords to best fit plane origin and bases \n");
    fprintf(filePtr,"# R11 R12 R13 R21 R22 R23 R31 R32 R33 tx ty tz \n");
    for(int_t i=0;i<num_coeffs;++i)
      fprintf(filePtr,"%e\n",all_on_zero_coeffs->local_value(i));
    fclose(filePtr);
  } // end work done on proc 0
  // broadcast the coeffs back to all
  MultiField_Exporter coeff_exporter(*all_on_all_coeff_map,*all_on_zero_coeffs->get_map());
  // export the field to zero
  all_coeffs->do_import(all_on_zero_coeffs,coeff_exporter,INSERT);
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R11: " << all_coeffs->local_value(0));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R12: " << all_coeffs->local_value(1));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R13: " << all_coeffs->local_value(2));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R21: " << all_coeffs->local_value(3));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R22: " << all_coeffs->local_value(4));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R23: " << all_coeffs->local_value(5));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R31: " << all_coeffs->local_value(6));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R32: " << all_coeffs->local_value(7));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R33: " << all_coeffs->local_value(8));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff tx: " << all_coeffs->local_value(9));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff ty: " << all_coeffs->local_value(10));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff tz: " << all_coeffs->local_value(11));

  // convert the existig tranformation matrix to the new one
  std::vector<std::vector<scalar_t> > temp(4,std::vector<scalar_t>(4,0.0));
  for(int_t i=0;i<4;++i){
    for(int_t j=0;j<4;++j){
      temp[i][j] = trans_extrinsics_[i][j];
      trans_extrinsics_[i][j] = 0.0;
    }
  }
  std::vector<std::vector<scalar_t> > fit(4,std::vector<scalar_t>(4,0.0));
  fit[0][0] = all_coeffs->local_value(0);
  fit[0][1] = all_coeffs->local_value(1);
  fit[0][2] = all_coeffs->local_value(2);
  fit[0][3] = all_coeffs->local_value(9);
  fit[1][0] = all_coeffs->local_value(3);
  fit[1][1] = all_coeffs->local_value(4);
  fit[1][2] = all_coeffs->local_value(5);
  fit[1][3] = all_coeffs->local_value(10);
  fit[2][0] = all_coeffs->local_value(6);
  fit[2][1] = all_coeffs->local_value(7);
  fit[2][2] = all_coeffs->local_value(8);
  fit[2][3] = all_coeffs->local_value(11);
  fit[3][3] = 1.0;

  for(int_t i=0;i<4;++i){
    for(int_t j=0;j<4;++j){
      for(int_t k=0;k<4;++k){
        trans_extrinsics_[i][k] += temp[i][j] * fit[j][k];
      }
    }
  }
}

scalar_t
Triangulation::cosine_of_two_vectors(const std::vector<scalar_t> & a,
  const std::vector<scalar_t> & b){
  assert(a.size()==3);
  assert(b.size()==3);

  scalar_t mag_a = 0.0;
  for(int_t i=0;i<3;++i)
    mag_a += a[i]*a[i];
  mag_a = std::sqrt(mag_a);
  assert(mag_a>0.0);
  scalar_t mag_b = 0.0;
  for(int_t i=0;i<3;++i)
    mag_b += b[i]*b[i];
  mag_b = std::sqrt(mag_b);
  assert(mag_b>0.0);
  scalar_t result = 0.0;
  for(int_t i=0;i<3;++i)
    result += a[i]*b[i];
  result /= (mag_a*mag_b);
  return result;
}


void Triangulation::triangulate(const scalar_t & x0,
  const scalar_t & y0,
  const scalar_t & x1,
  const scalar_t & y1,
  scalar_t & xc_out,
  scalar_t & yc_out,
  scalar_t & zc_out,
  scalar_t & xw_out,
  scalar_t & yw_out,
  scalar_t & zw_out,
  const bool correct_lens_distortion){
  DEBUG_MSG("Triangulation::triangulate(): camera 0 sensor coords " << x0 << " " << y0 << " camera 1 sensor coords " << x1 << " " << y1);
  static scalar_t xc0 = 0.0;
  static scalar_t yc0 = 0.0;
  static scalar_t xc1 = 0.0;
  static scalar_t yc1 = 0.0;
  xc0 = x0;
  yc0 = y0;
  xc1 = x1;
  yc1 = y1;
  if(correct_lens_distortion){
    correct_lens_distortion_radial(xc0,yc0,0);
    correct_lens_distortion_radial(xc1,yc1,1);
    DEBUG_MSG("Triangulation::triangulate(): distortion corrected camera 0 sensor coords " << xc0 << " " << yc0 << " camera 1 sensor coords " << xc1 << " " << yc1);
  }

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
  M(0,2) = cal_intrinsics_[0][0] - xc0; // cx1 - xs1
  M(1,1) = cal_intrinsics_[0][3]; // fy1
  M(1,2) = cal_intrinsics_[0][1] - yc0; // cy1 - ys1
  cmx = cal_intrinsics_[1][0] - xc1; // cx2 - xs2
  cmy = cal_intrinsics_[1][1] - yc1; // cy2 - ys2
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
  xc_out = XYZc0[0];
  yc_out = XYZc0[1];
  zc_out = XYZc0[2];
  DEBUG_MSG("Triangulation::triangulate(): camera 0 coordinates X " << XYZc0[0] << " Y " << XYZc0[1] << " Z "  << XYZc0[2]);

  // apply the camera 0 to world coord transform
  for(int_t i=0;i<4;++i){
    for(int_t j=0;j<4;++j){
      XYZ[i] += trans_extrinsics_[i][j]*XYZc0[j];
    }
  }
  xw_out = XYZ[0];
  yw_out = XYZ[1];
  zw_out = XYZ[2];
  DEBUG_MSG("Triangulation::triangulate(): world coordinates X " << xw_out << " Y " << yw_out << " Z "  << zw_out);
}

void
Triangulation::correct_lens_distortion_radial(scalar_t & x_s,
  scalar_t & y_s,
  const int_t camera_id){
  assert(cal_intrinsics_.size()>0);
  static scalar_t rho_tilde = 0.0; // = rho^2
  static scalar_t r1 = 0.0;
  static scalar_t r2 = 0.0;
  static scalar_t factor = 0.0;
  r1 = (x_s-cal_intrinsics_[camera_id][0])/cal_intrinsics_[camera_id][0]; // tested above to see that cx > 0 and cy > 0 when cal parameters loaded
  r2 = (y_s-cal_intrinsics_[camera_id][1])/cal_intrinsics_[camera_id][1];
  rho_tilde = r1*r1 + r2*r2;
  factor = (cal_intrinsics_[camera_id][5]*rho_tilde + cal_intrinsics_[camera_id][6]*rho_tilde*rho_tilde
      + cal_intrinsics_[camera_id][7]*rho_tilde*rho_tilde*rho_tilde);
  //DEBUG_MSG("Triangulation::correct_lens_distortion(): corrections x " << factor*r1*cal_intrinsics_[camera_id][0] << " y " << factor*r2*cal_intrinsics_[camera_id][1]);
  x_s = x_s - factor*r1*cal_intrinsics_[camera_id][0];
  y_s = y_s - factor*r2*cal_intrinsics_[camera_id][1];
}

void
Triangulation::project_camera_0_to_sensor_1(const scalar_t & xc,
  const scalar_t & yc,
  const scalar_t & zc,
  scalar_t & xs2_out,
  scalar_t & ys2_out){

  Teuchos::SerialDenseMatrix<int_t,double> F2(3,4,true);
  F2(0,0) = cal_intrinsics_[1][2];
  F2(0,1) = cal_intrinsics_[1][4];
  F2(0,2) = cal_intrinsics_[1][0];
  F2(1,1) = cal_intrinsics_[1][3];
  F2(1,2) = cal_intrinsics_[1][1];
  F2(2,2) = 1.0;

  Teuchos::SerialDenseMatrix<int_t,double> F2_T(3,4,true);
  for(int_t j=0;j<3;++j){
    for(int_t k=0;k<4;++k){
      for(int_t i=0;i<4;++i){
        F2_T(j,k) += F2(j,i)*cal_extrinsics_[i][k];
      }
    }
  }
//  std::cout << " F2 " << std::endl;
//  for(int_t j=0;j<3;++j){
//    for(int_t k=0;k<4;++k){
//      std::cout << F2(j,k) << " ";
//    }
//    std::cout << std::endl;
//  }
//  std::cout << " T " << std::endl;
//  for(int_t j=0;j<4;++j){
//    for(int_t k=0;k<4;++k){
//      std::cout << cal_extrinsics_[j][k] << " ";
//    }
//    std::cout << std::endl;
//  }
//  std::cout << " F2T " << std::endl;
//  for(int_t j=0;j<3;++j){
//    for(int_t k=0;k<4;++k){
//      std::cout << F2_T(j,k) << " ";
//    }
//    std::cout << std::endl;
//  }
  const scalar_t psi2 = cal_extrinsics_[2][0]*xc + cal_extrinsics_[2][1]*yc + cal_extrinsics_[2][2]*zc + cal_extrinsics_[2][3];
  assert(psi2!=0.0);
  xs2_out = 1.0/psi2*(F2_T(0,0)*xc + F2_T(0,1)*yc + F2_T(0,2)*zc + F2_T(0,3));
  ys2_out = 1.0/psi2*(F2_T(1,0)*xc + F2_T(1,1)*yc + F2_T(1,2)*zc + F2_T(1,3));
  scalar_t z2 = 1.0/psi2*(F2_T(2,0)*xc + F2_T(2,1)*yc + F2_T(2,2)*zc + F2_T(2,3));
  //std::cout << " xs2_out " << xs2_out << " ys2out " << ys2_out << std::endl;
  //std::cout << "z2 " << z2 << std::endl;
  assert(std::abs(z2-1.0) < 0.1);

}

/// estimate the projective transform from the left to right image
void
Triangulation::estimate_projective_transform(Teuchos::RCP<Image> left_img,
  Teuchos::RCP<Image> right_img,
  const bool output_projected_image){

  // read the projection points from projection_points.dat
  std::fstream projDataFile("projection_points.dat", std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!projDataFile.good(),std::runtime_error,
    "Error, could not open file projection_points.dat (required for cross-correlation)");
  int_t num_coords = 0;
  while(!projDataFile.eof()){
    Teuchos::ArrayRCP<std::string> tokens = tokenize_line(projDataFile);
    if(tokens.size()==0) continue;
    TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=4,std::runtime_error,
      "Error reading projection_points.dat, should be 4 values per line (x_left y_left x_righ y_right),"
        " but found " << tokens.size() << " values on line " << num_coords+1);
    num_coords++;
  }
  DEBUG_MSG("Triangulation::estimate_projective_transform(): found projection_points.dat file with " << num_coords << " points");
  TEUCHOS_TEST_FOR_EXCEPTION(num_coords<4,std::runtime_error,
    "Error, not enough sets of coordinates in projection_points.dat to estimate projection (needs at least 4)");
  std::vector<scalar_t> proj_xl(num_coords,0.0);
  std::vector<scalar_t> proj_yl(num_coords,0.0);
  std::vector<scalar_t> proj_xr(num_coords,0.0);
  std::vector<scalar_t> proj_yr(num_coords,0.0);
  projDataFile.clear();
  projDataFile.seekg(0, std::ios::beg);
  int_t coord_index = 0;
  while(!projDataFile.eof()){
    Teuchos::ArrayRCP<std::string> tokens = tokenize_line(projDataFile);
    if(tokens.size()==0) continue;
    assert(tokens.size()==4);
    proj_xl[coord_index] = strtod(tokens[0].c_str(),NULL);
    proj_yl[coord_index] = strtod(tokens[1].c_str(),NULL);
    proj_xr[coord_index] = strtod(tokens[2].c_str(),NULL);
    proj_yr[coord_index] = strtod(tokens[3].c_str(),NULL);
    DEBUG_MSG("Triangulation::estimate_projective_transform(): xl " <<
      proj_xl[coord_index] << " yl " << proj_yl[coord_index] << " xr " << proj_xr[coord_index] << " yr " << proj_yr[coord_index]);
    coord_index++;
  }
  projDataFile.close();

  // use a least squares fit to estimate the parameters
  int N = 8;
  Teuchos::SerialDenseMatrix<int_t,double> K(num_coords*2,N,true);
  Teuchos::SerialDenseMatrix<int_t,double> KTK(N,N,true);
  Teuchos::ArrayRCP<scalar_t> F(num_coords*2,0.0);
  Teuchos::ArrayRCP<scalar_t> KTu(N,0.0);
  for(int_t i=0;i<num_coords;++i){
    K(i*2+0,0) = proj_xl[i];
    K(i*2+0,1) = proj_yl[i];
    K(i*2+0,2) = 1.0;
    K(i*2+0,6) = -1.0*proj_xl[i]*proj_xr[i];
    K(i*2+0,7) = -1.0*proj_yl[i]*proj_xr[i];
    K(i*2+1,3) = proj_xl[i];
    K(i*2+1,4) = proj_yl[i];
    K(i*2+1,5) = 1.0;
    K(i*2+1,6) = -1.0*proj_xl[i]*proj_yr[i];
    K(i*2+1,7) = -1.0*proj_yl[i]*proj_yr[i];
    F[i*2+0] = proj_xr[i];
    F[i*2+1] = proj_yr[i];
  }
  // set up K^T*K
  for(int_t k=0;k<N;++k){
    for(int_t m=0;m<N;++m){
      for(int_t j=0;j<num_coords*2;++j){
        KTK(k,m) += K(j,k)*K(j,m);
      }
    }
  }
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  // Note, LAPACK does not allow templating on long int or scalar_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;
  // invert the KTK matrix
  lapack.GETRF(KTK.numRows(),KTK.numCols(),KTK.values(),KTK.numRows(),IPIV,&INFO);
  lapack.GETRI(KTK.numRows(),KTK.values(),KTK.numRows(),IPIV,WORK,LWORK,&INFO);
  // compute K^T*F
  for(int_t i=0;i<N;++i){
    for(int_t j=0;j<num_coords*2;++j){
      KTu[i] += K(j,i)*F[j];
    }
  }
  for(size_t i=0;i<projectives_->size();++i)
    (*projectives_)[i] = 0.0;
  // compute the coeffs
  for(int_t i=0;i<N;++i){
    for(int_t j=0;j<N;++j){
      (*projectives_)[i] += KTK(i,j)*KTu[j];
    }
  }
  delete [] WORK;
  delete [] IPIV;

  int_t num_iterations = 0;
  // create an output file with the initial solution and final solution for projection params
  std::FILE * filePtr = fopen("projection_out.dat","w");
  fprintf(filePtr,"Projection parameters from point matching: \n");
  for(size_t i=0;i<projectives_->size();++i){
    fprintf(filePtr,"%e\n",(*projectives_)[i]);
  }
  fclose(filePtr);

  // simplex optimize the coefficients
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::max_iterations,200);
  params->set(DICe::tolerance,0.00001);
  DICe::Homography_Simplex simplex(left_img,right_img,this,params);
  Teuchos::RCP<std::vector<scalar_t> > deltas = Teuchos::rcp(new std::vector<scalar_t>(8,0.0));
  (*deltas)[0] = 0.001;
  (*deltas)[1] = 0.001;
  (*deltas)[2] = 1.0;
  (*deltas)[3] = 0.001;
  (*deltas)[4] = 0.001;
  (*deltas)[5] = 1.0;
  (*deltas)[6] = 0.0001;
  (*deltas)[7] = 0.0001;
  Status_Flag corr_status = simplex.minimize(projectives_,deltas,num_iterations);
  TEUCHOS_TEST_FOR_EXCEPTION(corr_status!=CORRELATION_SUCCESSFUL,std::runtime_error,"Error, could not determine projective transform.");

  filePtr = fopen("projection_out.dat","a");
  fprintf(filePtr,"Projection parameters after simplex optimization: \n");
  for(size_t i=0;i<projectives_->size();++i){
    fprintf(filePtr,"%e\n",(*projectives_)[i]);
  }
  fprintf(filePtr,"Optimization took %i iterations\n",num_iterations);
  fclose(filePtr);

  if(output_projected_image){
    const int_t w = left_img->width();
    const int_t h = left_img->height();
    Teuchos::RCP<Image> img = Teuchos::rcp(new Image(w,h,0.0));
    Teuchos::ArrayRCP<intensity_t> intens = img->intensities();
    Teuchos::RCP<Image> diff_img = Teuchos::rcp(new Image(w,h,0.0));
    Teuchos::ArrayRCP<intensity_t> diff_intens = diff_img->intensities();
    scalar_t xr = 0.0;
    scalar_t yr = 0.0;
    for(int_t j=0.1*h;j<0.9*h;++j){
      for(int_t i=0.1*w;i<0.9*w;++i){
        project_left_to_right_sensor_coords(i,j,xr,yr);
        diff_intens[j*w+i] = (*left_img)(i,j) - right_img->interpolate_keys_fourth(xr,yr);
        intens[j*w+i] = right_img->interpolate_keys_fourth(xr,yr);
      }
    }
    diff_img->write("projection_diff.tif");
    img->write("right_projected_to_left.tif");
  }
}

void
Triangulation::project_left_to_right_sensor_coords(const scalar_t & xl,
  const scalar_t & yl,
  scalar_t & xr,
  scalar_t & yr){
  // correct for lens distortion
  //scalar_t xlc = xl;
  //scalar_t ylc = yl;
  //correct_lens_distortion_radial(xlc,ylc,0);
  assert(projectives_!=Teuchos::null);
  assert(projectives_->size()==8);
  std::vector<scalar_t> & pr = *projectives_;
  scalar_t denom = pr[6]*xl + pr[7]*yl + 1.0;
  xr = (pr[0]*xl+pr[1]*yl+pr[2])/denom;
  yr = (pr[3]*xl+pr[4]*yl+pr[5])/denom;
  //correct_lens_distortion_radial(xr,yr,1);
}

}// End DICe Namespace
