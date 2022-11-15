// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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
#include <DICe_Feature.h>
#include <DICe_Simplex.h>
#include <DICe_Parser.h>
#include <DICe_CameraSystem.h>
#include <DICe_ImageIO.h>

#include <Teuchos_LAPACK.hpp>
#include <fstream>

namespace DICe {

DICE_LIB_DLL_EXPORT
Teuchos::SerialDenseMatrix<int_t,double>
compute_affine_matrix(const std::vector<scalar_t> proj_xl,
  const std::vector<scalar_t> proj_yl,
  const std::vector<scalar_t> proj_xr,
  const std::vector<scalar_t> proj_yr){
  DEBUG_MSG("compute_affine_matrix(): begin");
  Teuchos::SerialDenseMatrix<int_t,double> affine_matrix(3,3,true);
  const size_t num_coords = proj_xl.size();
  assert(num_coords >= 4);
  assert(proj_yl.size()==num_coords);
  assert(proj_xr.size()==num_coords);
  assert(proj_yr.size()==num_coords);
  // normalize the points by centering on 0,0 and scaling so that the average distance from center is sqrt(2):
  DEBUG_MSG("compute_affine_matrix(): normalizing feature points");
  // compute the centroids
  scalar_t cl_x = 0.0;
  scalar_t cl_y = 0.0;
  scalar_t cr_x = 0.0;
  scalar_t cr_y = 0.0;
  for(size_t i=0;i<num_coords;++i){
    cl_x += proj_xl[i];
    cl_y += proj_yl[i];
    cr_x += proj_xr[i];
    cr_y += proj_yr[i];
  }

  cl_x /= num_coords;
  cl_y /= num_coords;
  cr_x /= num_coords;
  cr_y /= num_coords;
  // compute the average distances:
  scalar_t dl_x = 0.0;
  scalar_t dl_y = 0.0;
  scalar_t dr_x = 0.0;
  scalar_t dr_y = 0.0;
  for(size_t i=0;i<num_coords;++i){
    dl_x += std::abs(proj_xl[i] - cl_x);
    dl_y += std::abs(proj_yl[i] - cl_y);
    dr_x += std::abs(proj_xr[i] - cr_x);
    dr_y += std::abs(proj_yr[i] - cr_y);
  }
  dl_x /= num_coords;
  dl_y /= num_coords;
  dr_x /= num_coords;
  dr_y /= num_coords;

  assert(dl_x != 0.0);
  assert(dl_y != 0.0);
  assert(dr_x != 0.0);
  assert(dr_y != 0.0);
  scalar_t sl_x = 1.0 / dl_x;
  scalar_t sl_y = 1.0 / dl_y;
  scalar_t sr_x = 1.0 / dr_x;
  scalar_t sr_y = 1.0 / dr_y;

  // compute the similarity transform
  Teuchos::SerialDenseMatrix<int_t,double> Tl(3,3,true);
  Tl(0,0) = sl_x;
  Tl(0,2) = -sl_x*cl_x;
  Tl(1,1) = sl_y;
  Tl(1,2) = -sl_y*cl_y;
  Tl(2,2) = 1.0;
  Teuchos::SerialDenseMatrix<int_t,double> Tr(3,3,true);
  Tr(0,0) = sr_x;
  Tr(0,2) = -sr_x*cr_x;
  Tr(1,1) = sr_y;
  Tr(1,2) = -sr_y*cr_y;
  Tr(2,2) = 1.0;

  // check the centroid of the new points and average distance:
  std::vector<scalar_t> mod_xl(num_coords,0.0);
  std::vector<scalar_t> mod_yl(num_coords,0.0);
  std::vector<scalar_t> mod_xr(num_coords,0.0);
  std::vector<scalar_t> mod_yr(num_coords,0.0);

  for(size_t i=0;i<num_coords;++i){
    mod_xl[i] = Tl(0,0)*proj_xl[i] + Tl(0,1)*proj_yl[i] + Tl(0,2);
    mod_yl[i] = Tl(1,0)*proj_xl[i] + Tl(1,1)*proj_yl[i] + Tl(1,2);
    mod_xr[i] = Tr(0,0)*proj_xr[i] + Tr(0,1)*proj_yr[i] + Tr(0,2);
    mod_yr[i] = Tr(1,0)*proj_xr[i] + Tr(1,1)*proj_yr[i] + Tr(1,2);
  }

  scalar_t mcl_x = 0.0;
  scalar_t mcl_y = 0.0;
  scalar_t mcr_x = 0.0;
  scalar_t mcr_y = 0.0;
  for(size_t i=0;i<num_coords;++i){
    mcl_x += mod_xl[i];
    mcl_y += mod_yl[i];
    mcr_x += mod_xr[i];
    mcr_y += mod_yr[i];
  }
  mcl_x /= num_coords;
  mcl_y /= num_coords;
  mcr_x /= num_coords;
  mcr_y /= num_coords;
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(mcl_x)>1.0E-3,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(mcl_y)>1.0E-3,std::runtime_error,"");
  // compute the average distances:
  scalar_t mdl_x = 0.0;
  scalar_t mdl_y = 0.0;
  scalar_t mdr_x = 0.0;
  scalar_t mdr_y = 0.0;
  for(size_t i=0;i<num_coords;++i){
    mdl_x += std::abs(mod_xl[i] - mcl_x);
    mdl_y += std::abs(mod_yl[i] - mcl_y);
    mdr_x += std::abs(mod_xr[i] - mcr_x);
    mdr_y += std::abs(mod_yr[i] - mcr_y);
  }
  mdl_x /= num_coords;
  mdl_y /= num_coords;
  mdr_x /= num_coords;
  mdr_y /= num_coords;
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(mdl_x-1.0)>1.0E-3,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(mdl_y-1.0)>1.0E-3,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(mdr_x-1.0)>1.0E-3,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(mdr_y-1.0)>1.0E-3,std::runtime_error,"");

  // compute the inverse of Tr
  int *IPIV = new int[4];
  int TIWORK = 9;
  int TINFO = 0;
  double *TWORK = new double[TIWORK];
  // Note, LAPACK does not allow templating on long int or scalar_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;
  DEBUG_MSG("compute_affine_matrix(): inverting for projection parameters");

  // invert the KTK matrix
  lapack.GETRF(Tr.numRows(),Tr.numCols(),Tr.values(),Tr.numRows(),IPIV,&TINFO);
  lapack.GETRI(Tr.numRows(),Tr.values(),Tr.numRows(),IPIV,TWORK,TIWORK,&TINFO);
  // now Tr is inverted

  int N = 9;
  Teuchos::SerialDenseMatrix<int_t,double> K(num_coords*2,N,true);
  Teuchos::SerialDenseMatrix<int_t,double> KTK(N,N,true);
  Teuchos::ArrayRCP<scalar_t> u(N,0.0);
  for(size_t i=0;i<num_coords;++i){
    K(i*2+0,0) = -mod_xl[i];
    K(i*2+0,1) = -mod_yl[i];
    K(i*2+0,2) = -1.0;
    K(i*2+0,6) = mod_xr[i]*mod_xl[i];
    K(i*2+0,7) = mod_xr[i]*mod_yl[i];
    K(i*2+0,8) = mod_xr[i];
    K(i*2+1,3) = -mod_xl[i];
    K(i*2+1,4) = -mod_yl[i];
    K(i*2+1,5) = -1.0;
    K(i*2+1,6) = mod_yr[i]*mod_xl[i];
    K(i*2+1,7) = mod_yr[i]*mod_yl[i];
    K(i*2+1,8) = mod_yr[i];
  }
  // set up K^T*K
  for(int_t k=0;k<N;++k){
    for(int_t m=0;m<N;++m){
      for(size_t j=0;j<num_coords*2;++j){
        KTK(k,m) += K(j,k)*K(j,m);
      }
    }
  }
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  for(int_t j=0;j<9;++j){
    for(int_t i=0;i<j;++i){
      KTK(j,i) = 0.0;
    }
  }
  double *EIGS = new double[N];
  //Teuchos::LAPACK<int,double> lapack;
  lapack.SYEV('V','U',N,KTK.values(),N,EIGS,WORK,LWORK,&INFO);

  DEBUG_MSG("compute_affine_matrix(): Smallest eigenvalue: " << EIGS[0] );
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(EIGS[0]) > 10.0,std::runtime_error,
    "Error, too much noise in the projection estimation points");

  for(int_t i=0;i<N;++i){
    DEBUG_MSG("compute_affine_matrix(): Eigenvector: " << KTK(i,0) );
  }

  // convert the H back to the original coordinate system (post similarity transforms)
  Teuchos::SerialDenseMatrix<int_t,double> Htilde(3,3,true);
  Htilde(0,0) = KTK(0,0);
  Htilde(0,1) = KTK(1,0);
  Htilde(0,2) = KTK(2,0);
  Htilde(1,0) = KTK(3,0);
  Htilde(1,1) = KTK(4,0);
  Htilde(1,2) = KTK(5,0);
  Htilde(2,0) = KTK(6,0);
  Htilde(2,1) = KTK(7,0);
  Htilde(2,2) = KTK(8,0);
  // compute Tr^-1 Htilde Tl
  Teuchos::SerialDenseMatrix<int_t,double> HtildeTl(3,3,true);
  for(int_t i=0;i<3;++i){
    for(int_t j=0;j<3;++j){
      for(int_t k=0;k<3;++k){
        HtildeTl(i,j) += Htilde(i,k)*Tl(k,j);
      }
    }
  }
//  Teuchos::SerialDenseMatrix<int_t,double> proj_matrix(3,3,true);
  for(int_t i=0;i<3;++i){
    for(int_t j=0;j<3;++j){
      for(int_t k=0;k<3;++k){
        affine_matrix(i,j) += Tr(i,k)*HtildeTl(k,j);
      }
    }
  }
  delete [] WORK;
  delete [] EIGS;
  delete [] TWORK;
  delete [] IPIV;
  DEBUG_MSG("compute_affine_matrix(): end");
  return affine_matrix;
}

void
Triangulation::load_calibration_parameters(const std::string & param_file_name){
  DEBUG_MSG("Triangulation::load_calibration_parameters(): Parsing calibration parameters from file: " << param_file_name);

  camera_system_ = Teuchos::rcp(new Camera_System(param_file_name));

  // set the camera intrinsic parameters
  for(size_t i=0;i<camera_system_->num_cameras();++i)
    for(size_t j=0;j<Camera::MAX_CAM_INTRINSIC_PARAM;++j)
    cal_intrinsics_[i][j] = (*camera_system_->camera(i)->intrinsics())[j];

  if(camera_system_->num_cameras()==1) return;

  TEUCHOS_TEST_FOR_EXCEPTION(camera_system_->num_cameras()!=2,std::runtime_error,"");

  // The standard convention for cal extrinsics in DICeis to have two transforms, one transform from camera 0's coordinate
  // system to the world or model coordinate system, the second from camera 0 to camera 1. Depending on which type of cal file
  // is input, this may not be the convention. For example with OpenCV, the first transform is from world to camera 0 so this
  // transform needs to be inverted. With VIC3D the first transform is from world to camera 0 and the second is from world to camera 1.
  // In the VIC3D case, the first transform again needs to be inverted and the second (cam 0 to cam 1) becomes the product of the first
  // inverted transform and the second transform (which combines camera 0 to world then world to camera 1 to result in cam 0 to cam 1)

  // If the extrinsics_relative_camera_to_camera flag is set, the transformation matrices provided by the input
  // parameters (the rotation matrices and t-veces for each camera) are already set up to match the triangulation convention
  // so the only conversion necessary is to invert the first transform.

  cam_0_to_world_ = camera_system_->camera(0)->transformation_matrix();
  cam_0_to_world_ = cam_0_to_world_.inv(); // assume camera 0's transformation matrix is always provided as world/model to camera 0 so it needs to be inverted
  cam_0_to_cam_1_ = camera_system_->camera(1)->transformation_matrix();

  // if the camera extrinsics were given as world to camera 0 and world to camera 1, (now with the first inverted above to be camera 0 to world)
  // the second transform needs to be converted to combine the camera 0 to world and world to camera 1 transform
  if(!camera_system_->extrinsics_relative_camera_to_camera()){
    cam_0_to_cam_1_ = cam_0_to_cam_1_ * cam_0_to_world_;
  }

#ifdef DICE_DEBUG_MSG
  std::cout << *camera_system_.get() << std::endl;
#endif

  DEBUG_MSG("Triangulation::load_calibration_parameters(): successfully loaded camera system");
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
    all_entries->local_value(6) -= 1.0*x*z;
    all_entries->local_value(7) -= 1.0*y*z;
    all_entries->local_value(8) -= 1.0*z;
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
      std::cout << e.what() << '\n';
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
        "Error, could not invert the K matrix in best fit plane");
    }
    delete [] IPIV;
    delete [] WORK;
    // determine the coefficients
    // -z = u[0]*x + u[1]*y + u[3], equation of a plane
    for(int_t i=0;i<3;++i){
      for(int_t j=0;j<3;++j){
        u[i] += K(i,j)*F[j];
      }
    }
    for(int_t i=0;i<3;++i){
      DEBUG_MSG("[proc " << comm.get_rank() << "] best_fit_plane coeff " << i << " " << u[i]);
    }
    std::FILE * filePtr = fopen("best_fit_plane_out.dat","w");
    fprintf(filePtr,"# Best fit plane equation coefficients (-z = c[0]*x + c[1]*y + c[2]): \n");
    fprintf(filePtr,"%e\n",u[0]);
    fprintf(filePtr,"%e\n",u[1]);
    fprintf(filePtr,"%e\n",u[2]);

    bool is_y_axis = false;

    // read in origin in image left coordinates
    std::vector<int_t> fit_def_x_left(2,0);
    std::vector<int_t> fit_def_y_left(2,0);
    //std::vector<scalar_t> fit_def_x_right(2,0.0);
    //std::vector<scalar_t> fit_def_y_right(2,0.0);
    std::fstream bestFitDataFile("best_fit_plane.dat", std::ios_base::in);
    TEUCHOS_TEST_FOR_EXCEPTION(!bestFitDataFile.good(),std::runtime_error,
      "Error, could not open file best_fit_plane.dat (required to project output to best fit plane)");
    int_t line = 0;
    while(!bestFitDataFile.eof()){
      std::vector<std::string> tokens = tokenize_line(bestFitDataFile);
      if(tokens.size()==0) continue;
      if(tokens.size()>2){
        if(tokens[2] == "YAXIS"){
          is_y_axis = true;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=2&&!is_y_axis,std::runtime_error,
        "Error reading best_fit_plane.dat, should be 2 values per line (x_left y_left for origin and point on x axis),"
          " but found " << tokens.size() << " values on one line");
      assert(line<(int_t)fit_def_x_left.size());
      fit_def_x_left[line] = strtol(tokens[0].c_str(),NULL,0);
      fit_def_y_left[line] = strtol(tokens[1].c_str(),NULL,0);
      line++;
    }
    DEBUG_MSG("Best fit plane origin (left sensor coords):           " << fit_def_x_left[0] << " " << fit_def_y_left[0]);
    if(is_y_axis){
      DEBUG_MSG("Best fit plane point on y axis (left sensor coords):  " << fit_def_x_left[1] << " " << fit_def_y_left[1]);
    }else{
      DEBUG_MSG("Best fit plane point on x axis (left sensor coords):  " << fit_def_x_left[1] << " " << fit_def_y_left[1]);
    }
    // determine the corresponding point in the camera 0 coordinates, given the image coordinates of the origin:
    // this is done by using the psi[u,v,1] = [F]*[X,Y,Z] formula with Z = -1(u0*X+u1*Y+u2) to solve for X and Y
    const scalar_t cx = cal_intrinsics_[0][Camera::CX];
    const scalar_t cy = cal_intrinsics_[0][Camera::CY];
    const scalar_t fx = cal_intrinsics_[0][Camera::FX];
    const scalar_t fy = cal_intrinsics_[0][Camera::FY];
    const scalar_t fs = cal_intrinsics_[0][Camera::FS];
    scalar_t U = fit_def_x_left[0];
    scalar_t V = fit_def_y_left[0];
    const scalar_t c0 = u[0];
    const scalar_t c1 = u[1];
    const scalar_t c2 = u[2];
    const scalar_t XO = -(c2*cy*fs - c2*cx*fy + c2*fy*U - c2*fs*V)/(fx*fy + c0*cy*fs - c0*cx*fy - c1*cy*fx + c0*fy*U - c0*fs*V + c1*fx*V);
    const scalar_t YO = (c2*cy*fx - c2*fx*V)/(fx*fy + c0*cy*fs - c0*cx*fy - c1*cy*fx + c0*fy*U - c0*fs*V + c1*fx*V);
    const scalar_t ZO = -1.0*(c0*XO + c1*YO + c2);
    U = fit_def_x_left[1];
    V = fit_def_y_left[1];
    const scalar_t XP = -(c2*cy*fs - c2*cx*fy + c2*fy*U - c2*fs*V)/(fx*fy + c0*cy*fs - c0*cx*fy - c1*cy*fx + c0*fy*U - c0*fs*V + c1*fx*V);
    const scalar_t YP = (c2*cy*fx - c2*fx*V)/(fx*fy + c0*cy*fs - c0*cx*fy - c1*cy*fx + c0*fy*U - c0*fs*V + c1*fx*V);
    const scalar_t ZP = -1.0*(c0*XP + c1*YP + c2);

    fprintf(filePtr,"# Best fit origin in camera 0 coordinates \n");
    fprintf(filePtr,"%e\n",XO);
    fprintf(filePtr,"%e\n",YO);
    fprintf(filePtr,"%e\n",ZO);
    fprintf(filePtr,"# Best fit point along x axis in camera 0 coordinates \n");
    fprintf(filePtr,"%e\n",XP);
    fprintf(filePtr,"%e\n",YP);
    fprintf(filePtr,"%e\n",ZP);

    // the world coordinates are obtained from the standard basis vectors e1 = 1 0 0, e2 = 0 1 0, e3 = 0 0 1
    std::vector<scalar_t> e1(3,0.0);
    e1[0] = 1.0;
    std::vector<scalar_t> e2(3,0.0);
    e2[1] = 1.0;
    std::vector<scalar_t> e3(3,0.0);
    e3[2] = 1.0;

    // the best fit plane coordinates are determined by the basis vectors g1 g2 g3
    std::vector<scalar_t> g1(3,0.0);
    std::vector<scalar_t> g2(3,0.0);
    // g3 is the normal vector on the plane
    std::vector<scalar_t> g3(3,0.0);
    g3[0] = -u[0];
    g3[1] = -u[1];
    g3[2] = -1.0;

    if(is_y_axis){
      // g2 is from the origin to the point on the y-axis
      g2[0] = XP - XO;//xaXw - oXw;
      g2[1] = YP - YO;//xaYw - oYw;
      g2[2] = ZP - ZO;//xaZw - oZw;
      // g2 is obtained by the cross product of g3 with g1
      g1[0] = g2[1]*g3[2] - g3[1]*g2[2];
      g1[1] = g2[2]*g3[0] - g3[2]*g2[0];
      g1[2] = g2[0]*g3[1] - g3[0]*g2[1];
    }else{
      // g1 is from the origin to the point on the x-axis
      g1[0] = XP - XO;//xaXw - oXw;
      g1[1] = YP - YO;//xaYw - oYw;
      g1[2] = ZP - ZO;//xaZw - oZw;
      // g2 is obtained by the cross product of g3 with g1
      g2[0] = g3[1]*g1[2] - g1[1]*g3[2];
      g2[1] = g3[2]*g1[0] - g1[2]*g3[0];
      g2[2] = g3[0]*g1[1] - g1[0]*g3[1];
    }

    // need to invert the solution to get the transformation to the best fit plane
    Teuchos::SerialDenseMatrix<int_t,double> TK(4,4,true);
    TK(0,0) = cosine_of_two_vectors(e1,g1);
    TK(0,1) = cosine_of_two_vectors(e1,g2);
    TK(0,2) = cosine_of_two_vectors(e1,g3);
    TK(0,3) = XO;
    TK(1,0) = cosine_of_two_vectors(e2,g1);
    TK(1,1) = cosine_of_two_vectors(e2,g2);
    TK(1,2) = cosine_of_two_vectors(e2,g3);
    TK(1,3) = YO;
    TK(2,0) = cosine_of_two_vectors(e3,g1);
    TK(2,1) = cosine_of_two_vectors(e3,g2);
    TK(2,2) = cosine_of_two_vectors(e3,g3);
    TK(2,3) = ZO;
    TK(3,3) = 1.0;
//    std::cout << "Transformation Matrix: " << std::endl;
//    for(int_t i=0;i<4;++i){
//      for(int_t j=0;j<4;++j){
//        std::cout << TK(i,j) << " ";
//      }
//      std::cout << std::endl;
//    }

    // transformation has to be inverted to get trans_extrinsic matrix
    int *TIPIV = new int[5];
    int TLWORK = 16;
    int TINFO = 0;
    double *TWORK = new double[TLWORK];
    lapack.GETRF(4,4,TK.values(),4,TIPIV,&TINFO);
    for(int_t i=0;i<TLWORK;++i) TWORK[i] = 0.0;
    try
    {
      lapack.GETRI(4,TK.values(),4,TIPIV,TWORK,TLWORK,&TINFO);
    }
    catch(std::exception &e){
      std::cout << e.what() << '\n';
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,
        "Error, could not invert the TK matrix in best fit plane");
    }
    delete [] TIPIV;
    delete [] TWORK;

//    std::cout << "Inverse Transformation Matrix: " << std::endl;
//    for(int_t i=0;i<4;++i){
//      for(int_t j=0;j<4;++j){
//        std::cout << TK(i,j) << " ";
//      }
//      std::cout << std::endl;
//    }

    // check that the origin actually falls at (0,0,0) in the transformed system
    //scalar_t origin_error = 0.0;
    //for(int_t i=0;i<3;++i)
    //    origin_error += TK(i,0)*XO + TK(i,1)*YO + TK(i,2);
    //TEUCHOS_TEST_FOR_EXCEPTION(std::abs(origin_error) > 0.0, std::runtime_error,
    //  "Error, transformed origin is not at (0,0,0), origin: (" <<
    //  TK(0,0)*XO + TK(0,1)*YO + TK(0,2) << "," << TK(1,0)*XO + TK(1,1)*YO + TK(1,2) << "," << TK(2,0)*XO + TK(2,1)*YO + TK(2,2) <<
    //  "). The transformation cannot be correct");

    // the rotation matrix components are the cosines of the angles between the basis vectors
    all_on_zero_coeffs->local_value(0) = TK(0,0);
    all_on_zero_coeffs->local_value(1) = TK(0,1);
    all_on_zero_coeffs->local_value(2) = TK(0,2);
    all_on_zero_coeffs->local_value(3) = TK(0,3);
    all_on_zero_coeffs->local_value(4) = TK(1,0);
    all_on_zero_coeffs->local_value(5) = TK(1,1);
    all_on_zero_coeffs->local_value(6) = TK(1,2);
    all_on_zero_coeffs->local_value(7) = TK(1,3);
    all_on_zero_coeffs->local_value(8) = TK(2,0);
    all_on_zero_coeffs->local_value(9) = TK(2,1);
    all_on_zero_coeffs->local_value(10) = TK(2,2);
    all_on_zero_coeffs->local_value(11) = TK(2,3);
    fprintf(filePtr,"# Transform from current world coords to best fit plane origin and bases \n");
    fprintf(filePtr,"# R11 R12 R13 tx R21 R22 R23 ty R31 R32 R33 tz \n");
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
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff tx: "  << all_coeffs->local_value(3));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R21: " << all_coeffs->local_value(4));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R22: " << all_coeffs->local_value(5));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R23: " << all_coeffs->local_value(6));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff ty: "  << all_coeffs->local_value(7));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R31: " << all_coeffs->local_value(8));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R32: " << all_coeffs->local_value(9));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff R33: " << all_coeffs->local_value(10));
  DEBUG_MSG("[proc " << comm.get_rank() << "] best fit plane coeff tz: "  << all_coeffs->local_value(11));

  // set the new transformation matrix
  cam_0_to_world_(0,0) = all_coeffs->local_value(0);
  cam_0_to_world_(0,1) = all_coeffs->local_value(1);
  cam_0_to_world_(0,2) = all_coeffs->local_value(2);
  cam_0_to_world_(0,3) = all_coeffs->local_value(3);
  cam_0_to_world_(1,0) = all_coeffs->local_value(4);
  cam_0_to_world_(1,1) = all_coeffs->local_value(5);
  cam_0_to_world_(1,2) = all_coeffs->local_value(6);
  cam_0_to_world_(1,3) = all_coeffs->local_value(7);
  cam_0_to_world_(2,0) = all_coeffs->local_value(8);
  cam_0_to_world_(2,1) = all_coeffs->local_value(9);
  cam_0_to_world_(2,2) = all_coeffs->local_value(10);
  cam_0_to_world_(2,3) = all_coeffs->local_value(11);
  cam_0_to_world_(3,3) = 1.0;
}

void
Triangulation::undistort_points(std::vector<scalar_t> & points_x,
  std::vector<scalar_t> & points_y,
  const int_t camera_id)const{
  TEUCHOS_TEST_FOR_EXCEPTION(points_x.size()!=points_y.size()||points_x.size()==0,std::runtime_error,"");
  std::vector<scalar_t> sen_x(points_x.size(),0.0);
  std::vector<scalar_t> sen_y(points_x.size(),0.0);
  // convert the distorted image points to sensor coordinates
  camera_system_->camera(camera_id)->image_to_sensor(points_x,points_y,sen_x,sen_y);
  // convert the sensor coordiates back to undistorted image coordiates
  camera_system_->camera(camera_id)->sensor_to_image(sen_x,sen_y,points_x,points_y,DICe::Camera::NO_LENS_DISTORTION);
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


void
Triangulation::triangulate(const std::vector<scalar_t> & image_x,
  const std::vector<scalar_t> & image_y,
  std::vector<scalar_t> & world_x,
  std::vector<scalar_t> & world_y,
  std::vector<scalar_t> & world_z) const {

  TEUCHOS_TEST_FOR_EXCEPTION(camera_system_==Teuchos::null,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(camera_system_->num_cameras()!=1,std::runtime_error,"");
  DEBUG_MSG("Triangulation::triangulate(): executing projection from image coordinates to world coordinates using calibration parameters");

  // assume that the output vectors have been initialized
  TEUCHOS_TEST_FOR_EXCEPTION(image_x.size()!=image_y.size(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(image_x.size()!=world_x.size(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(image_x.size()!=world_y.size(),std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(image_x.size()!=world_z.size(),std::runtime_error,"");
  // in this case, the world coordinate system is aligned with the calibration plate or specimen surface
  camera_system_->camera(0)->image_to_world(image_x,image_y,world_x,world_y,world_z);
}

scalar_t Triangulation::triangulate(const scalar_t & x0,
  const scalar_t & y0,
  const scalar_t & x1,
  const scalar_t & y1,
  scalar_t & xc_out,
  scalar_t & yc_out,
  scalar_t & zc_out,
  scalar_t & xw_out,
  scalar_t & yw_out,
  scalar_t & zw_out,
  const bool correct_lens_distortion) const{
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
  M(0,0) = cal_intrinsics_[0][Camera::FX]; // fx0
  M(0,1) = cal_intrinsics_[0][Camera::FS]; // fs0
  M(0,2) = cal_intrinsics_[0][Camera::CX] - xc0; // cx1 - xs1
  M(1,1) = cal_intrinsics_[0][Camera::FY]; // fy1
  M(1,2) = cal_intrinsics_[0][Camera::CY] - yc0; // cy1 - ys1
  cmx = cal_intrinsics_[1][Camera::CX] - xc1; // cx2 - xs2
  cmy = cal_intrinsics_[1][Camera::CY] - yc1; // cy2 - ys2
  // (cx2-xs2)*R31 + fx2*R11 + fs2*R21
  M(2,0) = cmx*cam_0_to_cam_1_(2,0) + cal_intrinsics_[1][Camera::FX]*cam_0_to_cam_1_(0,0) + cal_intrinsics_[1][Camera::FS]*cam_0_to_cam_1_(1,0);
  // (cx2-xs2)*R32 + fx2*R12 + fs2*R22
  M(2,1) = cmx*cam_0_to_cam_1_(2,1) + cal_intrinsics_[1][Camera::FX]*cam_0_to_cam_1_(0,1) + cal_intrinsics_[1][Camera::FS]*cam_0_to_cam_1_(1,1);
  // (cx2-xs2)*R33 + fx2*R13 + fs2*R23
  M(2,2) = cmx*cam_0_to_cam_1_(2,2) + cal_intrinsics_[1][Camera::FX]*cam_0_to_cam_1_(0,2) + cal_intrinsics_[1][Camera::FS]*cam_0_to_cam_1_(1,2);
  // (cy2-ys2)*R31 + fy2*R21
  M(3,0) = cmy*cam_0_to_cam_1_(2,0) + cal_intrinsics_[1][Camera::FY]*cam_0_to_cam_1_(1,0);
  // (cy2-ys2)*R32 + fy2*R22
  M(3,1) = cmy*cam_0_to_cam_1_(2,1) + cal_intrinsics_[1][Camera::FY]*cam_0_to_cam_1_(1,1);
  // (cy2-ys2)*R33 + fy2*R23
  M(3,2) = cmy*cam_0_to_cam_1_(2,2) + cal_intrinsics_[1][Camera::FY]*cam_0_to_cam_1_(1,2);
  //-fx2tx - fs2ty -(cx2-xs2)*tz
  r[2] = -cal_intrinsics_[1][Camera::FX]*cam_0_to_cam_1_(0,3) - cal_intrinsics_[1][Camera::FS]*cam_0_to_cam_1_(1,3) - cmx*cam_0_to_cam_1_(2,3);
  //-fy2ty -(cy2-ys2)*tz
  r[3] = -cal_intrinsics_[1][Camera::FY]*cam_0_to_cam_1_(1,3) - cmy*cam_0_to_cam_1_(2,3);

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
    std::cout << e.what() << '\n';
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

  scalar_t max_m = std::abs(M(0,0));
  if(std::abs(M(1,1)) > max_m) max_m = std::abs(M(1,1));
  if(std::abs(M(2,2)) > max_m) max_m = std::abs(M(2,2));

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
      XYZ[i] += cam_0_to_world_(i,j)*XYZc0[j];
    }
  }
  xw_out = XYZ[0];
  yw_out = XYZ[1];
  zw_out = XYZ[2];
  DEBUG_MSG("Triangulation::triangulate(): world coordinates X " << xw_out << " Y " << yw_out << " Z "  << zw_out);
  return max_m;
}

void
Triangulation::correct_lens_distortion_radial(scalar_t & x_s,
  scalar_t & y_s,
  const int_t camera_id) const{
  assert(cal_intrinsics_.size()>0);
  static scalar_t rho_tilde = 0.0; // = rho^2
  static scalar_t r1 = 0.0;
  static scalar_t r2 = 0.0;
  static scalar_t factor = 0.0;
  r1 = (x_s-cal_intrinsics_[camera_id][Camera::CX])/cal_intrinsics_[camera_id][Camera::CX]; // tested above to see that cx > 0 and cy > 0 when cal parameters loaded
  r2 = (y_s-cal_intrinsics_[camera_id][Camera::CY])/cal_intrinsics_[camera_id][Camera::CY];
  rho_tilde = r1*r1 + r2*r2;
  factor = (cal_intrinsics_[camera_id][Camera::K1]*rho_tilde + cal_intrinsics_[camera_id][Camera::K2]*rho_tilde*rho_tilde
      + cal_intrinsics_[camera_id][Camera::K3]*rho_tilde*rho_tilde*rho_tilde);
  //DEBUG_MSG("Triangulation::correct_lens_distortion(): corrections x " << factor*r1*cal_intrinsics_[camera_id][0] << " y " << factor*r2*cal_intrinsics_[camera_id][1]);
  x_s = x_s - factor*r1*cal_intrinsics_[camera_id][Camera::CX];
  y_s = y_s - factor*r2*cal_intrinsics_[camera_id][Camera::CY];
}

//void
//Triangulation::project_camera_0_to_sensor_1(const scalar_t & xc,
//  const scalar_t & yc,
//  const scalar_t & zc,
//  scalar_t & xs2_out,
//  scalar_t & ys2_out){
//
//  Teuchos::SerialDenseMatrix<int_t,double> F2(3,4,true);
//  F2(0,0) = cal_intrinsics_[1][2];
//  F2(0,1) = cal_intrinsics_[1][4];
//  F2(0,2) = cal_intrinsics_[1][0];
//  F2(1,1) = cal_intrinsics_[1][3];
//  F2(1,2) = cal_intrinsics_[1][1];
//  F2(2,2) = 1.0;
//
//  Teuchos::SerialDenseMatrix<int_t,double> F2_T(3,4,true);
//  for(int_t j=0;j<3;++j){
//    for(int_t k=0;k<4;++k){
//      for(int_t i=0;i<4;++i){
//        F2_T(j,k) += F2(j,i)*cal_extrinsics_[i][k];
//      }
//    }
//  }
////  std::cout << " F2 " << std::endl;
////  for(int_t j=0;j<3;++j){
////    for(int_t k=0;k<4;++k){
////      std::cout << F2(j,k) << " ";
////    }
////    std::cout << std::endl;
////  }
////  std::cout << " T " << std::endl;
////  for(int_t j=0;j<4;++j){
////    for(int_t k=0;k<4;++k){
////      std::cout << cal_extrinsics_[j][k] << " ";
////    }
////    std::cout << std::endl;
////  }
////  std::cout << " F2T " << std::endl;
////  for(int_t j=0;j<3;++j){
////    for(int_t k=0;k<4;++k){
////      std::cout << F2_T(j,k) << " ";
////    }
////    std::cout << std::endl;
////  }
//  const scalar_t psi2 = cal_extrinsics_[2][0]*xc + cal_extrinsics_[2][1]*yc + cal_extrinsics_[2][2]*zc + cal_extrinsics_[2][3];
//  assert(psi2!=0.0);
//  xs2_out = 1.0/psi2*(F2_T(0,0)*xc + F2_T(0,1)*yc + F2_T(0,2)*zc + F2_T(0,3));
//  ys2_out = 1.0/psi2*(F2_T(1,0)*xc + F2_T(1,1)*yc + F2_T(1,2)*zc + F2_T(1,3));
//  scalar_t z2 = 1.0/psi2*(F2_T(2,0)*xc + F2_T(2,1)*yc + F2_T(2,2)*zc + F2_T(2,3));
//  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(z2-1.0) > 0.1,std::runtime_error,"");
//}

/// estimate the projective transform from the left to right image
int_t // returns 0 if successful -1 means linear projection failed, -2 means nonlinear projection failed
Triangulation::estimate_projective_transform(Teuchos::RCP<Image> left_img,
  Teuchos::RCP<Image> right_img,
  const bool output_projected_image,
  const bool use_nonlinear_projection,
  const int_t processor_id){

  scalar_t error = 0.0;

  // always estimate the projection without reading in a file even if the last run exists

//  // check if a file exists with the calculated projective parameters already:
//  std::fstream proj_params_file("projection_out.dat", std::ios_base::in);
//  if(proj_params_file.good()){
//    DEBUG_MSG("Triangulation::estimate_projective_transform(): found file: projection_out.dat, reading parameters from this file");
//    std::vector<scalar_t> values;
//    int_t line = 0;
//    while(!proj_params_file.eof()){
//      Teuchos::ArrayRCP<std::string> tokens = tokenize_line(proj_params_file);
//      line++;
//      if(tokens.size()==0)continue;
//      TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=1,std::runtime_error,
//        "Error reading projection_out.dat, should be 1 value per line (or a comment),"
//        " but found " << tokens.size() << " values on line " << line);
//      TEUCHOS_TEST_FOR_EXCEPTION(!is_number(tokens[0]),std::runtime_error,"Error, all values should be numeric");
//      values.push_back(std::strtod(tokens[0].c_str(),NULL));
//    }
//    TEUCHOS_TEST_FOR_EXCEPTION(values.size()!=18&&values.size()!=30,std::runtime_error,"Wrong number of values in projection_out.dat");
//    for(int_t i=9;i<18;++i){
//      (*projective_params_)[i-9] = values[i];
//    }
//    if(values.size()>18){
//      for(int_t i=18;i<30;++i){
//        (*warp_params_)[i-18] = values[i];
//      }
//    }
//  }
//  else{

    // A quick note on the strategy here: ultimately, we'd like to calibrate a 12 parameter
    // warp model and an 8 parameter projective transform from right to left.

    // To start we least squares fit the projective transform given four corresponding points
    // from the left to right image, or use feature matching. Then we add on top of that the warp to account for distortions

  std::vector<scalar_t> proj_xl;
  std::vector<scalar_t> proj_yl;
  std::vector<scalar_t> proj_xr;
  std::vector<scalar_t> proj_yr;
  int_t num_coords = 0;
  create_directory(".dice");

  if(use_nonlinear_projection){
    DEBUG_MSG("Triangulation::estimate_projective_transform(): reading initial guess points from file: projection_points.dat");
    // read the projection points from projection_points.dat
    std::fstream projDataFile("projection_points.dat", std::ios_base::in);
    TEUCHOS_TEST_FOR_EXCEPTION(!projDataFile.good(),std::runtime_error,
      "Error, could not open file projection_points.dat (required for cross-correlation)");
    while(!projDataFile.eof()){
      std::vector<std::string> tokens = tokenize_line(projDataFile);
      if(tokens.size()==0) continue;
      TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=4,std::runtime_error,
        "Error reading projection_points.dat, should be 4 values per line (x_left y_left x_righ y_right),"
        " but found " << tokens.size() << " values on line " << num_coords+1);
      num_coords++;
    }
    DEBUG_MSG("Triangulation::estimate_projective_transform(): found projection_points.dat file with " << num_coords << " points");
    TEUCHOS_TEST_FOR_EXCEPTION(num_coords<4,std::runtime_error,
      "Error, not enough sets of coordinates in projection_points.dat to estimate projection (needs at least 4)");
    proj_xl.resize(num_coords);
    proj_yl.resize(num_coords);
    proj_xr.resize(num_coords);
    proj_yr.resize(num_coords);
    projDataFile.clear();
    projDataFile.seekg(0, std::ios::beg);
    int_t coord_index = 0;
    while(!projDataFile.eof()){
      std::vector<std::string> tokens = tokenize_line(projDataFile);
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
  }else{
#ifdef DICE_ENABLE_OPENCV
    DEBUG_MSG("Triangulation::estimate_projective_transform(): begin matching features");
    float feature_tol = 0.005f;
    std::stringstream outname;
    outname << ".dice/fm_projective_trans_" << processor_id << ".png";
    match_features(left_img,right_img,proj_xl,proj_yl,proj_xr,proj_yr,feature_tol,outname.str());
    if(proj_xl.size() < 5){
      DEBUG_MSG("Triangulation::estimate_projective_transform(): initial attempt failed with tol = 0.005f, setting to 0.001f and trying again.");
      feature_tol = 0.001f;
      match_features(left_img,right_img,proj_xl,proj_yl,proj_xr,proj_yr,feature_tol,outname.str());
    }
    DEBUG_MSG("Triangulation::estimate_projective_transform(): matching features complete");
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,"Error, OpenCV required for cross correlation initialization.");
#endif
    if(proj_xl.size() < 5){
      return -1;
    }
    //TEUCHOS_TEST_FOR_EXCEPTION(proj_xl.size()<5, std::runtime_error,"Error, not enough features matched to estimate projection parameters");
    num_coords = proj_xl.size();
  }

  Teuchos::SerialDenseMatrix<int_t,double> TrHtildeTl = compute_affine_matrix(proj_xl,proj_yl,proj_xr,proj_yr);
  (*projective_params_)[0] = TrHtildeTl(0,0);
  (*projective_params_)[1] = TrHtildeTl(0,1);
  (*projective_params_)[2] = TrHtildeTl(0,2);
  (*projective_params_)[3] = TrHtildeTl(1,0);
  (*projective_params_)[4] = TrHtildeTl(1,1);
  (*projective_params_)[5] = TrHtildeTl(1,2);
  (*projective_params_)[6] = TrHtildeTl(2,0);
  (*projective_params_)[7] = TrHtildeTl(2,1);
  (*projective_params_)[8] = TrHtildeTl(2,2);

  // create an output file with the initial solution and final solution for projection params
  std::FILE * filePtr = fopen("projection_out.dat","w");
  fprintf(filePtr,"# Projection parameters from point matching: \n");
  for(size_t i=0;i<projective_params_->size();++i){
    fprintf(filePtr,"%e\n",(*projective_params_)[i]);
  }
  fclose(filePtr);

  if(output_projected_image){
    const int_t w = left_img->width();
    const int_t h = left_img->height();
    Teuchos::RCP<Image> img = Teuchos::rcp(new Image(w,h,0));
    Teuchos::ArrayRCP<storage_t> intens = img->intensities();
    scalar_t xr = 0.0;
    scalar_t yr = 0.0;
    for(int_t j=0;j<h;++j){
      for(int_t i=0;i<w;++i){
        project_left_to_right_sensor_coords(i,j,xr,yr);
        intens[j*w+i] = static_cast<storage_t>(right_img->interpolate_keys_fourth(xr,yr));
      }
    }
    img->write(".dice/right_projected_to_left_initial.tif");
  }

  // for each point, plug in the left coords and compute the right
  for(int_t i=0;i<num_coords;++i){
    scalar_t comp_right_x = 0.0;
    scalar_t comp_right_y = 0.0;
    project_left_to_right_sensor_coords(proj_xl[i],proj_yl[i],comp_right_x,comp_right_y);
    //DEBUG_MSG("input left x: " << proj_xl[i] << " y: " << proj_yl[i] << " right x: " << proj_xr[i] << " y: " << proj_yr[i] << " computed right x: " << comp_right_x << " y: " << comp_right_y);
    error+=(comp_right_x - proj_xr[i])*(comp_right_x - proj_xr[i]) + (comp_right_y - proj_yr[i])*(comp_right_y - proj_yr[i]);
  }
  error = std::sqrt(error)/num_coords;
  DEBUG_MSG("Triangulation::estimate_projective_transform(): initial projection error: " << error);
  if(error > 100) return -3;
  //TEUCHOS_TEST_FOR_EXCEPTION(error > 100.0,std::runtime_error,"Error, initial projection error too large");

  // simplex optimize the coefficients
  Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  params->set(DICe::max_iterations,200);
  params->set(DICe::tolerance,0.00001);
  DICe::Homography_Simplex simplex(left_img,right_img,this,params);
  Teuchos::RCP<std::vector<scalar_t> > deltas = Teuchos::rcp(new std::vector<scalar_t>(9,0.0));
  if(use_nonlinear_projection){
    for(size_t i=0;i<deltas->size();++i){
      (*deltas)[i] = 0.1*(*projective_params_)[i];
    }
    (*deltas)[2] = 0.1;
    (*deltas)[5] = 0.1;
  }
  else{
    (*deltas)[0] = 0.0001;
    (*deltas)[1] = 0.0001;
    (*deltas)[2] = 0.1;
    (*deltas)[3] = 0.0000001;
    (*deltas)[4] = 0.00001;
    (*deltas)[5] = 0.1;
    (*deltas)[6] = 0.00000001;
    (*deltas)[7] = 0.00000001;
    (*deltas)[8] = 0.0001;
  }
  int_t num_iterations = 0;
  Status_Flag corr_status = simplex.minimize(projective_params_,deltas,num_iterations);
  TEUCHOS_TEST_FOR_EXCEPTION(corr_status!=CORRELATION_SUCCESSFUL,std::runtime_error,"Error, could not determine projective transform.");

  filePtr = fopen("projection_out.dat","a");
  fprintf(filePtr,"# Projection parameters after simplex optimization: \n");
  for(size_t i=0;i<projective_params_->size();++i){
    fprintf(filePtr,"%e\n",(*projective_params_)[i]);
  }
  fprintf(filePtr,"# Optimization took %i iterations\n",num_iterations);
  fclose(filePtr);

  const int_t w = left_img->width();
  const int_t h = left_img->height();
  Teuchos::RCP<Image> proj_img = Teuchos::rcp(new Image(w,h,0));
  if(output_projected_image){
    Teuchos::ArrayRCP<storage_t> intens = proj_img->intensities();
    scalar_t xr = 0.0;
    scalar_t yr = 0.0;
    for(int_t j=0;j<h;++j){
      for(int_t i=0;i<w;++i){
        project_left_to_right_sensor_coords(i,j,xr,yr);
        intens[j*w+i] = static_cast<storage_t>(right_img->interpolate_keys_fourth(xr,yr));
      }
    }
    proj_img->write(".dice/right_projected_to_left_proj_opt.tif");
  }

  if(use_nonlinear_projection){
    // now that the initial projection is complete, optimize a warp to rectify the projected images:
    std::vector<scalar_t> warp_xl;
    std::vector<scalar_t> warp_yl;
    std::vector<scalar_t> warp_xr;
    std::vector<scalar_t> warp_yr;
#ifdef DICE_ENABLE_OPENCV
    DEBUG_MSG("Triangulation::estimate_projective_transform(): begin matching features for warp parameters");
    Teuchos::RCP<Image> projection_opt_img = Teuchos::rcp(new Image(".dice/right_projected_to_left_proj_opt.tif"));
    const float tol = 0.001f;
    std::stringstream outname_nonlin;
    outname_nonlin << ".dice/fm_nonlinear_proj_trans_" << processor_id << ".png";
    match_features(left_img,projection_opt_img,warp_xl,warp_yl,warp_xr,warp_yr,tol,outname_nonlin.str());
    DEBUG_MSG("Triangulation::estimate_projective_transform(): matching warp features complete");
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,"Error, OpeCV required for cross correlation initialization.");
#endif

    const int_t num_warp_coords = warp_xl.size();
    DEBUG_MSG("Triangulation::estimate_projective_transform(): number of warp feature matches: " << num_warp_coords);
    if(num_warp_coords < 6)
      return -2;

    // use a least squares fit to estimate the parameters
    int WN = 12;
    Teuchos::SerialDenseMatrix<int_t,double> WK(num_warp_coords*2,WN,true);
    Teuchos::SerialDenseMatrix<int_t,double> WKTK(WN,WN,true);
    Teuchos::ArrayRCP<scalar_t> WF(num_warp_coords*2,0.0);
    Teuchos::ArrayRCP<scalar_t> WKTu(WN,0.0);
    for(int_t i=0;i<num_warp_coords;++i){
      WK(i*2+0,0) = warp_xl[i];
      WK(i*2+0,1) = warp_yl[i];
      WK(i*2+0,2) = warp_xl[i]*warp_yl[i];
      WK(i*2+0,3) = warp_xl[i]*warp_xl[i];
      WK(i*2+0,4) = warp_yl[i]*warp_yl[i];
      WK(i*2+0,5) = 1.0;
      WK(i*2+1,6) = warp_xl[i];
      WK(i*2+1,7) = warp_yl[i];
      WK(i*2+1,8) = warp_xl[i]*warp_yl[i];
      WK(i*2+1,9) = warp_xl[i]*warp_xl[i];
      WK(i*2+1,10) = warp_yl[i]*warp_yl[i];
      WK(i*2+1,11) = 1.0;
      WF[i*2+0] = warp_xr[i];
      WF[i*2+1] = warp_yr[i];
    }
    // set up K^T*K
    for(int_t k=0;k<WN;++k){
      for(int_t m=0;m<WN;++m){
        for(int_t j=0;j<num_warp_coords*2;++j){
          WKTK(k,m) += WK(j,k)*WK(j,m);
        }
      }
    }
    int *WIPIV = new int[WN+1];
    int WLWORK = WN*WN;
    int WINFO = 0;
    double *WWORK = new double[WLWORK];
    // invert the WKTK matrix
    static Teuchos::LAPACK<int_t,double> lapack;
    lapack.GETRF(WKTK.numRows(),WKTK.numCols(),WKTK.values(),WKTK.numRows(),WIPIV,&WINFO);
    lapack.GETRI(WKTK.numRows(),WKTK.values(),WKTK.numRows(),WIPIV,WWORK,WLWORK,&WINFO);
    // compute K^T*F
    for(int_t i=0;i<WN;++i){
      for(int_t j=0;j<num_warp_coords*2;++j){
        WKTu[i] += WK(j,i)*WF[j];
      }
    }
    std::vector<scalar_t> warp_param_update(WN,0.0);
    // compute the coeffs
    for(int_t i=0;i<WN;++i){
      for(int_t j=0;j<WN;++j){
        warp_param_update[i] += WKTK(i,j)*WKTu[j];
      }
    }
    (*warp_params_)[0] = warp_param_update[5];
    (*warp_params_)[1] = warp_param_update[0];
    (*warp_params_)[2] = warp_param_update[1];
    (*warp_params_)[3] = warp_param_update[2];
    (*warp_params_)[4] = warp_param_update[3];
    (*warp_params_)[5] = warp_param_update[4];
    (*warp_params_)[6] = warp_param_update[11];
    (*warp_params_)[7] = warp_param_update[6];
    (*warp_params_)[8] = warp_param_update[7];
    (*warp_params_)[9] = warp_param_update[8];
    (*warp_params_)[10] = warp_param_update[9];
    (*warp_params_)[11] = warp_param_update[10];

    delete [] WWORK;
    delete [] WIPIV;

    for(size_t i=0;i<warp_params_->size();++i){
      TEUCHOS_TEST_FOR_EXCEPTION(std::isnan((*warp_params_)[i]),std::runtime_error,"Error, warp parameter " << i << " is nan");
      DEBUG_MSG("Triangulation::estimate_projective_transform(): initial least squares min value for param " << i << " " << (*warp_params_)[i]);
    }

    Teuchos::RCP<Teuchos::ParameterList> warp_params = rcp(new Teuchos::ParameterList());
    warp_params->set(DICe::max_iterations,500);
    warp_params->set(DICe::tolerance,0.00001);
    DICe::Warp_Simplex warp_simplex(left_img,right_img,this,warp_params);
    Teuchos::RCP<std::vector<scalar_t> > warp_deltas = Teuchos::rcp(new std::vector<scalar_t>(12,0.0));
    (*warp_deltas)[0]  = 1.0;  // shift x
    (*warp_deltas)[1]  = 0.1; // x scale
    (*warp_deltas)[2]  = 0.05; // x shear
    (*warp_deltas)[3]  = 0.000001; // y-dep x scale
    (*warp_deltas)[4]  = 0.000001; // nonlin x scale
    (*warp_deltas)[5]  = 0.0001; // y-dep x shift
    (*warp_deltas)[6]  = 0.1; // y shift
    (*warp_deltas)[7]  = 0.001; // y shear
    (*warp_deltas)[8]  = 0.01; // y scale
    (*warp_deltas)[9]  = 0.000001; // x-dep y scale
    (*warp_deltas)[10] = 0.000001; // x-dep y shift
    (*warp_deltas)[11] = 0.00001; // nonlin y scale
    num_iterations = 0;
    corr_status = warp_simplex.minimize(warp_params_,warp_deltas,num_iterations);
    TEUCHOS_TEST_FOR_EXCEPTION(corr_status!=CORRELATION_SUCCESSFUL,std::runtime_error,"Error, could not determine warp transform.");

    filePtr = fopen("projection_out.dat","a");
    fprintf(filePtr,"# Nonlinear warp parameters after simplex optimization: \n");
    for(size_t i=0;i<warp_params_->size();++i){
      fprintf(filePtr,"%e\n",(*warp_params_)[i]);
    }
    fprintf(filePtr,"# Optimization took %i iterations\n",num_iterations);
    fclose(filePtr);
  }
  //  } // end projection_out.dat file does not exist

  // create an image that overlaps the right and left using tinted colors:
  if(output_projected_image){
    const int_t w = left_img->width();
    const int_t h = left_img->height();
    Teuchos::RCP<Image> img = Teuchos::rcp(new Image(w,h,0));
    Teuchos::ArrayRCP<storage_t> intens = img->intensities();
    Teuchos::RCP<Image> diff_img = Teuchos::rcp(new Image(w,h,0));
    Teuchos::ArrayRCP<storage_t> diff_intens = diff_img->intensities();
    scalar_t xr = 0.0;
    scalar_t yr = 0.0;
    for(int_t j=0;j<h;++j){
      for(int_t i=0;i<w;++i){
        project_left_to_right_sensor_coords(i,j,xr,yr);
        diff_intens[j*w+i] = static_cast<storage_t>((*left_img)(i,j) - right_img->interpolate_keys_fourth(xr,yr));
        intens[j*w+i] = static_cast<storage_t>(right_img->interpolate_keys_fourth(xr,yr));
      }
    }
    diff_img->write(".dice/right_projected_to_left_diff.tif");
    img->write(".dice/right_projected_to_left_final.tif");
    left_img->write_overlap_image(".dice/right_projected_to_left_color.tif",img);
  }
  return 0;
}

void
Triangulation::project_left_to_right_sensor_coords(const scalar_t & xl,
  const scalar_t & yl,
  scalar_t & xr,
  scalar_t & yr){

  // first apply the quadratic warp transform

  assert(warp_params_!=Teuchos::null);
  assert(warp_params_->size()==12);
  std::vector<scalar_t> & wp = *warp_params_;
  const scalar_t xt = wp[0] + wp[1]*xl + wp[2]*yl + wp[3]*xl*yl + wp[4]*xl*xl + wp[5]*yl*yl;
  const scalar_t yt = wp[6] + wp[7]*xl + wp[8]*yl + wp[9]*xl*yl + wp[10]*xl*xl + wp[11]*yl*yl;

  // then apply the projective transform

  assert(projective_params_!=Teuchos::null);
  assert(projective_params_->size()==9);
  std::vector<scalar_t> & pr = *projective_params_;
  xr = (pr[0]*xt + pr[1]*yt + pr[2])/(pr[6]*xt + pr[7]*yt + pr[8]);
  yr = (pr[3]*xt + pr[4]*yt + pr[5])/(pr[6]*xt + pr[7]*yt + pr[8]);
}

/// compute the fundamental matrix and return it as an opencv mat
cv::Mat
Triangulation::camera_matrix(const int index) const{
  cv::Mat mat = cv::Mat::zeros(3, 3, CV_64F); // the data type here is double since opencv routines that use these matrices require this
  mat.at<double>(0,0) = cal_intrinsics_[index][Camera::FX];
  mat.at<double>(0,2) = cal_intrinsics_[index][Camera::CX];
  mat.at<double>(1,1) = cal_intrinsics_[index][Camera::FY];
  mat.at<double>(1,2) = cal_intrinsics_[index][Camera::CY];
  mat.at<double>(2,2) = 1.0;
  return mat;
}

/// compute the distortion matrix and return it as an opencv mat
cv::Mat
Triangulation::distortion_matrix(const int index) const{
  cv::Mat mat = cv::Mat::zeros(5, 1, CV_64F);
  mat.at<double>(0) = cal_intrinsics_[index][Camera::K1];
  mat.at<double>(1) = cal_intrinsics_[index][Camera::K2];
  mat.at<double>(2) = cal_intrinsics_[index][Camera::P1];
  mat.at<double>(3) = cal_intrinsics_[index][Camera::P2];
  mat.at<double>(4) = cal_intrinsics_[index][Camera::K3];
  return mat;
}

/// compute the translation matrix and return it as an opencv mat
cv::Mat
Triangulation::translation_matrix() const{
  cv::Mat mat = cv::Mat::zeros(3, 1, CV_64F);
  for(size_t i=0;i<3;++i)
	  mat.at<double>(i) = cam_0_to_cam_1_(i,3);
  return mat;
}

/// compute the rotation matrix and return it as an opencv mat
cv::Mat
Triangulation::rotation_matrix() const{
  cv::Mat mat = cv::Mat::zeros(3, 3, CV_64F);
  for(size_t i=0;i<3;++i){
	  for(size_t j=0;j<3;++j){
		  mat.at<double>(i,j) = cam_0_to_cam_1_(i,j);
	  }
  }
  return mat;
}

void update_legacy_txt_cal_input(const Teuchos::RCP<Teuchos::ParameterList> & input_params){
  DEBUG_MSG("update_legacy_txt_cal_input(): function called");
  if(!input_params->isParameter(DICe::calibration_parameters_file)) return; // the legacy txt files would have had this as an input parameter
  const std::string cal_file_name = input_params->get<std::string>(DICe::calibration_parameters_file);
  if (cal_file_name.find(".txt") == std::string::npos) return; // not a text file
  std::fstream dataFile(cal_file_name.c_str(), std::ios_base::in);
  if (!dataFile.good()) {
    TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(), std::runtime_error,
      "Error, the camera system file does not exist or is corrupt: " << cal_file_name);
  }
  int_t total_num_values = 0;
  //Run through the file to determine the format
  while (!dataFile.eof()){
    std::vector<std::string>tokens = tokenize_line(dataFile, " \t<>");
    if (tokens.size() == 0) continue;
    if (tokens[0] == "#") continue;
    if (tokens[0] == "TRANSFORM") {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, custom transforms are no longer supported in the txt calibration file format");
    }
    total_num_values++;
  }
  dataFile.close();
  if(total_num_values!=22&&total_num_values!=28){ // either an invalid file or the width and height have already been added
    return;
  }
  int_t width = -1;
  int_t height = -1;
  std::vector<std::string> image_files;
  std::vector<std::string> stereo_image_files;
  int_t frame_id_start=0,num_frames=1,frame_skip=1;
  DICe::decipher_image_file_names(input_params,image_files,stereo_image_files,frame_id_start,num_frames,frame_skip);
  DICe::utils::read_image_dimensions(image_files[0].c_str(),width,height);
  TEUCHOS_TEST_FOR_EXCEPTION(width<0||height<0,std::runtime_error,
    "Error, invalid image dimensions");
  DEBUG_MSG("update_legacy_txt_cal_input(): updating the legacy txt input file to include WIDTH " << width << " HEIGHT " << height);
  std::fstream outFile(cal_file_name.c_str(), std::ios_base::app);
  outFile << height << " # HEIGHT" << std::endl;
  outFile << width << " # WIDTH" << std::endl;
  outFile.close();
}

void update_vic3d_cal_input(const Teuchos::RCP<Teuchos::ParameterList> & input_params,
		const int_t im_width,
		const int_t im_height){
  bool has_cal_file = input_params->isParameter(DICe::calibration_parameters_file);
  const std::string file = has_cal_file ? input_params->get<std::string>(DICe::calibration_parameters_file):
      input_params->get<std::string>(DICe::camera_system_file);
  DEBUG_MSG("update_vic3d_cal_input(): cal file: " << file);
  const std::string xml("xml");
  if(file.find(xml) == std::string::npos){
	  DEBUG_MSG("update_vic3d_cal_input(): not an xml file, exiting");
	  return; // not an xml file
  }

  // check for IMAGE_HEIGHT_WIDTH key word
  std::ifstream t(file);
  std::string str((std::istreambuf_iterator<char>(t)),
                   std::istreambuf_iterator<char>());
  std::transform(str.begin(), str.end(),str.begin(),::toupper);
  //std::cout << str << std::endl;
  const std::string imhw("IMAGE_HEIGHT_WIDTH");
  if(str.find(imhw)!=std::string::npos){
	  DEBUG_MSG("update_vic3d_cal_input(): found image width and height as xml parameters, exiting");
	  return; // no updates needed

  }
  DEBUG_MSG("update_vic3d_cal_input(): adding width and height to xml parameters" << file);
  create_directory(".dice");
  std::ifstream  src(file,                 std::ios::binary);
  std::ofstream  dst(".dice/.cal_copy.xml",std::ios::binary);
  dst << src.rdbuf();

  int_t width = -1;
  int_t height = -1;
  // get the width and height from an image or the input parameters
  if(im_width<0||im_height<0){
	  std::vector<std::string> image_files;
	  std::vector<std::string> stereo_image_files;
	  int_t frame_id_start=0,num_frames=1,frame_skip=1;
	  DICe::decipher_image_file_names(input_params,image_files,stereo_image_files,frame_id_start,num_frames,frame_skip);
	  DICe::utils::read_image_dimensions(image_files[0].c_str(),width,height);
  }
  else{
	  width = im_width;
	  height = im_height;
  }
  dst << "<polygonmask width=\"" << width << "\" height=\"" << height << "\" lri=\"polygonmask\">\n";
  dst.close();

  if(has_cal_file)
    input_params->set(calibration_parameters_file,".dice/.cal_copy.xml");
  else
    input_params->set(camera_system_file,".dice/.cal_copy.xml");

  DEBUG_MSG("update_vic3d_cal_input(): updated file .dice/.cal_copy.xml");
}


}// End DICe Namespace
