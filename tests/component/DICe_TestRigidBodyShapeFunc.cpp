// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright 2015 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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

#include <DICe.h>
#include <DICe_Subset.h>
#include <DICe_LocalShapeFunction.h>
#include <DICe_Image.h>
#include <DICe_Shape.h>
#include <DICe_PointCloud.h>
#include <DICe_Schema.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_ParameterList.hpp>

using namespace DICe;

// function that takes an image that represents an object in physical space (pixels coords are treated as mm coordinates)
// and a set of global coordinates and returns a pixel value
work_t get_intensity_from_world_coords(Teuchos::RCP<Image> image,
  const work_t world_x,
  const work_t world_y,
  const work_t world_z){
#if DICE_USE_DOUBLE
  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(world_z)>1.0E-8,std::runtime_error,"world_z " << world_z);
#endif
  const work_t facet_width = 100.0;
  const work_t factor_x = image->width()-1.0;
  const work_t factor_y = image->height()-1.0;

  const work_t scaled_x = (world_x + 0.5*facet_width)*factor_x/facet_width;
  const work_t scaled_y = (world_y + 0.5*facet_width)*factor_y/facet_width;

  // out of bounds pixels are assigned a value of 0.0
  return image->interpolate_keys_fourth(scaled_x,scaled_y);
}

void least_squares_inverse_map(const std::vector<work_t> & in_x,
  const std::vector<work_t> & in_y,
  const std::vector<work_t> & in_mapped_x,
  const std::vector<work_t> & in_mapped_y,
  std::vector<work_t> & out_unmapped_x,
  std::vector<work_t> & out_unmapped_y,
  std::vector<bool> & is_inside_neighborhood){

  const size_t num_pixels = in_x.size();
  TEUCHOS_TEST_FOR_EXCEPTION(in_y.size()!=num_pixels,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(in_mapped_x.size()!=num_pixels,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(in_mapped_y.size()!=num_pixels,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(out_unmapped_x.size()!=num_pixels,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(out_unmapped_y.size()!=num_pixels,std::runtime_error,"");
  TEUCHOS_TEST_FOR_EXCEPTION(is_inside_neighborhood.size()!=num_pixels,std::runtime_error,"");

  // create a point cloud of pixels to use for least squares regression below:
  Point_Cloud_2D<work_t> cloud;
  cloud.pts.resize(num_pixels);
  const int_t N = 3;
  const int_t num_neigh = 11;
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  double *GWORK = new double[10*N];
  int *IWORK = new int[LWORK];
  // Note, LAPACK does not allow templating on long int or work_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;

  for(size_t i=0;i<num_pixels;++i){
    cloud.pts[i].x = in_mapped_x[i];
    cloud.pts[i].y = in_mapped_y[i];
  }
  std::cout << "building the kd-tree" << std::endl;
  kd_tree_2d_t kd_tree(2 /*dim*/, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
  kd_tree.buildIndex();
  std::cout << "kd-tree completed" << std::endl;

  std::vector<size_t> ret_index(num_neigh);
  std::vector<work_t> out_dist_sqr(num_neigh);
  for(size_t i=0;i<num_pixels;++i){
    work_t query_pt[2];
    query_pt[0] = in_x[i];
    query_pt[1] = in_y[i];
    kd_tree.knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);

    Teuchos::ArrayRCP<double> neigh_x(num_neigh,0.0);
    Teuchos::ArrayRCP<double> neigh_y(num_neigh,0.0);
    Teuchos::ArrayRCP<double> X_t_u_x(N,0.0);
    Teuchos::ArrayRCP<double> X_t_u_y(N,0.0);
    Teuchos::ArrayRCP<double> coeffs_x(N,0.0);
    Teuchos::ArrayRCP<double> coeffs_y(N,0.0);
    Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_neigh, true);
    Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);
    work_t centroid_x = 0.0;
    work_t centroid_y = 0.0;
    for(size_t n=0;n<num_neigh;++n){
      const int_t neigh_id = ret_index[n];
      neigh_x[n] = in_x[neigh_id];
      neigh_y[n] = in_y[neigh_id];
      centroid_x += cloud.pts[neigh_id].x;
      centroid_y += cloud.pts[neigh_id].y;
      //std::cout << " neigh " << n << " neigh_x " << neigh_x[n] << " neigh_y " << neigh_y[n] << std::endl;
      // set up the X^T matrix
      X_t(0,n) = 1.0;
      X_t(1,n) = cloud.pts[neigh_id].x - query_pt[0];
      X_t(2,n) = cloud.pts[neigh_id].y - query_pt[1];
    }
    centroid_x /= num_neigh;
    centroid_y /= num_neigh;
    work_t max_neigh_dist_from_centroid = 0.0;
    for(size_t n=0;n<num_neigh;++n){
      const int_t neigh_id = ret_index[n];
      work_t neigh_dist_from_centroid = (cloud.pts[neigh_id].x - centroid_x)*(cloud.pts[neigh_id].x - centroid_x) + (cloud.pts[neigh_id].y - centroid_y)*(cloud.pts[neigh_id].y - centroid_y);
      if(neigh_dist_from_centroid > max_neigh_dist_from_centroid)
        max_neigh_dist_from_centroid = neigh_dist_from_centroid;
    }
    const work_t dist_from_centroid = (query_pt[0] - centroid_x)*(query_pt[0] - centroid_x) + (query_pt[1] - centroid_y)*(query_pt[1] - centroid_y);
    is_inside_neighborhood[i] = dist_from_centroid <= max_neigh_dist_from_centroid;

    // set up X^T*X
    for(int_t k=0;k<N;++k){
      for(int_t m=0;m<N;++m){
        for(int_t j=0;j<num_neigh;++j){
          X_t_X(k,m) += X_t(k,j)*X_t(m,j);
        }
      }
    }
    lapack.GETRF(X_t_X.numRows(),X_t_X.numCols(),X_t_X.values(),X_t_X.numRows(),IPIV,&INFO);
    lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),IPIV,WORK,LWORK,&INFO);
    // compute X^T*u
    for(int_t k=0;k<N;++k){
      for(int_t j=0;j<num_neigh;++j){
        X_t_u_x[k] += X_t(k,j)*neigh_x[j];
        X_t_u_y[k] += X_t(k,j)*neigh_y[j];
      }
    }
    // compute the coeffs
    for(int_t k=0;k<N;++k){
      for(int_t j=0;j<N;++j){
        coeffs_x[k] += X_t_X(k,j)*X_t_u_x[j];
        coeffs_y[k] += X_t_X(k,j)*X_t_u_y[j];
      }
    }
    out_unmapped_x[i] = coeffs_x[0];
    out_unmapped_y[i] = coeffs_y[0];
  }

  // memory clean up
  delete [] WORK;
  delete [] GWORK;
  delete [] IWORK;
  delete [] IPIV;
}

std::vector<work_t> motion(const work_t & t){
  std::vector<work_t> rigid_body_params(6,0.0);
  rigid_body_params[Rigid_Body_Shape_Function::ANGLE_X] = -5.0 + 0.5*t;
  rigid_body_params[Rigid_Body_Shape_Function::ANGLE_Y] = 2.3 - 0.45*t;
  rigid_body_params[Rigid_Body_Shape_Function::ANGLE_Z] = -23.7 + t;
  rigid_body_params[Rigid_Body_Shape_Function::TRANS_X] = 0.75*t;
  rigid_body_params[Rigid_Body_Shape_Function::TRANS_Y] = -0.65*t;
  rigid_body_params[Rigid_Body_Shape_Function::TRANS_Z] = 4.56 + 0.5*t; //9.784735812133
  return rigid_body_params;
}


int main(int argc, char *argv[]) {

  DICe::initialize(argc, argv);

  // only print output if args are given (for testing the output is quiet)
  int_t iprint     = argc - 1;
  int_t errorFlag  = 0;
  const work_t max_diff = 7.0;
  const work_t max_base_diff = 0.1;
  Teuchos::RCP<std::ostream> outStream;
  Teuchos::oblackholestream bhs; // outputs nothing
  if (iprint > 0)
    outStream = Teuchos::rcp(&std::cout, false);
  else
    outStream = Teuchos::rcp(&bhs, false);

  *outStream << "--- Begin test ---" << std::endl;

  Teuchos::RCP<Image> physical_ref = Teuchos::rcp(new Image("./images/refSpeckled.tif"));

  DICe::Camera_System cam_sys("./cal/rigid_body_shape_function.xml");
  //*outStream << cam_sys << std::endl;
  Teuchos::RCP<DICe::Camera> cam = cam_sys.camera(0);

  const int_t image_w = cam->image_width();
  const int_t image_h = cam->image_height();
  const int_t num_pixels = image_w*image_h;

  // create the reference image:
  std::vector<work_t> image_x(num_pixels,0.0);
  std::vector<work_t> image_y(num_pixels,0.0);
  std::vector<work_t> mapped_x(num_pixels,0.0);
  std::vector<work_t> mapped_y(num_pixels,0.0);
  std::vector<work_t> unmapped_x(num_pixels,0.0);
  std::vector<work_t> unmapped_y(num_pixels,0.0);
  std::vector<bool> is_in_neigh(num_pixels,true);
  std::vector<work_t> world_x(num_pixels,0.0);
  std::vector<work_t> world_y(num_pixels,0.0);
  std::vector<work_t> world_z(num_pixels,0.0);
  for(int_t y=0;y<image_h;++y){
    for(int_t x=0;x<image_w;++x){
      image_x[y*image_w+x] = x;
      image_y[y*image_w+x] = y;
    }
  }

  // create a base image to represent the plate in it's reference position
  std::vector<work_t> rigid_body_params(6,0.0);
  cam->image_to_world(image_x,image_y,rigid_body_params,world_x,world_y,world_z);
  Teuchos::ArrayRCP<storage_t> base_intensities(num_pixels,0.0);
  for(int_t i=0;i<num_pixels;++i){
    base_intensities[i] = std::round(get_intensity_from_world_coords(physical_ref,world_x[i],world_y[i],world_z[i]));
  }
  Teuchos::RCP<Image> base_img = Teuchos::rcp(new Image(image_w,image_h,base_intensities));
  //base_img->write("proj_shape_base_img.tiff");
  work_t base_diff = base_img->diff(physical_ref);
  base_diff/=num_pixels;
  *outStream << "diff of the base image vs. synthetic base image: " << base_diff << std::endl;
  if(base_diff>max_base_diff){
    errorFlag++;
    *outStream << "error: base image reconstruction failed" << std::endl;
  }

  Teuchos::RCP<Rigid_Body_Shape_Function> rbsf = Teuchos::rcp(new Rigid_Body_Shape_Function("./cal/rigid_body_shape_function.xml"));
  const size_t max_step = 20;
  *outStream << std::left << std::setw (10) << "step" << std::setw (10) << "result"  << std::setw (10) << "diff" << std::setw (10) <<
      "alpha" << std::setw (10) << "beta" << std::setw (10) << "gamma" << std::setw (10) <<
      "tx" << std::setw (10) << "ty" << std::setw (10) << "tz" << std::endl;
  for(size_t step=0;step<=max_step;++step){
    std::vector<work_t> rbp = motion((work_t)step);
    for(size_t i=0;i<6;++i)
      (*rbsf->parameters())[i] = rbp[i];

    std::stringstream file_name;
    file_name << "./images/rbm_speckle_" << step << ".tif";
    Teuchos::RCP<Image> def_img = Teuchos::rcp(new Image(file_name.str().c_str()));
    work_t avg_diff = 0.0;
    for(int_t i=0;i<num_pixels;++i){
      work_t mx = 0.0;
      work_t my = 0.0;
      rbsf->map(image_x[i],image_y[i],-1.0,-1.0,mx,my);
      // skip the outer edges:
      if(image_x[i]<10.0||image_x[i]>image_w-10.0||image_y[i]<10.0||image_y[i]>image_h-10.0)
        continue;
      // skip pixels out of the field of view in the deformed image
      if(def_img->interpolate_keys_fourth(mx,my)<=0.0)
        continue;
      work_t diff = std::abs(physical_ref->intensities()[i] - def_img->interpolate_keys_fourth(mx,my));
      avg_diff+=diff;
    }
    avg_diff/=num_pixels;
    std::string res = "PASS";
    if(avg_diff > max_diff){
      res = "FAIL";
      errorFlag++;
    }
    *outStream << std::left << std::setw (10) << step << std::setw (10) << res << std::setw (10) << avg_diff << std::setw (10) <<
        (*rbsf->parameters())[Rigid_Body_Shape_Function::ANGLE_X] << std::setw (10) << (*rbsf->parameters())[Rigid_Body_Shape_Function::ANGLE_Y] << std::setw (10) << (*rbsf->parameters())[Rigid_Body_Shape_Function::ANGLE_Z] << std::setw (10) <<
        (*rbsf->parameters())[Rigid_Body_Shape_Function::TRANS_X] << std::setw (10) << (*rbsf->parameters())[Rigid_Body_Shape_Function::TRANS_Y] << std::setw (10) << (*rbsf->parameters())[Rigid_Body_Shape_Function::TRANS_Z] << std::endl;

    // USE CODE BELOW TO GENERATE THE TEMPLATE IMAGES

//    // populate the point cloud to use least squares regression to get the inverse map
//    Teuchos::ArrayRCP<intensity_t> def_intensities(num_pixels,0.0);
//    for(int_t i=0;i<num_pixels;++i){
//      rbsf->map(image_x[i],image_y[i],-1.0,-1.0,mapped_x[i],mapped_y[i]);
//    }
//    least_squares_inverse_map(image_x,image_y,mapped_x,mapped_y,unmapped_x,unmapped_y,is_in_neigh);
//    for(int_t i=0;i<num_pixels;++i){
//      if(is_in_neigh[i])
//        def_intensities[i] = base_img->interpolate_keys_fourth(unmapped_x[i],unmapped_y[i]);
//    }
//    Teuchos::RCP<Image> def_img = Teuchos::rcp(new Image(image_w,image_h,def_intensities));
//    std::stringstream file_name;
//    file_name << "rbm_speckle_" << step << ".tif";
//    def_img->write(file_name.str());
  }

  // TODO uncomment this to start testing the correlation with the rigid_body_shape_function

//  *outStream << "testing correlation with rigid body shape function" << std::endl;
//
//  Teuchos::RCP<Teuchos::ParameterList> schema_params = rcp(new Teuchos::ParameterList());
//  schema_params->set(DICe::shape_function_type,DICe::RIGID_BODY_SF);
//  schema_params->set(DICe::camera_system_file,"./cal/rigid_body_shape_function.xml");
//  schema_params->set(DICe::max_solver_iterations_fast,500);
//  schema_params->set(DICe::optimization_method,SIMPLEX);
//  Teuchos::ArrayRCP<work_t> coords_x(1,255.0);
//  Teuchos::ArrayRCP<work_t> coords_y(1,255.0);
//  const int_t subset_size = 101;
//  Teuchos::RCP<DICe::Schema> schema =
//      Teuchos::rcp(new DICe::Schema(coords_x,coords_y,subset_size,Teuchos::null,Teuchos::null,schema_params));
//  // get the dimensions of the images and set up an array of points
//  schema->set_ref_image("./images/rbm_speckle_0.tif");
//
//  schema->global_field_value(0,DICe::field_enums::ROT_TRANS_3D_ANG_X_FS) = 0.5;
//  schema->global_field_value(0,DICe::field_enums::ROT_TRANS_3D_ANG_Y_FS) = -0.45;
//  schema->global_field_value(0,DICe::field_enums::ROT_TRANS_3D_ANG_Z_FS) = 1.0;
//  schema->global_field_value(0,DICe::field_enums::ROT_TRANS_3D_TRANS_X_FS) = 0.75;
//  schema->global_field_value(0,DICe::field_enums::ROT_TRANS_3D_TRANS_Y_FS) = -0.65;
//  schema->global_field_value(0,DICe::field_enums::ROT_TRANS_3D_TRANS_Z_FS) = 0.5;
//
//  // TODO change to all 20 steps instead of just this one
//  for(size_t step=1;step<=1;++step){
//    std::stringstream file_name;
//    file_name << "./images/rbm_speckle_" << step << ".tif";
//    schema->set_def_image(file_name.str().c_str());
//    schema->execute_correlation();
//  }

  *outStream << "--- End test ---" << std::endl;

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";

  return 0;

}

