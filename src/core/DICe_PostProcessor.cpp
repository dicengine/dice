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

#include <DICe_PostProcessor.h>

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

namespace DICe {

Post_Processor::Post_Processor(Schema * schema,
  const std::string & name) :
  schema_(schema),
  name_(name),
  data_num_points_(0)
{
  assert(schema);
}

void
Post_Processor::initialize(){
  data_num_points_ = schema_->global_num_subsets();
  assert(data_num_points_>0);
  std::vector<scalar_t> tmp_vec(data_num_points_,0.0);
  for(size_t i=0;i<field_names_.size();++i)
    fields_.insert(std::pair<std::string,std::vector<scalar_t> >(field_names_[i],tmp_vec));
}

VSG_Strain_Post_Processor::VSG_Strain_Post_Processor(Schema * schema,
  const Teuchos::RCP<Teuchos::ParameterList> & params) :
  Post_Processor(schema,post_process_vsg_strain)
{
  field_names_.push_back(vsg_strain_xx);
  field_names_.push_back(vsg_strain_yy);
  field_names_.push_back(vsg_strain_xy);
  field_names_.push_back(vsg_dudx);
  field_names_.push_back(vsg_dudy);
  field_names_.push_back(vsg_dvdx);
  field_names_.push_back(vsg_dvdy);
  DEBUG_MSG("Enabling post processor VSG_Strain_Post_Processor with associated fields:");
  for(size_t i=0;i<field_names_.size();++i){
    DEBUG_MSG(field_names_[i]);
  }
  set_params(params);
}

void
VSG_Strain_Post_Processor::set_params(const Teuchos::RCP<Teuchos::ParameterList> & params){
  assert(params!=Teuchos::null);
  if(!params->isParameter(strain_window_size_in_pixels)){
    std::cout << "Error: The strain window size must be specified in the VSG_Strain_Post_Processor block of the input" << std::endl;
    std::cout << "Please set the parameter \"strain_window_size_in_pixels\" " << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }
  window_size_ = params->get<int_t>(strain_window_size_in_pixels);
  TEUCHOS_TEST_FOR_EXCEPTION(window_size_<=0,std::runtime_error,"Error, window size must be greater than 0");
  DEBUG_MSG("VSG_Strain_Post_Processor strain window size: " << window_size_);
}

void
VSG_Strain_Post_Processor::pre_execution_tasks(){
  //if(schema_->mesh()->get_comm()->get_rank()!=0) return;
  DEBUG_MSG("VSG_Strain_Post_Processor pre_execution_tasks() begin");

  // Note neighborhood information is collected over square windows, not circular

  // This post processor requires that the points are set in a regular grid (although there can be gaps)
  // compute the i and j indices of each subset in terms of the step size:
  // get the step size for this analysis
  const int_t step_size_x = schema_->step_size_x();
  const int_t step_size_y = schema_->step_size_y();
  TEUCHOS_TEST_FOR_EXCEPTION(step_size_x<=0,std::runtime_error,"Error VSG requires that the step size parameter is used to layout the subsets in x.");
  TEUCHOS_TEST_FOR_EXCEPTION(step_size_y<=0,std::runtime_error,"Error VSG requires that the step size parameter is used to layout the subsets in x.");

  // gather an all owned field here
  Teuchos::RCP<MultiField> coords_x = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::INITIAL_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> coords_y = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::INITIAL_COORDINATES_Y_FS);

  // find the min/max x and y, these will be used to set up the rows and columns
  int_t min_x = schema_->ref_img()->width();
  int_t min_y = schema_->ref_img()->height();
  int_t max_x = 0;
  int_t max_y = 0;
  int_t x=0,y=0;
  assert(min_x>0&&min_y>0);
  DEBUG_MSG("width " << min_x << " height " << min_y);
  for(int_t i=0;i<data_num_points_;++i){
    x = coords_x->local_value(i);
    y = coords_y->local_value(i);
    if(x < min_x) min_x = x;
    if(y < min_y) min_y = y;
    if(x > max_x) max_x = x;
    if(y > max_y) max_y = y;
  }
  DEBUG_MSG("min x: " << min_x << " max_x " << max_x << " min_y " << min_y << " max_y " << max_y);
  const int_t num_cols = (max_x - min_x) / step_size_x + 1;
  const int_t num_rows = (max_y - min_y) / step_size_y + 1;
  DEBUG_MSG("num_rows " << num_rows << " num_cols " << num_cols);
  int_t arm_x = (window_size_/2)/step_size_x;
  int_t arm_y = (window_size_/2)/step_size_y;
  vec_stride_ = (arm_x*2+1)*(arm_y*2+1);
  DEBUG_MSG("Max number of neighbors per subset = vec_stride: " << vec_stride_);
  neighbor_lists_ = Teuchos::ArrayRCP<int_t>(data_num_points_*vec_stride_,-1);
  num_neigh_ = Teuchos::ArrayRCP<int_t>(data_num_points_,0);
  neighbor_distances_x_ = Teuchos::ArrayRCP<scalar_t>(data_num_points_*vec_stride_,-1.0);
  neighbor_distances_y_ = Teuchos::ArrayRCP<scalar_t>(data_num_points_*vec_stride_,-1.0);

  Teuchos::ArrayRCP<int_t> subset_id_grid(num_rows*num_cols,-1);
  // organize the subsets according to the step size grid
  int_t row=0,col=0;
  for(int_t subset=0;subset<data_num_points_;++subset){
    x = coords_x->local_value(subset);
    y = coords_y->local_value(subset);
    row = (y - min_y)/step_size_y;
    col = (x - min_x)/step_size_x;
    assert(row*num_cols + col < subset_id_grid.size());
    subset_id_grid[row*num_cols + col] = subset;
  }
//  // print out the subset_id_grid:
//  for(int_t j=0;j<num_rows;++j){
//    std::cout << "ROW : " << j << " ";
//    for(int_t i=0;i<num_cols;++i){
//      std::cout << " " << subset_id_grid[j*num_cols+i];
//    }
//    std::cout << std::endl;
//  }

  // now assign the neighbors:
  int_t subset_gid = 0;
  int_t neigh_gid = 0;
  for(row=0;row<num_rows;++row){
    for(col=0;col<num_cols;++col){
      subset_gid = subset_id_grid[row*num_cols + col];
      if(subset_gid<0)continue;
      for(int_t j=row-arm_y;j<=row+arm_y;++j){
        if(j<0||j>=num_rows)continue;
        for(int_t i=col-arm_x;i<=col+arm_x;++i){
          if(i<0||i>=num_cols)continue;
          neigh_gid = subset_id_grid[j*num_cols + i];
          if(neigh_gid<0)continue;
          assert(num_neigh_[subset_gid]>=0&&num_neigh_[subset_gid]<vec_stride_);
          neighbor_lists_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = neigh_gid;
          neighbor_distances_x_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = (i-col)*step_size_x;
          neighbor_distances_y_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = (j-row)*step_size_y;
          num_neigh_[subset_gid] += 1;
        } // i loop
      } // j loop
      if(num_neigh_[subset_gid]<3){
        std::cout << "Error: Subset " << subset_gid << " does not have enough subsets inside the strain window: " << window_size_ << std::endl;
        std::cout << "       There aren't enough neighbor points to fit the polynomial. " << std::endl;
        std::cout << "       The input parameter strain_window_size_in_pixels should be increased. " << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
    } // col loop
  } // row loop

//  DEBUG_MSG("VSG_Strain_Post_Processor neighbor lists");
//#ifdef DICE_DEBUG_MSG
//  for(int_t i=0;i<data_num_points_;++i){
//    std::stringstream ss;
//    ss << "Subset " << i << ":";
//    for(int_t j=0;j<num_neigh_[i];++j){
//      ss << " " << neighbor_lists_[i*vec_stride_ + j] << " (" << neighbor_distances_x_[i*vec_stride_ + j] << "," << neighbor_distances_y_[i*vec_stride_ + j] << ")";
//    }
//    DEBUG_MSG(ss.str());
//  }
//#endif
  DEBUG_MSG("VSG_Strain_Post_Processor pre_execution_tasks() end");
}

void
VSG_Strain_Post_Processor::execute(){
  //if(schema_->mesh()->get_comm()->get_rank()!=0) return;
  DEBUG_MSG("VSG_Strain_Post_Processor execute() begin");

  // gather an all owned field here
  Teuchos::RCP<MultiField> disp_x = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> disp_y = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::DISPLACEMENT_Y_FS);

  const int_t N = 3;
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  double *GWORK = new double[10*N];
  int *IWORK = new int[LWORK];
  // FIXME, LAPACK does not allow templating on long int or scalar_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;

  int_t num_neigh = 0;
  int_t neigh_id = 0;
  for(int_t subset=0;subset<data_num_points_;++subset){
    DEBUG_MSG("Processing subset " << subset << " of " << data_num_points_);
    num_neigh = num_neigh_[subset];
    Teuchos::ArrayRCP<double> u_x(num_neigh,0.0);
    Teuchos::ArrayRCP<double> u_y(num_neigh,0.0);
    Teuchos::ArrayRCP<double> X_t_u_x(N,0.0);
    Teuchos::ArrayRCP<double> X_t_u_y(N,0.0);
    Teuchos::ArrayRCP<double> coeffs_x(N,0.0);
    Teuchos::ArrayRCP<double> coeffs_y(N,0.0);
    Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_neigh, true);
    Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);

    // gather the displacements of the neighbors
    for(int_t j=0;j<num_neigh;++j){
      neigh_id = neighbor_lists_[subset*vec_stride_ + j];
      u_x[j] = disp_x->local_value(neigh_id);
      u_y[j] = disp_y->local_value(neigh_id);
    }

    // set up the X^T matrix
    for(int_t j=0;j<num_neigh;++j){
      X_t(0,j) = 1.0;
      X_t(1,j) = neighbor_distances_x_[subset*vec_stride_ + j];
      X_t(2,j) = neighbor_distances_y_[subset*vec_stride_ + j];
    }
    // set up X^T*X
    for(int_t k=0;k<N;++k){
      for(int_t m=0;m<N;++m){
        for(int_t j=0;j<num_neigh;++j){
          X_t_X(k,m) += X_t(k,j)*X_t(m,j);
        }
      }
    }
    //X_t_X.print(std::cout);

    // Invert X^T*X
    // TODO: remove for performance?
    // compute the 1-norm of H:
    std::vector<double> colTotals(X_t_X.numCols(),0.0);
    for(int_t i=0;i<X_t_X.numCols();++i){
      for(int_t j=0;j<X_t_X.numRows();++j){
        colTotals[i]+=std::abs(X_t_X(j,i));
      }
    }
    double anorm = 0.0;
    for(int_t i=0;i<X_t_X.numCols();++i){
      if(colTotals[i] > anorm) anorm = colTotals[i];
    }
    DEBUG_MSG("Subset " << subset << " anorm " << anorm);
    double rcond=0.0; // reciporical condition number
    try
    {
      lapack.GETRF(X_t_X.numRows(),X_t_X.numCols(),X_t_X.values(),X_t_X.numRows(),IPIV,&INFO);
      lapack.GECON('1',X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),anorm,&rcond,GWORK,IWORK,&INFO);
      DEBUG_MSG("Subset " << subset << " VSG X^T*X RCOND(H): "<< rcond);
      if(rcond < 1.0E-12) {
        std::cout << "Error: The pseudo-inverse of the VSG strain calculation is (or is near) singular." << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      std::cout << "Error: Something went wrong in the condition number calculation" << std::endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
    }
    try
    {
      lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),IPIV,WORK,LWORK,&INFO);
    }
    catch(std::exception &e){
      DEBUG_MSG( e.what() << '\n');
      std::cout << "Error: Something went wrong in the inverse calculation of X^T*X " << std::endl;
    }

    // compute X^T*u
    for(int_t i=0;i<N;++i){
      for(int_t j=0;j<num_neigh;++j){
        X_t_u_x[i] += X_t(i,j)*u_x[j];
        X_t_u_y[i] += X_t(i,j)*u_y[j];
      }
    }

    // compute the coeffs
    for(int_t i=0;i<N;++i){
      for(int_t j=0;j<N;++j){
        coeffs_x[i] += X_t_X(i,j)*X_t_u_x[j];
        coeffs_y[i] += X_t_X(i,j)*X_t_u_y[j];
      }
    }

    // update the field values
    const double dudx = coeffs_x[1];
    const double dudy = coeffs_x[2];
    const double dvdx = coeffs_y[1];
    const double dvdy = coeffs_y[2];
    field_value(subset,vsg_dudx) = dudx;
    field_value(subset,vsg_dudy) = dudy;
    field_value(subset,vsg_dvdx) = dvdx;
    field_value(subset,vsg_dvdy) = dvdy;

    DEBUG_MSG("Subset " << subset << " dudx " << field_value(subset,vsg_dudx) << " dudy " << field_value(subset,vsg_dudy) <<
      " dvdx " << field_value(subset,vsg_dvdx) << " dvdy " << field_value(subset,vsg_dvdy));

    // compute the Green-Lagrange strain based on the derivatives computed above:
    const scalar_t GL_xx = 0.5*(2.0*dudx + dudx*dudx + dvdx*dvdx);
    const scalar_t GL_yy = 0.5*(2.0*dvdy + dudy*dudy + dvdy*dvdy);
    const scalar_t GL_xy = 0.5*(dudy + dvdx + dudx*dudy + dvdx*dvdy);
    field_value(subset,vsg_strain_xx) = GL_xx;
    field_value(subset,vsg_strain_yy) = GL_yy;
    field_value(subset,vsg_strain_xy) = GL_xy;

    DEBUG_MSG("Subset " << subset << " VSG Green-Lagrange strain XX: " << field_value(subset,vsg_strain_xx) << " YY: " << field_value(subset,vsg_strain_yy) <<
      " XY: " << field_value(subset,vsg_strain_xy));

  } // end subset loop

  delete [] WORK;
  delete [] GWORK;
  delete [] IWORK;
  delete [] IPIV;

  DEBUG_MSG("VSG_Strain_Post_Processor execute() end");
}
//
//Global_Strain_Post_Processor::Global_Strain_Post_Processor(Schema * schema,
//  const Teuchos::RCP<Teuchos::ParameterList> & params) :
//  Post_Processor(schema,post_process_global_strain)
//{
//  field_names_.push_back(global_strain_xx);
//  field_names_.push_back(global_strain_yy);
//  field_names_.push_back(global_strain_xy);
//  field_names_.push_back(global_dudx);
//  field_names_.push_back(global_dudy);
//  field_names_.push_back(global_dvdx);
//  field_names_.push_back(global_dvdy);
//  DEBUG_MSG("Enabling post processor Global_Strain_Post_Processor with associated fields:");
//  for(size_t i=0;i<field_names_.size();++i){
//    DEBUG_MSG(field_names_[i]);
//  }
//  set_params(params);
//}
//
//void
//Global_Strain_Post_Processor::set_params(const Teuchos::RCP<Teuchos::ParameterList> & params){
//  assert(params!=Teuchos::null);
//  mesh_size_ = schema_->mesh_size();
//}
//
//void
//Global_Strain_Post_Processor::pre_execution_tasks(){
//  DEBUG_MSG("Global_Strain_Post_Processor pre_execution_tasks() begin");
//  DEBUG_MSG("Global_Strain_Post_Processor pre_execution_tasks() end");
//}
//
//void
//Global_Strain_Post_Processor::execute(){
//  DEBUG_MSG("Global_Strain_Post_Processor execute() begin");
//  // make sure the connectivity matrix is populated
//  assert(schema_->connectivity()->numRows()>0);
//  Teuchos::ArrayRCP<scalar_t> dN_dxi(4,0.0);
//  Teuchos::ArrayRCP<scalar_t> dN_deta(4,0.0);
//  Teuchos::ArrayRCP<scalar_t> elem_disp(8,0.0);
//  Teuchos::ArrayRCP<int_t> node_ids(4,0.0);
//  Teuchos::ArrayRCP<scalar_t> node_du_dx(data_num_points_,0.0);
//  Teuchos::ArrayRCP<scalar_t> node_du_dy(data_num_points_,0.0);
//  Teuchos::ArrayRCP<scalar_t> node_dv_dx(data_num_points_,0.0);
//  Teuchos::ArrayRCP<scalar_t> node_dv_dy(data_num_points_,0.0);
//  Teuchos::ArrayRCP<int_t> node_num_contribs(data_num_points_,0);
//  int_t start_x=0,end_x=0;
//  int_t start_y=0,end_y=0;
//  scalar_t center_x=0,center_y=0;
//  int_t elem_p_width=0,elem_p_height=0;
//
//  scalar_t xi=0.0,eta=0.0;
//  const int_t num_elem = schema_->connectivity()->numRows();
//  for(int_t elem=0;elem<num_elem;++elem){
//
//    for(int_t i=0;i<4;++i)
//      node_ids[i] = (*schema_->connectivity())(elem,i);
//
//    // create the local disp vector:
//    for(int_t i=0;i<4;++i){
//      elem_disp[i*2+0] = schema_->field_value(node_ids[i],DICe::DISPLACEMENT_X);
//      elem_disp[i*2+1] = schema_->field_value(node_ids[i],DICe::DISPLACEMENT_Y);
//    }
//
//    // get the pixel dimensions for this element's integration area
//    start_x = (int_t)(schema_->field_value(node_ids[0],DICe::COORDINATE_X) + 0.5);
//    end_x = (int_t)(schema_->field_value(node_ids[1],DICe::COORDINATE_X) - 0.5);
//    start_y = (int_t)(schema_->field_value(node_ids[0],DICe::COORDINATE_Y) + 0.5);
//    end_y = (int_t)(schema_->field_value(node_ids[3],DICe::COORDINATE_Y) - 0.5);
//    elem_p_width = end_x - start_x + 1;
//    elem_p_height = end_y - start_y + 1;
//    center_x = start_x + elem_p_width/2.0;
//    center_y = start_y + elem_p_height/2.0;
//
//    // iterate over the nodes of the element:
//    for(int_t node=0;node<4;++node){
//      xi = 2.0 * (schema_->field_value(node_ids[node],DICe::COORDINATE_X) - center_x)/elem_p_width;
//      eta = 2.0 * (schema_->field_value(node_ids[node],DICe::COORDINATE_Y) - center_y)/elem_p_height;
//
//      dN_dxi[0] = 0.25*(-1.0)*(1.0 - eta);
//      dN_dxi[1] = 0.25*(1.0 - eta);
//      dN_dxi[2] = 0.25*(1.0 + eta);
//      dN_dxi[3] = 0.25*(-1.0)*(1.0 + eta);
//      dN_deta[0] = 0.25*(1.0 - xi)*(-1.0);
//      dN_deta[1] = 0.25*(1.0 + xi)*(-1.0);
//      dN_deta[2] = 0.25*(1.0 + xi);
//      dN_deta[3] = 0.25*(1.0 - xi);
//
//      for(int_t i=0;i<4;++i){
//        node_du_dx[node_ids[node]]+=(2.0/elem_p_width)*dN_dxi[i]*elem_disp[i*2+0];
//        node_dv_dx[node_ids[node]]+=(2.0/elem_p_width)*dN_dxi[i]*elem_disp[i*2+1];
//        node_du_dy[node_ids[node]]+=(2.0/elem_p_height)*dN_deta[i]*elem_disp[i*2+0];
//        node_dv_dy[node_ids[node]]+=(2.0/elem_p_height)*dN_deta[i]*elem_disp[i*2+1];
//      }
//      for(int_t i=0;i<4;++i){
//        node_num_contribs[node_ids[i]]++;
//      }
//    } // elem node
//  } // elem
//
//  for(int_t i=0;i<data_num_points_;++i){
//    scalar_t du_dx = node_du_dx[i]/node_num_contribs[i];
//    scalar_t du_dy = node_du_dy[i]/node_num_contribs[i];
//    scalar_t dv_dx = node_dv_dx[i]/node_num_contribs[i];
//    scalar_t dv_dy = node_dv_dy[i]/node_num_contribs[i];
//    field_value(i,global_dudx) = du_dx;
//    field_value(i,global_dudy) = du_dy;
//    field_value(i,global_dvdx) = dv_dx;
//    field_value(i,global_dvdy) = dv_dy;
//
//    DEBUG_MSG("Node " << i << " dudx " << field_value(i,global_dudx) << " dudy " << field_value(i,global_dudy) <<
//      " dvdx " << field_value(i,global_dvdx) << " dvdy " << field_value(i,global_dvdy));
//
//    // compute the Green-Lagrange strain based on the derivatives computed above:
//    const scalar_t GL_xx = 0.5*(2.0*du_dx + du_dx*du_dx + dv_dx*dv_dx);
//    const scalar_t GL_yy = 0.5*(2.0*dv_dy + du_dy*du_dy + dv_dy*dv_dy);
//    const scalar_t GL_xy = 0.5*(du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy);
//    field_value(i,global_strain_xx) = GL_xx;
//    field_value(i,global_strain_yy) = GL_yy;
//    field_value(i,global_strain_xy) = GL_xy;
//
//    DEBUG_MSG("Node " << i << " global Green-Lagrange strain XX: " << field_value(i,global_strain_xx) << " YY: " << field_value(i,global_strain_yy) <<
//      " XY: " << field_value(i,global_strain_xy));
//  }
//  DEBUG_MSG("Global_Strain_Post_Processor execute() end");
//}

Keys4_Strain_Post_Processor::Keys4_Strain_Post_Processor(Schema * schema,
  const Teuchos::RCP<Teuchos::ParameterList> & params) :
  Post_Processor(schema,post_process_keys4_strain)
{
  field_names_.push_back(keys4_strain_xx);
  field_names_.push_back(keys4_strain_yy);
  field_names_.push_back(keys4_strain_xy);
  field_names_.push_back(keys4_dudx);
  field_names_.push_back(keys4_dudy);
  field_names_.push_back(keys4_dvdx);
  field_names_.push_back(keys4_dvdy);

  DEBUG_MSG("Enabling post processor Keys4_Strain_Post_Processor with associated fields:");
  for(size_t i=0;i<field_names_.size();++i){
    DEBUG_MSG(field_names_[i]);
  }

  set_params(params);
}

void
Keys4_Strain_Post_Processor::set_params(const Teuchos::RCP<Teuchos::ParameterList> & params){
  // currently a no-op
}

void
Keys4_Strain_Post_Processor::pre_execution_tasks(){
  //if(schema_->mesh()->get_comm()->get_rank()!=0) return;
  DEBUG_MSG("Keys4_Strain_Post_Processor pre_execution_tasks() begin");

  // gather an all owned field here
  Teuchos::RCP<MultiField> coords_x = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::INITIAL_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> coords_y = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::INITIAL_COORDINATES_Y_FS);

  // Note neighborhood information is collected over square windows, not circular
  // The window size is automatically set to make a 7 by 7 window of subsets
  // The window size is tied closely to the order of the strain measure

  // get the step size for this analysis
  const int_t step_size = schema_->step_size_x();
  TEUCHOS_TEST_FOR_EXCEPTION(step_size<=0,std::runtime_error,
    "Error Keys4 strain requires that the step size parameter is used to layout the subsets in x and y.");
  TEUCHOS_TEST_FOR_EXCEPTION(step_size!=schema_->step_size_y(),std::runtime_error,
    "Error Keys4 strain requires that the grid of subsets is equally spaced in x and y.");

  // find the min/max x and y, these will be used to set up the rows and columns
  int_t min_x = schema_->ref_img()->width();
  int_t min_y = schema_->ref_img()->height();
  int_t max_x = 0;
  int_t max_y = 0;
  int_t x=0,y=0;
  assert(min_x>0&&min_y>0);
  DEBUG_MSG("width " << min_x << " height " << min_y);
  for(int_t i=0;i<data_num_points_;++i){
    x = coords_x->local_value(i);
    y = coords_y->local_value(i);
    if(x < min_x) min_x = x;
    if(y < min_y) min_y = y;
    if(x > max_x) max_x = x;
    if(y > max_y) max_y = y;
  }
  DEBUG_MSG("min x: " << min_x << " max_x " << max_x << " min_y " << min_y << " max_y " << max_y);
  const int_t num_cols = (max_x - min_x) / step_size + 1;
  const int_t num_rows = (max_y - min_y) / step_size + 1;
  DEBUG_MSG("num_rows " << num_rows << " num_cols " << num_cols);
  int_t arm = 3;
  vec_stride_ = (arm*2+1)*(arm*2+1);
  DEBUG_MSG("Max number of neighbors per subset = vec_stride: " << vec_stride_);
  neighbor_lists_ = Teuchos::ArrayRCP<int_t>(data_num_points_*vec_stride_,-1);
  num_neigh_ = Teuchos::ArrayRCP<int_t>(data_num_points_,0);
  neighbor_distances_x_ = Teuchos::ArrayRCP<scalar_t>(data_num_points_*vec_stride_,-1.0);
  neighbor_distances_y_ = Teuchos::ArrayRCP<scalar_t>(data_num_points_*vec_stride_,-1.0);

  Teuchos::ArrayRCP<int_t> subset_id_grid(num_rows*num_cols,-1);
  // organize the subsets according to the step size grid
  int_t row=0,col=0;
  for(int_t subset=0;subset<data_num_points_;++subset){
    x = coords_x->local_value(subset);
    y = coords_y->local_value(subset);
    row = (y - min_y)/step_size;
    col = (x - min_x)/step_size;
    assert(row*num_cols + col < subset_id_grid.size());
    subset_id_grid[row*num_cols + col] = subset;
  }
//  // print out the subset_id_grid:
//  for(int_t j=0;j<num_rows;++j){
//    std::cout << "ROW : " << j << " ";
//    for(int_t i=0;i<num_cols;++i){
//      std::cout << " " << subset_id_grid[j*num_cols+i];
//    }
//    std::cout << std::endl;
//  }

  // now assign the neighbors:
  int_t subset_gid = 0;
  int_t neigh_gid = 0;
  for(row=0;row<num_rows;++row){
    for(col=0;col<num_cols;++col){
      subset_gid = subset_id_grid[row*num_cols + col];
      for(int_t j=row-arm;j<=row+arm;++j){
        if(j<0||j>=num_rows)continue;
        for(int_t i=col-arm;i<=col+arm;++i){
          if(i<0||i>=num_cols)continue;
          neigh_gid = subset_id_grid[j*num_cols + i];
          if(neigh_gid<0)continue;
          assert(num_neigh_[subset_gid]>=0&&num_neigh_[subset_gid]<vec_stride_);
          neighbor_lists_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = neigh_gid;
          neighbor_distances_x_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = (i-col);
          neighbor_distances_y_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = (j-row);
          num_neigh_[subset_gid] += 1;
        } // i loop
      } // j loop
      if(num_neigh_[subset_gid]<3){
        std::cout << "Error: Subset " << subset_gid << " does not have enough subsets inside the strain window for fourth order Keys." << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
    } // col loop
  } // row loop

//  DEBUG_MSG("Keys4_Strain_Post_Processor neighbor lists");
//#ifdef DICE_DEBUG_MSG
//  for(int_t i=0;i<data_num_points_;++i){
//    std::stringstream ss;
//    ss << "Subset " << i << ":";
//    for(int_t j=0;j<num_neigh_[i];++j){
//      ss << " " << neighbor_lists_[i*vec_stride_ + j] << " (" << neighbor_distances_x_[i*vec_stride_ + j] << "," << neighbor_distances_y_[i*vec_stride_ + j] << ")";
//    }
//    DEBUG_MSG(ss.str());
//  }
//#endif
  DEBUG_MSG("Keys4_Strain_Post_Processor pre_execution_tasks() end");
}

void
Keys4_Strain_Post_Processor::execute(){
  //if(schema_->mesh()->get_comm()->get_rank()!=0) return;
  DEBUG_MSG("Keys4_Strain_Post_Processor execute() begin");

  // gather an all owned field here
  Teuchos::RCP<MultiField> disp_x = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> disp_y = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::DISPLACEMENT_Y_FS);

  int_t num_neigh = 0;
  int_t neigh_id = 0;
  const int_t step_size = schema_->step_size_x(); // already checked above that x and y have the same step size and step_size > 0
  const scalar_t factor = 1.0/(step_size);
  for(int_t subset=0;subset<data_num_points_;++subset){
    DEBUG_MSG("Processing subset " << subset << " of " << data_num_points_);
    num_neigh = num_neigh_[subset];
    scalar_t dudx = 0.0;
    scalar_t dudy = 0.0;
    scalar_t dvdx = 0.0;
    scalar_t dvdy = 0.0;
    scalar_t dxS = 0.0, dx = 0.0, dx2 = 0.0, dx3 = 0.0;
    scalar_t dyS = 0.0, dy = 0.0, dy2 = 0.0, dy3 = 0.0;
    scalar_t sign_x = 1.0, sign_y = 1.0;
    scalar_t f0x=0.0, f0y=0.0;
    scalar_t f0xdx=0.0, f0ydy=0.0;
    scalar_t u_x=0.0, u_y=0.0;
    for(int_t j=0;j<num_neigh;++j){
      neigh_id = neighbor_lists_[subset*vec_stride_ + j];
      dxS = neighbor_distances_x_[subset*vec_stride_ + j];
      dyS = neighbor_distances_y_[subset*vec_stride_ + j];
      u_x = disp_x->local_value(neigh_id);
      u_y = disp_y->local_value(neigh_id);
      dx = std::abs(dxS);
      dy = std::abs(dyS);
      sign_x = (dxS < 0.0) ? -1.0 : 1.0;
      sign_y = (dyS < 0.0) ? -1.0 : 1.0;
      dy2=dy*dy;
      dy3=dy2*dy;
      dx2=dx*dx;
      dx3=dx2*dx;
      f0y = 0.0;
      f0ydy = 0.0;
      f0x = 0.0;
      f0xdx = 0.0;
      // Y values
      if(dy <= 1.0){
        f0y = 4.0/3.0*dy3 - 7.0/3.0*dy2 + 1.0;
        f0ydy = sign_y*4.0*dy2 - sign_y*14.0/3.0*dy;
      }
      else if(dy <= 2.0){
        f0y = -7.0/12.0*dy3 + 3.0*dy2 - 59.0/12.0*dy + 15.0/6.0;
        f0ydy = -sign_y*21.0/12.0*dy2 + sign_y*6.0*dy - sign_y*59.0/12.0;
      }
      else if(dy <= 3.0){
        f0y = 1.0/12.0*dy3 - 2.0/3.0*dy2 + 21.0/12.0*dy - 3.0/2.0;
        f0ydy = sign_y*3.0/12.0*dy2 - sign_y*4.0/3.0*dy + sign_y*21.0/12.0;
      }
      // X values
      if(dx <= 1.0){
        f0x = 4.0/3.0*dx3 - 7.0/3.0*dx2 + 1.0;
        f0xdx = sign_x*4.0*dx2 - sign_x*14.0/3.0*dx;
      }
      else if(dx <= 2.0){
        f0x = -7.0/12.0*dx3 + 3.0*dx2 - 59.0/12.0*dx + 15.0/6.0;
        f0xdx = -sign_x*21.0/12.0*dx2 + sign_x*6.0*dx - sign_x*59.0/12.0;
      }
      else if(dx <= 3.0){
        f0x = 1.0/12.0*dx3 - 2.0/3.0*dx2 + 21.0/12.0*dx - 3.0/2.0;
        f0xdx = sign_x*3.0/12.0*dx2 - sign_x*4.0/3.0*dx + sign_x*21.0/12.0;
      }
      dudx -= u_x*f0xdx*f0y*factor;
      dudy -= u_x*f0ydy*f0x*factor;
      dvdx -= u_y*f0xdx*f0y*factor;
      dvdy -= u_y*f0ydy*f0x*factor;
      DEBUG_MSG("Subset " << subset << " dxS " << dxS << " dyS " << dyS << " ux " << u_x << " uy " << u_y << " f0x " << f0x << " f0xdx " << f0xdx);
    } // neighbor loop
    field_value(subset,keys4_dudx) = dudx;
    field_value(subset,keys4_dudy) = dudy;
    field_value(subset,keys4_dvdx) = dvdx;
    field_value(subset,keys4_dvdy) = dvdy;
    DEBUG_MSG("Subset " << subset << " dudx " << field_value(subset,keys4_dudx) << " dudy " << field_value(subset,keys4_dudy) <<
      " dvdx " << field_value(subset,keys4_dvdx) << " dvdy " << field_value(subset,keys4_dvdy));
    // compute the Green-Lagrange strain based on the derivatives computed above:
    const scalar_t GL_xx = 0.5*(2.0*dudx + dudx*dudx + dvdx*dvdx);
    const scalar_t GL_yy = 0.5*(2.0*dvdy + dudy*dudy + dvdy*dvdy);
    const scalar_t GL_xy = 0.5*(dudy + dvdx + dudx*dudy + dvdx*dvdy);
    field_value(subset,keys4_strain_xx) = GL_xx;
    field_value(subset,keys4_strain_yy) = GL_yy;
    field_value(subset,keys4_strain_xy) = GL_xy;

    DEBUG_MSG("Subset " << subset << " Keys4 Green-Lagrange strain XX: " << field_value(subset,keys4_strain_xx) << " YY: " << field_value(subset,keys4_strain_yy) <<
      " XY: " << field_value(subset,keys4_strain_xy));
  } // end subset loop
  DEBUG_MSG("Keys4_Strain_Post_Processor execute() end");
}

NLVC_Strain_Post_Processor::NLVC_Strain_Post_Processor(Schema * schema,
  const Teuchos::RCP<Teuchos::ParameterList> & params) :
  Post_Processor(schema,post_process_nlvc_strain)
{
  field_names_.push_back(nlvc_strain_xx);
  field_names_.push_back(nlvc_strain_yy);
  field_names_.push_back(nlvc_strain_xy);
  field_names_.push_back(nlvc_dudx);
  field_names_.push_back(nlvc_dudy);
  field_names_.push_back(nlvc_dvdx);
  field_names_.push_back(nlvc_dvdy);
  field_names_.push_back(nlvc_integrated_alpha_x);
  field_names_.push_back(nlvc_integrated_alpha_y);
  field_names_.push_back(nlvc_integrated_phi_x);
  field_names_.push_back(nlvc_integrated_phi_y);
  DEBUG_MSG("Enabling post processor NLVC_Strain_Post_Processor with associated fields:");
  for(size_t i=0;i<field_names_.size();++i){
    DEBUG_MSG(field_names_[i]);
  }
  set_params(params);
}

void
NLVC_Strain_Post_Processor::set_params(const Teuchos::RCP<Teuchos::ParameterList> & params){
  assert(params!=Teuchos::null);
  if(!params->isParameter(horizon_diameter_in_pixels)){
    std::cout << "Error: The horizon diamter size must be specified in the NLVC_Strain_Post_Processor block of the input" << std::endl;
    std::cout << "Please set the parameter \"horizon_diameter_in_pixels\" " << std::endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
  }
  horizon_ = params->get<int_t>(horizon_diameter_in_pixels);
  TEUCHOS_TEST_FOR_EXCEPTION(horizon_<=0,std::runtime_error,
    "Error, horizon must be greater than 0");
  DEBUG_MSG("NLVC_Strain_Post_Processor horizon diameter size: " << horizon_);
}

void
NLVC_Strain_Post_Processor::pre_execution_tasks(){
  //if(schema_->mesh()->get_comm()->get_rank()!=0) return;

  DEBUG_MSG("NLVC_Strain_Post_Processor pre_execution_tasks() begin");

  // gather an all owned field here
  Teuchos::RCP<MultiField> coords_x = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::INITIAL_COORDINATES_X_FS);
  Teuchos::RCP<MultiField> coords_y = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::INITIAL_COORDINATES_Y_FS);

  // Note neighborhood information is collected over square windows, not circular

  // TODO unify all the neighbor searching algs so that they share code

  // This post processor requires that the points are set in a regular grid (although there can be gaps)
  // compute the i and j indices of each subset in terms of the step size:
  // get the step size for this analysis
  const int_t step_size_x = schema_->step_size_x();
  const int_t step_size_y = schema_->step_size_y();
  TEUCHOS_TEST_FOR_EXCEPTION(step_size_x<=0,std::runtime_error,
    "Error NLVC strain requires that the step size parameter is used to layout the subsets in x.");
  TEUCHOS_TEST_FOR_EXCEPTION(step_size_y<=0,std::runtime_error,
    "Error NLVC strain requires that the step size parameter is used to layout the subsets in x.");

  // find the min/max x and y, these will be used to set up the rows and columns
  int_t min_x = schema_->ref_img()->width();
  int_t min_y = schema_->ref_img()->height();
  int_t max_x = 0;
  int_t max_y = 0;
  int_t x=0,y=0;
  assert(min_x>0&&min_y>0);
  DEBUG_MSG("width " << min_x << " height " << min_y);
  for(int_t i=0;i<data_num_points_;++i){
    x = coords_x->local_value(i);
    y = coords_y->local_value(i);
    if(x < min_x) min_x = x;
    if(y < min_y) min_y = y;
    if(x > max_x) max_x = x;
    if(y > max_y) max_y = y;
  }
  DEBUG_MSG("min x: " << min_x << " max_x " << max_x << " min_y " << min_y << " max_y " << max_y);
  const int_t num_cols = (max_x - min_x) / step_size_x + 1;
  const int_t num_rows = (max_y - min_y) / step_size_y + 1;
  DEBUG_MSG("num_rows " << num_rows << " num_cols " << num_cols);
  int_t arm_x = (horizon_/2)/step_size_x;
  int_t arm_y = (horizon_/2)/step_size_y;
  vec_stride_ = (arm_x*2+1)*(arm_y*2+1);
  DEBUG_MSG("Max number of neighbors per subset = vec_stride: " << vec_stride_);
  neighbor_lists_ = Teuchos::ArrayRCP<int_t>(data_num_points_*vec_stride_,-1);
  num_neigh_ = Teuchos::ArrayRCP<int_t>(data_num_points_,0);
  neighbor_distances_x_ = Teuchos::ArrayRCP<scalar_t>(data_num_points_*vec_stride_,-1.0);
  neighbor_distances_y_ = Teuchos::ArrayRCP<scalar_t>(data_num_points_*vec_stride_,-1.0);

  Teuchos::ArrayRCP<int_t> subset_id_grid(num_rows*num_cols,-1);
  // organize the subsets according to the step size grid
  int_t row=0,col=0;
  for(int_t subset=0;subset<data_num_points_;++subset){
    x = coords_x->local_value(subset);
    y = coords_y->local_value(subset);
    row = (y - min_y)/step_size_y;
    col = (x - min_x)/step_size_x;
    assert(row*num_cols + col < subset_id_grid.size());
    subset_id_grid[row*num_cols + col] = subset;
  }
//  // print out the subset_id_grid:
//  for(int_t j=0;j<num_rows;++j){
//    std::cout << "ROW : " << j << " ";
//    for(int_t i=0;i<num_cols;++i){
//      std::cout << " " << subset_id_grid[j*num_cols+i];
//    }
//    std::cout << std::endl;
//  }

  // now assign the neighbors:
  int_t subset_gid = 0;
  int_t neigh_gid = 0;
  for(row=0;row<num_rows;++row){
    for(col=0;col<num_cols;++col){
      subset_gid = subset_id_grid[row*num_cols + col];
      for(int_t j=row-arm_y;j<=row+arm_y;++j){
        if(j<0||j>=num_rows)continue;
        for(int_t i=col-arm_x;i<=col+arm_x;++i){
          if(i<0||i>=num_cols)continue;
          neigh_gid = subset_id_grid[j*num_cols + i];
          if(neigh_gid<0)continue;
          assert(num_neigh_[subset_gid]>=0&&num_neigh_[subset_gid]<vec_stride_);
          neighbor_lists_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = neigh_gid;
          neighbor_distances_x_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = (i-col)*step_size_x;
          neighbor_distances_y_[subset_gid*vec_stride_ + num_neigh_[subset_gid]] = (j-row)*step_size_y;
          num_neigh_[subset_gid] += 1;
        } // i loop
      } // j loop
      if(num_neigh_[subset_gid]<3){
        std::cout << "Error: Subset " << subset_gid << " does not have enough subsets inside the nonlocal horizon: " << horizon_ << std::endl;
        std::cout << "       There aren't enough neighbor points to fit the polynomial. " << std::endl;
        std::cout << "       The input parameter horizon_diameter_in_pixels should be increased. " << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
    } // col loop
  } // row loop

//  DEBUG_MSG("NLVC_Strain_Post_Processor neighbor lists");
//#ifdef DICE_DEBUG_MSG
//  for(int_t i=0;i<data_num_points_;++i){
//    std::stringstream ss;
//    ss << "Subset " << i << ":";
//    for(int_t j=0;j<num_neigh_[i];++j){
//      ss << " " << neighbor_lists_[i*vec_stride_ + j] << " (" << neighbor_distances_x_[i*vec_stride_ + j] << "," << neighbor_distances_y_[i*vec_stride_ + j] << ")";
//    }
//    DEBUG_MSG(ss.str());
//  }
//#endif
  DEBUG_MSG("NLVC_Strain_Post_Processor pre_execution_tasks() end");
}

void
NLVC_Strain_Post_Processor::execute(){
  //if(schema_->mesh()->get_comm()->get_rank()!=0) return;
  DEBUG_MSG("NLVC_Strain_Post_Processor execute() begin");

  // gather an all owned field here
  Teuchos::RCP<MultiField> disp_x = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::DISPLACEMENT_X_FS);
  Teuchos::RCP<MultiField> disp_y = schema_->mesh()->get_overlap_field(DICe::mesh::field_enums::DISPLACEMENT_Y_FS);

  const int_t step_size = schema_->step_size_x();
  const scalar_t factor = step_size*step_size;
  scalar_t alpha_x = 0.0;
  scalar_t alpha_y = 0.0;
  scalar_t phi_x = 0.0;
  scalar_t phi_y = 0.0;
  scalar_t u_x = 0.0;
  scalar_t u_y = 0.0;
  for(int_t subset=0;subset<data_num_points_;++subset){
    scalar_t dudx = 0.0;
    scalar_t dudy = 0.0;
    scalar_t dvdx = 0.0;
    scalar_t dvdy = 0.0;
    scalar_t integrated_alpha_x = 0.0;
    scalar_t integrated_alpha_y = 0.0;
    scalar_t integrated_phi_x = 0.0;
    scalar_t integrated_phi_y = 0.0;
    DEBUG_MSG("Processing subset " << subset << " with " << neighbor_lists_[subset] << "neighbors");
    for(int_t j=0;j<num_neigh_[subset];++j){
      const int_t neighbor_gid = neighbor_lists_[subset*vec_stride_ + j];
      //assert(neighbor_index >=0);
      //assert(neighbor_index < neighbor_lists_[self_global_id].size());
      //assert(neighbor_distances_mag_[subset][j] <= horizon_/2.0 && "Error, the distance between these two points is greater than the horizon, but somehow they are in each other's neighborhoods.");
      const scalar_t dist_x = neighbor_distances_x_[subset*vec_stride_ + j];
      const scalar_t dist_y = neighbor_distances_y_[subset*vec_stride_ + j];
      u_x = disp_x->local_value(neighbor_gid);
      u_y = disp_y->local_value(neighbor_gid);
      if(dist_x==0){
        phi_x = 4.0*(dist_x + 0.5*horizon_)/(horizon_*horizon_);
        alpha_x = 0.0;
      }
      else if(dist_x < 0){
        phi_x = 4.0*(dist_x + 0.5*horizon_)/(horizon_*horizon_);
        alpha_x = 4.0 / (horizon_*horizon_);
      }
      else{
        phi_x = 4.0*(0.5*horizon_ - dist_x)/(horizon_*horizon_);
        alpha_x = -4.0 / (horizon_*horizon_);
      }
      if(dist_y==0){
        phi_y = 4.0*(dist_y + 0.5*horizon_)/(horizon_*horizon_);
        alpha_y = 0.0;
      }
      else if(dist_y < 0){
        phi_y = 4.0*(dist_y + 0.5*horizon_)/(horizon_*horizon_);
        alpha_y = 4.0 / (horizon_*horizon_);
      }
      else{
        phi_y = 4.0*(0.5*horizon_ - dist_y)/(horizon_*horizon_);
        alpha_y = -4.0 / (horizon_*horizon_);
      }
      integrated_alpha_x += alpha_x*factor;
      integrated_alpha_y += alpha_y*factor;
      integrated_phi_x += phi_x*factor;
      integrated_phi_y += phi_y*factor;
      dudx -= u_x * alpha_x*phi_y * factor;
      dudy -= u_x * alpha_y*phi_x * factor;
      dvdx -= u_y * alpha_x*phi_y * factor;
      dvdy -= u_y * alpha_y*phi_x * factor;
    } // neighbor loop
    field_value(subset,nlvc_dudx) = dudx;
    field_value(subset,nlvc_dudy) = dudy;
    field_value(subset,nlvc_dvdx) = dvdx;
    field_value(subset,nlvc_dvdy) = dvdy;
    field_value(subset,nlvc_integrated_alpha_x) = integrated_alpha_x;
    field_value(subset,nlvc_integrated_alpha_y) = integrated_alpha_y;
    field_value(subset,nlvc_integrated_phi_x) = integrated_phi_x;
    field_value(subset,nlvc_integrated_phi_y) = integrated_phi_y;
    DEBUG_MSG("Subset " << subset << " dudx " << field_value(subset,nlvc_dudx) << " dudy " << field_value(subset,nlvc_dudy) <<
      " dvdx " << field_value(subset,nlvc_dvdx) << " dvdy " << field_value(subset,nlvc_dvdy));
    DEBUG_MSG("Subset " << subset << " integrated_alpha_x " << integrated_alpha_x << " _y " << integrated_alpha_y << " integrated_phi_x " << integrated_phi_x << " _y " << integrated_phi_y);

    // compute the Green-Lagrange strain based on the derivatives computed above:
    const scalar_t GL_xx = 0.5*(2.0*dudx + dudx*dudx + dvdx*dvdx);
    const scalar_t GL_yy = 0.5*(2.0*dvdy + dudy*dudy + dvdy*dvdy);
    const scalar_t GL_xy = 0.5*(dudy + dvdx + dudx*dudy + dvdx*dvdy);
    field_value(subset,nlvc_strain_xx) = GL_xx;
    field_value(subset,nlvc_strain_yy) = GL_yy;
    field_value(subset,nlvc_strain_xy) = GL_xy;
    DEBUG_MSG("Subset " << subset << " NLVC Green-Lagrange strain XX: " << field_value(subset,nlvc_strain_xx) << " YY: " << field_value(subset,nlvc_strain_yy) <<
      " XY: " << field_value(subset,nlvc_strain_xy));

  } // subset loop
  DEBUG_MSG("NLVC_Strain_Post_Processor execute() end");
}

}// End DICe Namespace
