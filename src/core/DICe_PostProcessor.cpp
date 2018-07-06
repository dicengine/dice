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

#include <DICe_PostProcessor.h>

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <fstream>

namespace DICe {

Post_Processor::Post_Processor(const std::string & name) :
  name_(name),
  local_num_points_(0),
  overlap_num_points_(0),
  neighborhood_initialized_(false),
  coords_x_name_(DICe::field_enums::INITIAL_COORDINATES_FS.get_name_label()),
  coords_y_name_(DICe::field_enums::INITIAL_COORDINATES_FS.get_name_label()),
  disp_x_name_(DICe::field_enums::DISPLACEMENT_FS.get_name_label()),
  disp_y_name_(DICe::field_enums::DISPLACEMENT_FS.get_name_label()),
  has_custom_field_names_(false)
{}

void
Post_Processor::initialize(Teuchos::RCP<DICe::mesh::Mesh> & mesh){
  mesh_ = mesh;
  assert(mesh_!=Teuchos::null);
  local_num_points_ = mesh_->get_scalar_node_dist_map()->get_num_local_elements();
  overlap_num_points_ = mesh_->get_scalar_node_overlap_map()->get_num_local_elements();
  assert(local_num_points_>0);
  assert(overlap_num_points_>0);
  for(size_t i=0;i<field_specs_.size();++i)
    mesh_->create_field(field_specs_[i]);
}

void
Post_Processor::set_field_names(const Teuchos::RCP<Teuchos::ParameterList> & params){
  // change the field specs for disp and coords if provided
  if(params->isParameter(coordinates_x_field_name)){
    coords_x_name_ = params->get<std::string>(coordinates_x_field_name);
    if(coords_x_name_!=DICe::field_enums::INITIAL_COORDINATES_FS.get_name_label()&&coords_x_name_!=DICe::field_enums::SUBSET_COORDINATES_X_FS.get_name_label())
      has_custom_field_names_ = true;
  }
  if(params->isParameter(coordinates_y_field_name)){
    coords_y_name_ = params->get<std::string>(coordinates_y_field_name);
    if(coords_y_name_!=DICe::field_enums::INITIAL_COORDINATES_FS.get_name_label()&&coords_y_name_!=DICe::field_enums::SUBSET_COORDINATES_Y_FS.get_name_label())
      has_custom_field_names_ = true;
  }
  if(params->isParameter(displacement_x_field_name)){
    disp_x_name_ = params->get<std::string>(displacement_x_field_name);
    if(disp_x_name_!=DICe::field_enums::DISPLACEMENT_FS.get_name_label()&&disp_x_name_!=DICe::field_enums::SUBSET_DISPLACEMENT_X_FS.get_name_label())
      has_custom_field_names_ = true;
  }
  if(params->isParameter(displacement_y_field_name)){
    disp_y_name_ = params->get<std::string>(displacement_y_field_name);
    if(disp_y_name_!=DICe::field_enums::DISPLACEMENT_FS.get_name_label()&&disp_y_name_!=DICe::field_enums::SUBSET_DISPLACEMENT_Y_FS.get_name_label())
      has_custom_field_names_ = true;
  }
  DEBUG_MSG("Setting the coords x field to " << coords_x_name_ << " for post processor " << name_);
  DEBUG_MSG("Setting the coords y field to " << coords_y_name_ << " for post processor " << name_);
  DEBUG_MSG("Setting the displacement x field to " << disp_x_name_ << " for post processor " << name_);
  DEBUG_MSG("Setting the displacement y field to " << disp_y_name_ << " for post processor " << name_);
}

void
Post_Processor::set_stereo_field_names(){
  if(has_custom_field_names_) return; // no op

  // test to see if the analysis is stereo and therefor the model coordinates are populated
  bool use_model_coordinates = false;
  try{
    mesh_->get_field(DICe::field_enums::MODEL_COORDINATES_X_FS);
    use_model_coordinates = true;
  }catch(std::exception & e){
    use_model_coordinates = false;
  }
  if(use_model_coordinates){
    Teuchos::RCP<DICe::MultiField> model_x_coords = mesh_->get_field(DICe::field_enums::MODEL_COORDINATES_X_FS);
    if(model_x_coords->norm() < 1.0E-8) // also check that the field has values, not just zeros
      use_model_coordinates = false;
  }
  if(!use_model_coordinates) return;  // no op
  coords_x_name_ = DICe::field_enums::MODEL_COORDINATES_X_FS.get_name_label();
  coords_y_name_ = DICe::field_enums::MODEL_COORDINATES_Y_FS.get_name_label();
  disp_x_name_ = DICe::field_enums::MODEL_DISPLACEMENT_X_FS.get_name_label();
  disp_y_name_ = DICe::field_enums::MODEL_DISPLACEMENT_Y_FS.get_name_label();
  DEBUG_MSG("Setting the coords x field to " << coords_x_name_ << " for post processor " << name_);
  DEBUG_MSG("Setting the coords y field to " << coords_y_name_ << " for post processor " << name_);
  DEBUG_MSG("Setting the displacement x field to " << disp_x_name_ << " for post processor " << name_);
  DEBUG_MSG("Setting the displacement y field to " << disp_y_name_ << " for post processor " << name_);
}


void
Post_Processor::initialize_neighborhood(const scalar_t & neighborhood_radius){
  DEBUG_MSG("Post_Processor::initialize_neighborhood(): begin");

  // gather an all owned field here
  const int_t spa_dim = mesh_->spatial_dimension();
  DICe::field_enums::Field_Spec coords_x_spec = mesh_->get_field_spec(coords_x_name_);
  DICe::field_enums::Field_Spec coords_y_spec = mesh_->get_field_spec(coords_y_name_);
  Teuchos::RCP<MultiField> coords;
  if(coords_x_spec.get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE){
    Teuchos::RCP<MultiField> coords_x = mesh_->get_overlap_field(coords_x_spec);
    Teuchos::RCP<MultiField> coords_y = mesh_->get_overlap_field(coords_y_spec);
    Teuchos::RCP<MultiField_Map> overlap_map = mesh_->get_vector_node_overlap_map();
    coords = Teuchos::rcp( new MultiField(overlap_map,1,true));
    for(int_t i=0;i<mesh_->get_scalar_node_overlap_map()->get_num_local_elements();++i){
      coords->local_value(i*spa_dim+0) = coords_x->local_value(i);
      coords->local_value(i*spa_dim+1) = coords_y->local_value(i);
    }
  }else{
    // note assumes that the same vector field spec was given for x and y
    coords = mesh_->get_overlap_field(coords_x_spec);
  }
  // create neighborhood lists using nanoflann:
  DEBUG_MSG("creating the point cloud using nanoflann");
  point_cloud_ = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
  point_cloud_->pts.resize(overlap_num_points_);
  for(int_t i=0;i<overlap_num_points_;++i){
    point_cloud_->pts[i].x = coords->local_value(i*spa_dim+0);
    point_cloud_->pts[i].y = coords->local_value(i*spa_dim+1);
  }
  DEBUG_MSG("building the kd-tree");
  Teuchos::RCP<kd_tree_2d_t> kd_tree = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud_.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree->buildIndex();
  DEBUG_MSG("kd-tree completed");

  // perform a pass to size the neighbor lists
  neighbor_list_.resize(local_num_points_);
  neighbor_dist_x_.resize(local_num_points_);
  neighbor_dist_y_.resize(local_num_points_);

  scalar_t query_pt[2];
  if(neighborhood_radius < 0){ // k-nearest search
    const int_t num_neigh = (int_t)(-1.0*neighborhood_radius);
    DEBUG_MSG("performing k-nearest neighbors search for " << num_neigh << " neighbors");
    std::vector<size_t> ret_index(num_neigh);
    std::vector<scalar_t> out_dist_sqr(num_neigh);
    for(int_t i=0;i<local_num_points_;++i){
      // get the gid of the point
      const int_t gid = mesh_->get_scalar_node_dist_map()->get_global_element(i);
      // get the overlap local id of the point
      const int_t olid = mesh_->get_scalar_node_overlap_map()->get_local_element(gid);
      assert(olid<overlap_num_points_);
      query_pt[0] = point_cloud_->pts[olid].x;
      query_pt[1] = point_cloud_->pts[olid].y;
      kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
      for(int_t j=0;j<num_neigh;++j){
        const int_t neigh_olid = ret_index[j];
        neighbor_list_[i].push_back(neigh_olid);
        neighbor_dist_x_[i].push_back(coords->local_value(neigh_olid*spa_dim+0) - coords->local_value(olid*spa_dim+0));
        neighbor_dist_y_[i].push_back(coords->local_value(neigh_olid*spa_dim+1) - coords->local_value(olid*spa_dim+1));
      }
    }
  }else{ // radius search
    std::vector<std::pair<size_t,scalar_t> > ret_matches;
    nanoflann::SearchParams params;
    params.sorted = true; // sort by distance in ascending order
    const scalar_t tiny = 1.0E-5;
    scalar_t neigh_rad_2 = neighborhood_radius*neighborhood_radius + tiny;
    DEBUG_MSG("performing radius neighbor search with rad^2 " << neigh_rad_2);

    for(int_t i=0;i<local_num_points_;++i){
      // get the gid of the point
      const int_t gid = mesh_->get_scalar_node_dist_map()->get_global_element(i);
      // get the overlap local id of the point
      const int_t olid = mesh_->get_scalar_node_overlap_map()->get_local_element(gid);
      assert(olid<overlap_num_points_);
      query_pt[0] = point_cloud_->pts[olid].x;
      query_pt[1] = point_cloud_->pts[olid].y;
      kd_tree->radiusSearch(&query_pt[0],neigh_rad_2,ret_matches,params);
      for(size_t j=0;j<ret_matches.size();++j){
        const int_t neigh_olid = ret_matches[j].first;
        neighbor_list_[i].push_back(neigh_olid);
        neighbor_dist_x_[i].push_back(coords->local_value(neigh_olid*spa_dim+0) - coords->local_value(olid*spa_dim+0));
        neighbor_dist_y_[i].push_back(coords->local_value(neigh_olid*spa_dim+1) - coords->local_value(olid*spa_dim+1));
      }
    }
  }

  neighborhood_initialized_ = true;
  DEBUG_MSG("Post_Processor::initialize_neighborhood(): end");
}

VSG_Strain_Post_Processor::VSG_Strain_Post_Processor(const Teuchos::RCP<Teuchos::ParameterList> & params) :
  Post_Processor(post_process_vsg_strain){
  field_specs_.push_back(DICe::field_enums::VSG_STRAIN_XX_FS);
  field_specs_.push_back(DICe::field_enums::VSG_STRAIN_YY_FS);
  field_specs_.push_back(DICe::field_enums::VSG_STRAIN_XY_FS);
  field_specs_.push_back(DICe::field_enums::VSG_DUDX_FS);
  field_specs_.push_back(DICe::field_enums::VSG_DUDY_FS);
  field_specs_.push_back(DICe::field_enums::VSG_DVDX_FS);
  field_specs_.push_back(DICe::field_enums::VSG_DVDY_FS);
  DEBUG_MSG("Enabling post processor VSG_Strain_Post_Processor with associated fields:");
  for(size_t i=0;i<field_specs_.size();++i){
    DEBUG_MSG(field_specs_[i].get_name_label());
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
  set_field_names(params);
}

void
VSG_Strain_Post_Processor::pre_execution_tasks(){
  DEBUG_MSG("VSG_Strain_Post_Processor pre_execution_tasks() begin");
  const scalar_t neigh_rad = (scalar_t)window_size_/2.0;
  initialize_neighborhood(neigh_rad);
  TEUCHOS_TEST_FOR_EXCEPTION(!neighborhood_initialized_,std::runtime_error,"Error, neighborhoods should be initialized here.");
  DEBUG_MSG("VSG_Strain_Post_Processor pre_execution_tasks() end");
  set_stereo_field_names();
  DICe::field_enums::Field_Spec disp_x_spec = mesh_->get_field_spec(disp_x_name_);
  DICe::field_enums::Field_Spec disp_y_spec = mesh_->get_field_spec(disp_y_name_);
  DICe::field_enums::Field_Spec coords_x_spec = mesh_->get_field_spec(coords_x_name_);
  DICe::field_enums::Field_Spec coords_y_spec = mesh_->get_field_spec(coords_y_name_);
  TEUCHOS_TEST_FOR_EXCEPTION(disp_x_spec.get_field_type()!=disp_y_spec.get_field_type(),
    std::runtime_error,"Error: invalid field selections");
  TEUCHOS_TEST_FOR_EXCEPTION(disp_x_spec.get_rank()!=disp_y_spec.get_rank(),
    std::runtime_error,"Error: invalid field selections");
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x_spec.get_field_type()!=coords_y_spec.get_field_type(),
    std::runtime_error,"Error: invalid field selections");
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x_spec.get_rank()!=coords_y_spec.get_rank(),
    std::runtime_error,"Error: invalid field selections");
}

void
VSG_Strain_Post_Processor::execute(){
  DEBUG_MSG("VSG_Strain_Post_Processor execute() begin");
  if(!neighborhood_initialized_) pre_execution_tasks();

  // gather an all owned fields here
  DICe::field_enums::Field_Spec disp_x_spec = mesh_->get_field_spec(disp_x_name_);
  DICe::field_enums::Field_Spec disp_y_spec = mesh_->get_field_spec(disp_y_name_);
  const int_t spa_dim = mesh_->spatial_dimension();
  Teuchos::RCP<MultiField> disp;
  if(disp_x_spec.get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE){
    Teuchos::RCP<MultiField> disp_x = mesh_->get_overlap_field(disp_x_spec);
    Teuchos::RCP<MultiField> disp_y = mesh_->get_overlap_field(disp_y_spec);
    Teuchos::RCP<MultiField_Map> overlap_map = mesh_->get_vector_node_overlap_map();
    disp = Teuchos::rcp( new MultiField(overlap_map,1,true));
    for(int_t i=0;i<mesh_->get_scalar_node_overlap_map()->get_num_local_elements();++i){
      disp->local_value(i*spa_dim+0) = disp_x->local_value(i);
      disp->local_value(i*spa_dim+1) = disp_y->local_value(i);
    }
  }else{
    // note assumes that the same vector field spec was given for x and y
    disp = mesh_->get_overlap_field(disp_x_spec);
  }
  // get the sigma field:
  Teuchos::RCP<MultiField> sigma = mesh_->get_overlap_field(DICe::field_enums::SIGMA_FS);
  // get pointers to the local fields
  Teuchos::RCP<DICe::MultiField> vsg_strain_xx_rcp = mesh_->get_field(DICe::field_enums::VSG_STRAIN_XX_FS);
  Teuchos::RCP<DICe::MultiField> vsg_strain_yy_rcp = mesh_->get_field(DICe::field_enums::VSG_STRAIN_YY_FS);
  Teuchos::RCP<DICe::MultiField> vsg_strain_xy_rcp = mesh_->get_field(DICe::field_enums::VSG_STRAIN_XY_FS);
  Teuchos::RCP<DICe::MultiField> vsg_dudx_rcp = mesh_->get_field(DICe::field_enums::VSG_DUDX_FS);
  Teuchos::RCP<DICe::MultiField> vsg_dudy_rcp = mesh_->get_field(DICe::field_enums::VSG_DUDY_FS);
  Teuchos::RCP<DICe::MultiField> vsg_dvdx_rcp = mesh_->get_field(DICe::field_enums::VSG_DVDX_FS);
  Teuchos::RCP<DICe::MultiField> vsg_dvdy_rcp = mesh_->get_field(DICe::field_enums::VSG_DVDY_FS);
  Teuchos::RCP<DICe::MultiField> match = mesh_->get_field(DICe::field_enums::MATCH_FS);

  const int_t N = 3;
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  double *GWORK = new double[10*N];
  int *IWORK = new int[LWORK];
  // Note, LAPACK does not allow templating on long int or scalar_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;

  int_t num_neigh = 0;
  std::vector<bool> neigh_valid;
  int_t neigh_id = 0;
  for(int_t subset=0;subset<local_num_points_;++subset){
    DEBUG_MSG("Processing subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << ", " << subset + 1 << " of " << local_num_points_);
    // search the neighbors to see how many valid neighbors exist:
    num_neigh = neighbor_list_[subset].size();
    neigh_valid.resize(num_neigh);
    int_t num_valid_neigh = 0;
    for(int_t j=0;j<num_neigh;++j){
      if(sigma->local_value(neighbor_list_[subset][j])>=0.0){
        neigh_valid[j] = true;
        num_valid_neigh++;
      }else{
        neigh_valid[j] = false;
      }
    }
    DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " num valid neighbors: " << num_valid_neigh);
    if(num_valid_neigh < 3 || sigma->local_value(neighbor_list_[subset][0]) < 0.0){
      vsg_dudx_rcp->local_value(subset) = 0.0;
      vsg_dudy_rcp->local_value(subset) = 0.0;
      vsg_dvdx_rcp->local_value(subset) = 0.0;
      vsg_dvdy_rcp->local_value(subset) = 0.0;
      vsg_strain_xx_rcp->local_value(subset) = 0.0;
      vsg_strain_yy_rcp->local_value(subset) = 0.0;
      vsg_strain_xy_rcp->local_value(subset) = 0.0;
      DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " failed subset (sigma=-1) or not enough neighbors to calculate VSG strain."
          " Setting all strain values to zero.");
      match->local_value(subset) = -1;
    }else{
      Teuchos::ArrayRCP<double> u_x(num_valid_neigh,0.0);
      Teuchos::ArrayRCP<double> u_y(num_valid_neigh,0.0);
      Teuchos::ArrayRCP<double> X_t_u_x(N,0.0);
      Teuchos::ArrayRCP<double> X_t_u_y(N,0.0);
      Teuchos::ArrayRCP<double> coeffs_x(N,0.0);
      Teuchos::ArrayRCP<double> coeffs_y(N,0.0);
      Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_valid_neigh, true);
      Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);

      // gather the displacements of the neighbors
      int_t valid_id = 0;
      for(int_t j=0;j<num_neigh;++j){
        if(!neigh_valid[j])continue;
        neigh_id = neighbor_list_[subset][j];
        assert(sigma->local_value(neigh_id)>=0.0);
        u_x[valid_id] = disp->local_value(neigh_id*spa_dim+0);
        u_y[valid_id] = disp->local_value(neigh_id*spa_dim+1);
        // set up the X^T matrix
        X_t(0,valid_id) = 1.0;
        X_t(1,valid_id) = neighbor_dist_x_[subset][j];
        X_t(2,valid_id) = neighbor_dist_y_[subset][j];
        valid_id++;
      }

      // set up X^T*X
      for(int_t k=0;k<N;++k){
        for(int_t m=0;m<N;++m){
          for(int_t j=0;j<num_valid_neigh;++j){
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
          vsg_dudx_rcp->local_value(subset) = 0.0;
          vsg_dudy_rcp->local_value(subset) = 0.0;
          vsg_dvdx_rcp->local_value(subset) = 0.0;
          vsg_dvdy_rcp->local_value(subset) = 0.0;
          vsg_strain_xx_rcp->local_value(subset) = 0.0;
          vsg_strain_yy_rcp->local_value(subset) = 0.0;
          vsg_strain_xy_rcp->local_value(subset) = 0.0;
          DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " failed subset, the pseudo-inverse of the VSG strain calculation is (or is near) singular."
              " Setting all strain values to zero.");
          match->local_value(subset) = -1;
          continue;
          //std::cout << "Error: The pseudo-inverse of the VSG strain calculation is (or is near) singular." << std::endl;
          //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
        }
      }
      catch(std::exception &e){
        if(&e!=0){
          DEBUG_MSG( e.what() << '\n');
        }
        std::cout << "Error: Something went wrong in the condition number calculation" << std::endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"");
      }
      try
      {
        lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),IPIV,WORK,LWORK,&INFO);
      }
      catch(std::exception &e){
        if(&e!=0){
          DEBUG_MSG( e.what() << '\n');
        }
        std::cout << "Error: Something went wrong in the inverse calculation of X^T*X " << std::endl;
      }

      // compute X^T*u
      for(int_t i=0;i<N;++i){
        for(int_t j=0;j<num_valid_neigh;++j){
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
      vsg_dudx_rcp->local_value(subset) = dudx;
      vsg_dudy_rcp->local_value(subset) = dudy;
      vsg_dvdx_rcp->local_value(subset) = dvdx;
      vsg_dvdy_rcp->local_value(subset) = dvdy;

      DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " dudx " << dudx << " dudy " << dudy <<
        " dvdx " << dvdx << " dvdy " << dvdy);

      // compute the Green-Lagrange strain based on the derivatives computed above:
      const scalar_t GL_xx = 0.5*(2.0*dudx + dudx*dudx + dvdx*dvdx);
      const scalar_t GL_yy = 0.5*(2.0*dvdy + dudy*dudy + dvdy*dvdy);
      const scalar_t GL_xy = 0.5*(dudy + dvdx + dudx*dudy + dvdx*dvdy);
      vsg_strain_xx_rcp->local_value(subset) = GL_xx;
      vsg_strain_yy_rcp->local_value(subset) = GL_yy;
      vsg_strain_xy_rcp->local_value(subset) = GL_xy;

      DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " VSG Green-Lagrange strain XX: " << GL_xx << " YY: " << GL_yy <<
        " XY: " << GL_xy);
    } // end has enough valid neighbors to compute strain check

  } // end subset loop

  delete [] WORK;
  delete [] GWORK;
  delete [] IWORK;
  delete [] IPIV;
  DEBUG_MSG("VSG_Strain_Post_Processor execute() end");
}

NLVC_Strain_Post_Processor::NLVC_Strain_Post_Processor(const Teuchos::RCP<Teuchos::ParameterList> & params) :
  Post_Processor(post_process_nlvc_strain)
{
  field_specs_.push_back(DICe::field_enums::NLVC_STRAIN_XX_FS);
  field_specs_.push_back(DICe::field_enums::NLVC_STRAIN_YY_FS);
  field_specs_.push_back(DICe::field_enums::NLVC_STRAIN_XY_FS);
  field_specs_.push_back(DICe::field_enums::NLVC_DUDX_FS);
  field_specs_.push_back(DICe::field_enums::NLVC_DUDY_FS);
  field_specs_.push_back(DICe::field_enums::NLVC_DVDX_FS);
  field_specs_.push_back(DICe::field_enums::NLVC_DVDY_FS);
  DEBUG_MSG("Enabling post processor NLVC_Strain_Post_Processor with associated fields:");
  for(size_t i=0;i<field_specs_.size();++i){
    DEBUG_MSG(field_specs_[i].get_name_label());
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
  set_field_names(params);
}

void
NLVC_Strain_Post_Processor::pre_execution_tasks(){
  DEBUG_MSG("NLVC_Strain_Post_Processor pre_execution_tasks() begin");
  const scalar_t neigh_rad = (scalar_t)horizon_/2.0;
  initialize_neighborhood(neigh_rad);
  TEUCHOS_TEST_FOR_EXCEPTION(!neighborhood_initialized_,std::runtime_error,"Error, neighborhoods should be initialized here.");
  DEBUG_MSG("NLVC_Strain_Post_Processor pre_execution_tasks() end");
  set_stereo_field_names();
  DICe::field_enums::Field_Spec disp_x_spec = mesh_->get_field_spec(disp_x_name_);
  DICe::field_enums::Field_Spec disp_y_spec = mesh_->get_field_spec(disp_y_name_);
  DICe::field_enums::Field_Spec coords_x_spec = mesh_->get_field_spec(coords_x_name_);
  DICe::field_enums::Field_Spec coords_y_spec = mesh_->get_field_spec(coords_y_name_);
  TEUCHOS_TEST_FOR_EXCEPTION(disp_x_spec.get_field_type()!=disp_y_spec.get_field_type(),
    std::runtime_error,"Error: invalid field selections");
  TEUCHOS_TEST_FOR_EXCEPTION(disp_x_spec.get_rank()!=disp_y_spec.get_rank(),
    std::runtime_error,"Error: invalid field selections");
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x_spec.get_field_type()!=coords_y_spec.get_field_type(),
    std::runtime_error,"Error: invalid field selections");
  TEUCHOS_TEST_FOR_EXCEPTION(coords_x_spec.get_rank()!=coords_y_spec.get_rank(),
    std::runtime_error,"Error: invalid field selections");
}

void
NLVC_Strain_Post_Processor::compute_kernel(const scalar_t & dx,
  const scalar_t & dy,
  scalar_t & kx,
  scalar_t & ky){
  static scalar_t h = (scalar_t)horizon_*0.5;
  static scalar_t s = h / 3.0;
  const scalar_t r = std::sqrt(dx*dx+dy*dy);
  kx = s==0.0?0.0:1.0/(2*DICE_PI*s*s)*(-2*dx/(2*s*s))*exp(-(r*r/(2*s*s)));
  ky = s==0.0?0.0:1.0/(2*DICE_PI*s*s)*(-2*dy/(2*s*s))*exp(-(r*r/(2*s*s)));
}


void
NLVC_Strain_Post_Processor::execute(){
  DEBUG_MSG("NLVC_Strain_Post_Processor execute() begin");
  if(!neighborhood_initialized_) pre_execution_tasks();

  // gather an all owned field here
  DICe::field_enums::Field_Spec disp_x_spec = mesh_->get_field_spec(disp_x_name_);
  DICe::field_enums::Field_Spec disp_y_spec = mesh_->get_field_spec(disp_y_name_);
  const int_t spa_dim = mesh_->spatial_dimension();
  Teuchos::RCP<MultiField> disp;
  if(disp_x_spec.get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE){
    Teuchos::RCP<MultiField> disp_x = mesh_->get_overlap_field(disp_x_spec);
    Teuchos::RCP<MultiField> disp_y = mesh_->get_overlap_field(disp_y_spec);
    Teuchos::RCP<MultiField_Map> overlap_map = mesh_->get_vector_node_overlap_map();
    disp = Teuchos::rcp( new MultiField(overlap_map,1,true));
    for(int_t i=0;i<mesh_->get_scalar_node_overlap_map()->get_num_local_elements();++i){
      disp->local_value(i*spa_dim+0) = disp_x->local_value(i);
      disp->local_value(i*spa_dim+1) = disp_y->local_value(i);
    }
  }else{
    // note assumes that the same vector field spec was given for x and y
    disp = mesh_->get_overlap_field(disp_x_spec);
  }
  // get the sigma field:
  Teuchos::RCP<MultiField> sigma = mesh_->get_overlap_field(DICe::field_enums::SIGMA_FS);
  // get pointers to the local fields
  Teuchos::RCP<DICe::MultiField> nlvc_strain_xx_rcp = mesh_->get_field(DICe::field_enums::NLVC_STRAIN_XX_FS);
  Teuchos::RCP<DICe::MultiField> nlvc_strain_yy_rcp = mesh_->get_field(DICe::field_enums::NLVC_STRAIN_YY_FS);
  Teuchos::RCP<DICe::MultiField> nlvc_strain_xy_rcp = mesh_->get_field(DICe::field_enums::NLVC_STRAIN_XY_FS);
  Teuchos::RCP<DICe::MultiField> nlvc_dudx_rcp = mesh_->get_field(DICe::field_enums::NLVC_DUDX_FS);
  Teuchos::RCP<DICe::MultiField> nlvc_dudy_rcp = mesh_->get_field(DICe::field_enums::NLVC_DUDY_FS);
  Teuchos::RCP<DICe::MultiField> nlvc_dvdx_rcp = mesh_->get_field(DICe::field_enums::NLVC_DVDX_FS);
  Teuchos::RCP<DICe::MultiField> nlvc_dvdy_rcp = mesh_->get_field(DICe::field_enums::NLVC_DVDY_FS);
  Teuchos::RCP<DICe::MultiField> f9_rcp = mesh_->get_field(DICe::field_enums::FIELD_9_FS);
  Teuchos::RCP<DICe::MultiField> f10_rcp = mesh_->get_field(DICe::field_enums::FIELD_10_FS);
  Teuchos::RCP<DICe::MultiField> match = mesh_->get_field(DICe::field_enums::MATCH_FS);

  std::vector<bool> neigh_valid;
  for(int_t subset=0;subset<local_num_points_;++subset){
    DEBUG_MSG("Processing subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << ", " << subset + 1 << " of " << local_num_points_);
    scalar_t dudx = 0.0;
    scalar_t dudy = 0.0;
    scalar_t dvdx = 0.0;
    scalar_t dvdy = 0.0;
    scalar_t ux = 0.0;
    scalar_t uy = 0.0;
    scalar_t dx = 0.0;
    scalar_t dy = 0.0;
    scalar_t sum_int_x = 0.0;
    scalar_t sum_int_y = 0.0;
    scalar_t kx = 0.0;
    scalar_t ky = 0.0;
    int_t neigh_id = 0;
    const int_t num_neigh = neighbor_dist_x_[subset].size();
    neigh_valid.resize(num_neigh);
    int_t num_valid_neigh = 0;
    for(int_t j=0;j<num_neigh;++j){
      if(sigma->local_value(neighbor_list_[subset][j])>=0.0){
        neigh_valid[j] = true;
        num_valid_neigh++;
      }else{
        neigh_valid[j] = false;
      }
    }
    DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " num valid neighbors: " << num_valid_neigh);
    if(num_valid_neigh < 3 || sigma->local_value(neighbor_list_[subset][0])<0.0){
      nlvc_dudx_rcp->local_value(subset) = 0.0;
      nlvc_dudy_rcp->local_value(subset) = 0.0;
      nlvc_dvdx_rcp->local_value(subset) = 0.0;
      nlvc_dvdy_rcp->local_value(subset) = 0.0;
      nlvc_strain_xx_rcp->local_value(subset) = 0.0;
      nlvc_strain_yy_rcp->local_value(subset) = 0.0;
      nlvc_strain_xy_rcp->local_value(subset) = 0.0;
      DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " failed subset (sigma=-1) or not enough neighbors to calculate NLVC strain."
          " Setting all strain values to zero.");
      match->local_value(subset) = -1;
    }else{
      assert(neighbor_dist_x_[subset].size()>1);
      assert(neighbor_dist_y_[subset].size()>1);
      // neighbor 0 is yourself
      const scalar_t nearest_neigh_dist = std::sqrt(neighbor_dist_x_[subset][1]*neighbor_dist_x_[subset][1] +
        neighbor_dist_y_[subset][1]*neighbor_dist_y_[subset][1]);
      const scalar_t patch_area = nearest_neigh_dist*nearest_neigh_dist;
      for(int_t j=0;j<num_neigh;++j){
        if(!neigh_valid[j]) continue;
        neigh_id = neighbor_list_[subset][j];
        assert(sigma->local_value(neigh_id)>=0.0);
        ux = disp->local_value(neigh_id*spa_dim+0);
        uy = disp->local_value(neigh_id*spa_dim+1);
        dx = neighbor_dist_x_[subset][j];
        dy = neighbor_dist_y_[subset][j];
        compute_kernel(dx,dy,kx,ky);
        sum_int_x += kx*patch_area;
        sum_int_y += ky*patch_area;
        dudx -= ux * kx*patch_area;
        dudy -= ux * ky*patch_area;
        dvdx -= uy * kx*patch_area;
        dvdy -= uy * ky*patch_area;
      } // neighbor loop
      nlvc_dudx_rcp->local_value(subset) = dudx;
      nlvc_dudy_rcp->local_value(subset) = dudy;
      nlvc_dvdx_rcp->local_value(subset) = dvdx;
      nlvc_dvdy_rcp->local_value(subset) = dvdy;
      f9_rcp->local_value(subset) = sum_int_x;
      f10_rcp->local_value(subset) = sum_int_y;

      DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " dudx " << dudx << " dudy " << dudy <<
        " dvdx " << dvdx << " dvdy " << dvdy);
      DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " sum_int_x " << sum_int_x <<
        " sum_int_y " << sum_int_y);

      // compute the Green-Lagrange strain based on the derivatives computed above:
      const scalar_t GL_xx = 0.5*(2.0*dudx + dudx*dudx + dvdx*dvdx);
      const scalar_t GL_yy = 0.5*(2.0*dvdy + dudy*dudy + dvdy*dvdy);
      const scalar_t GL_xy = 0.5*(dudy + dvdx + dudx*dudy + dvdx*dvdy);
      nlvc_strain_xx_rcp->local_value(subset) = GL_xx;
      nlvc_strain_yy_rcp->local_value(subset) = GL_yy;
      nlvc_strain_xy_rcp->local_value(subset) = GL_xy;
      if(sum_int_x > 0.01 || sum_int_y > 0.01 || sum_int_x < -0.01 || sum_int_y < -0.01){
        match->local_value(subset) = -1;
      }
      DEBUG_MSG("Subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << " NLVC Green-Lagrange strain XX: " << GL_xx << " YY: " << GL_yy <<
        " XY: " << GL_xy);
    }
  } // subset loop

  DEBUG_MSG("NLVC_Strain_Post_Processor execute() end");
}

Altitude_Post_Processor::Altitude_Post_Processor(const Teuchos::RCP<Teuchos::ParameterList> & params) :
  Post_Processor(post_process_altitude),
  radius_of_earth_(6371000.0),
  apogee_(35800000.0),
  ground_level_initialized_(false)
{
  field_specs_.push_back(DICe::field_enums::ALTITUDE_FS);
  field_specs_.push_back(DICe::field_enums::ALTITUDE_ABOVE_GROUND_FS);
  field_specs_.push_back(DICe::field_enums::GROUND_LEVEL_FS);
  DEBUG_MSG("Enabling post processor Altitude_Post_Processor with associated fields:");
  for(size_t i=0;i<field_specs_.size();++i){
    DEBUG_MSG(field_specs_[i].get_name_label());
  }
  set_params(params);
}

void
Altitude_Post_Processor::execute(){
  DEBUG_MSG("Altitude_Post_Processor::execute(): begin");

  Teuchos::RCP<DICe::MultiField> ground_level_rcp = mesh_->get_field(DICe::field_enums::GROUND_LEVEL_FS);
  // if this is the first time called, check for an elevations file and interpolate the ground level from that:
  // Note: all processors read the elevations file
  if(!ground_level_initialized_){
    // see if the file exists
    std::ifstream elev_file("elevation_data.txt", std::ios_base::in);
    if(elev_file.good()){
      // parse the file to read in the elevations
      DEBUG_MSG("Altitude_Post_Processor::execute(): found elevations file: elevation_data.txt");
      std::vector<scalar_t> xs;
      std::vector<scalar_t> ys;
      std::vector<scalar_t> elevs;
      int_t line = 0;
      while(!elev_file.eof()){
        std::vector<std::string> tokens = tokenize_line(elev_file);
        if(tokens.size()==0) continue;
        TEUCHOS_TEST_FOR_EXCEPTION(tokens.size()!=5,std::runtime_error,"Error, invalid number of tokens for line " << line <<
          " of file elevation_data.txt. Has " << tokens.size() << " tokens, but should have 5");
        scalar_t elev = strtod(tokens[2].c_str(),NULL);
        if(elev < 0.0) elev = 0.0; // prevent negative ground heights
        elevs.push_back(elev + radius_of_earth_);
        xs.push_back(strtod(tokens[3].c_str(),NULL));
        ys.push_back(strtod(tokens[4].c_str(),NULL));
        line++;
      }
      const int_t num_elev_pts = xs.size();
      DEBUG_MSG("Altitude_Post_Processor::execute(): found " << num_elev_pts << " points in elevation_data.txt");
      // create a point cloud
      TEUCHOS_TEST_FOR_EXCEPTION(neighborhood_initialized_,std::runtime_error,"");
      DEBUG_MSG("creating the point cloud using nanoflann");
      point_cloud_ = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
      point_cloud_->pts.resize(num_elev_pts);
      for(int_t i=0;i<num_elev_pts;++i){
        point_cloud_->pts[i].x = xs[i];
        point_cloud_->pts[i].y = ys[i];
      }
      DEBUG_MSG("building the kd-tree");
      Teuchos::RCP<kd_tree_2d_t> kd_tree = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud_.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
      kd_tree->buildIndex();
      DEBUG_MSG("kd-tree completed");

      // compute the 5 nearest neighbors for each subset and interpolate the ground height from them.
      Teuchos::RCP<MultiField> subset_coords_x = mesh_->get_field(DICe::field_enums::SUBSET_COORDINATES_X_FS);
      Teuchos::RCP<MultiField> subset_coords_y = mesh_->get_field(DICe::field_enums::SUBSET_COORDINATES_Y_FS);
      const int_t num_neigh = 5;
      scalar_t query_pt[2];
      std::vector<size_t> ret_index(num_neigh);
      std::vector<scalar_t> out_dist_sqr(num_neigh);

      const int_t N = 3;
      int *IPIV = new int[N+1];
      int LWORK = N*N;
      int INFO = 0;
      double *WORK = new double[LWORK];
      double *GWORK = new double[10*N];
      int *IWORK = new int[LWORK];
      // Note, LAPACK does not allow templating on long int or scalar_t...must use int and double
      Teuchos::LAPACK<int,double> lapack;

      for(int_t subset=0;subset<local_num_points_;++subset){
        Teuchos::ArrayRCP<double> neigh_elevs(num_neigh,0.0);
        Teuchos::ArrayRCP<double> X_t_u_x(N,0.0);
        Teuchos::ArrayRCP<double> coeffs(N,0.0);
        Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_neigh, true);
        Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);

        query_pt[0] = subset_coords_x->local_value(subset);
        query_pt[1] = subset_coords_y->local_value(subset);
        kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
        for(size_t i=0;i<num_neigh;++i){
          const int_t neigh_id = ret_index[i];
          neigh_elevs[i] = elevs[neigh_id];
          // set up the X^T matrix
          X_t(0,i) = 1.0;
          X_t(1,i) = xs[neigh_id] - query_pt[0];
          X_t(2,i) = ys[neigh_id] - query_pt[1];
        }

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
        for(int_t i=0;i<N;++i){
          for(int_t j=0;j<num_neigh;++j){
            X_t_u_x[i] += X_t(i,j)*neigh_elevs[j];
          }
        }
        // compute the coeffs
        for(int_t i=0;i<N;++i){
          for(int_t j=0;j<N;++j){
            coeffs[i] += X_t_X(i,j)*X_t_u_x[j];
          }
        }
        ground_level_rcp->local_value(subset) = coeffs[0];
      } // end local num points
      delete [] WORK;
      delete [] GWORK;
      delete [] IWORK;
      delete [] IPIV;
    } // end has elevations file
    else{
      DEBUG_MSG("Altitude_Post_Processor::execute(): elevations file: elevation_data.txt not found, using radius of the earth as ground level");
      for(int_t subset=0;subset<local_num_points_;++subset){
        ground_level_rcp->local_value(subset) = radius_of_earth_;
      }
    }
    ground_level_initialized_ = true;
  } // end !ground level initialized

  // gather the X Y and Z model coordinates:
  Teuchos::RCP<DICe::MultiField> X_rcp = mesh_->get_field(DICe::field_enums::MODEL_COORDINATES_X_FS);
  Teuchos::RCP<DICe::MultiField> Y_rcp = mesh_->get_field(DICe::field_enums::MODEL_COORDINATES_Y_FS);
  Teuchos::RCP<DICe::MultiField> Z_rcp = mesh_->get_field(DICe::field_enums::MODEL_COORDINATES_Z_FS);
  Teuchos::RCP<DICe::MultiField> altitude_rcp = mesh_->get_field(DICe::field_enums::ALTITUDE_FS);
  Teuchos::RCP<DICe::MultiField> altitude_above_ground_rcp = mesh_->get_field(DICe::field_enums::ALTITUDE_ABOVE_GROUND_FS);
  TEUCHOS_TEST_FOR_EXCEPTION(altitude_rcp==Teuchos::null || altitude_above_ground_rcp==Teuchos::null, std::runtime_error,"");
  for(int_t subset=0;subset<local_num_points_;++subset){
    DEBUG_MSG("Processing altitude subset gid " << mesh_->get_scalar_node_dist_map()->get_global_element(subset) << ", " << subset + 1 << " of " << local_num_points_);
    // at this point, X Y and Z are in terms of camera 0, convert back to center of the earth coords
    scalar_t X = X_rcp->local_value(subset);
    scalar_t Y = Y_rcp->local_value(subset);
    scalar_t Z = Z_rcp->local_value(subset) - apogee_;
    // convert X Y Z to raius
    altitude_rcp->local_value(subset) = std::sqrt(X*X + Y*Y + Z*Z);
    altitude_above_ground_rcp->local_value(subset) = altitude_rcp->local_value(subset) - ground_level_rcp->local_value(subset);
  }
  DEBUG_MSG("Altitude_Post_Processor::execute(): end");
}

Uncertainty_Post_Processor::Uncertainty_Post_Processor(const Teuchos::RCP<Teuchos::ParameterList> & params) :
  Post_Processor(post_process_uncertainty)
{
  field_specs_.push_back(DICe::field_enums::UNCERTAINTY_FS);
  field_specs_.push_back(DICe::field_enums::UNCERTAINTY_ANGLE_FS);
  DEBUG_MSG("Enabling post processor Uncertainty_Post_Processor with associated fields:");
  for(size_t i=0;i<field_specs_.size();++i){
    DEBUG_MSG(field_specs_[i].get_name_label());
  }
  set_params(params);
}

void
Uncertainty_Post_Processor::execute(){
  DEBUG_MSG("Uncertainty_Post_Processor::execute(): begin");

  Teuchos::RCP<DICe::MultiField> sigma_rcp = mesh_->get_field(DICe::field_enums::SIGMA_FS);
  // cosine of the angle goes into field_1 by convention (See DICe_ObjectiveZNSSD.cpp)
  Teuchos::RCP<DICe::MultiField> field1_rcp = mesh_->get_field(DICe::field_enums::FIELD_1_FS);
  Teuchos::RCP<DICe::MultiField> uncertainty_rcp = mesh_->get_field(DICe::field_enums::UNCERTAINTY_FS);
  Teuchos::RCP<DICe::MultiField> uncertainty_angle_rcp = mesh_->get_field(DICe::field_enums::UNCERTAINTY_ANGLE_FS);
  Teuchos::RCP<DICe::MultiField> noise_rcp = mesh_->get_field(DICe::field_enums::NOISE_LEVEL_FS);
  Teuchos::RCP<DICe::MultiField> max_m_rcp = mesh_->get_field(DICe::field_enums::STEREO_M_MAX_FS);
  TEUCHOS_TEST_FOR_EXCEPTION(uncertainty_rcp==Teuchos::null || uncertainty_angle_rcp==Teuchos::null, std::runtime_error,"");
  for(int_t subset=0;subset<local_num_points_;++subset){
    const scalar_t angle = field1_rcp->local_value(subset);
    const scalar_t sig = sigma_rcp->local_value(subset);
    uncertainty_angle_rcp->local_value(subset) = field1_rcp->local_value(subset);
    if(sig < 0.0){ // filter failed subsets
      uncertainty_rcp->local_value(subset) = 0.0;
      continue;
    }
    // relying on the max-m field to be zero for 2D and have a non-zero value for stereo
    scalar_t max_m  = max_m_rcp->local_value(subset);
    if(max_m > 0.0){
      scalar_t noise_level = noise_rcp->local_value(subset);
      uncertainty_rcp->local_value(subset) = max_m==0.0?0.0:std::sqrt(2.0*noise_level*noise_level/max_m);
    }
    else{
      uncertainty_rcp->local_value(subset) = angle == 0.0 ? 0.0 : 1.0 / angle * sig;
    }
  }
  DEBUG_MSG("Uncertainty_Post_Processor::execute(): end");
}

Live_Plot_Post_Processor::Live_Plot_Post_Processor(const Teuchos::RCP<Teuchos::ParameterList> & params) :
  Post_Processor(post_process_live_plots),
  num_neigh_(7),
  num_individual_pts_(0),
  frame_index_(0),
  num_field_entries_(0){
  set_field_names(params);
}

void
Live_Plot_Post_Processor::pre_execution_tasks(){
  DEBUG_MSG("Live_Plot_Post_Processor::pre_execution_tasks() begin");

  // list of fields to output in the live plots
  // check if stereo or 2D
  bool has_model_coordinates = false;
  if(mesh_->has_field(DICe::field_enums::MODEL_COORDINATES_X)){
    Teuchos::RCP<DICe::MultiField> model_x_coords = mesh_->get_field(DICe::field_enums::MODEL_COORDINATES_X_FS);
    if(model_x_coords->norm() > 1.0E-8) // check that the field has values, not just zeros
      has_model_coordinates = true;
  }
  if(has_model_coordinates){
    field_specs_.push_back(DICe::field_enums::MODEL_DISPLACEMENT_X_FS);
    field_specs_.push_back(DICe::field_enums::MODEL_DISPLACEMENT_Y_FS);
    field_specs_.push_back(DICe::field_enums::MODEL_DISPLACEMENT_Z_FS);
  }else if(mesh_->has_field(DICe::field_enums::SUBSET_DISPLACEMENT_X)){
    field_specs_.push_back(DICe::field_enums::SUBSET_DISPLACEMENT_X_FS);
    field_specs_.push_back(DICe::field_enums::SUBSET_DISPLACEMENT_Y_FS);
  }else if(mesh_->has_field(DICe::field_enums::DISPLACEMENT)){
    field_specs_.push_back(DICe::field_enums::DISPLACEMENT_FS);
  }
  if(mesh_->has_field(DICe::field_enums::ROTATION_Z)){
      field_specs_.push_back(DICe::field_enums::ROTATION_Z_FS);
  }
  if(mesh_->has_field(DICe::field_enums::SIGMA)){
      field_specs_.push_back(DICe::field_enums::SIGMA_FS);
  }
  if(mesh_->has_field(DICe::field_enums::VSG_STRAIN_XX)){
    field_specs_.push_back(DICe::field_enums::VSG_STRAIN_XX_FS);
    field_specs_.push_back(DICe::field_enums::VSG_STRAIN_XY_FS);
    field_specs_.push_back(DICe::field_enums::VSG_STRAIN_YY_FS);
  }
  if(mesh_->has_field(DICe::field_enums::GREEN_LAGRANGE_STRAIN_XX)){
    field_specs_.push_back(DICe::field_enums::GREEN_LAGRANGE_STRAIN_XX_FS);
    field_specs_.push_back(DICe::field_enums::GREEN_LAGRANGE_STRAIN_XY_FS);
    field_specs_.push_back(DICe::field_enums::GREEN_LAGRANGE_STRAIN_YY_FS);
  }

  std::fstream livePlotDataFile("live_plot.dat", std::ios_base::in);
  TEUCHOS_TEST_FOR_EXCEPTION(!livePlotDataFile.good(),std::runtime_error,
    "Error, could not open file live_plot.dat");
  // find all the single points listed in the file:
  while(!livePlotDataFile.eof()){
    std::vector<std::string> tokens = tokenize_line(livePlotDataFile);
    if(tokens.size()==2){
      if(is_number(tokens[0]) && is_number(tokens[1])){
        pts_x_.push_back(strtod(tokens[0].c_str(),NULL));
        pts_y_.push_back(strtod(tokens[1].c_str(),NULL));
        DEBUG_MSG("Live_Plot_Post_Processor::pre_execution_tasks(): found point " << pts_x_[pts_x_.size()-1] << " " << pts_y_[pts_y_.size()-1]);
      }
    }
  }
  num_individual_pts_ = pts_x_.size();
  // find a line in the file if it is defined
  livePlotDataFile.seekg(0,std::ios::beg);
  bool line_defined = false;
  while(!livePlotDataFile.eof()){
    std::vector<std::string> tokens = tokenize_line(livePlotDataFile);
    if(tokens.size()==4){
      if(is_number(tokens[0]) && is_number(tokens[1]) && is_number(tokens[2]) && is_number(tokens[3])){
        TEUCHOS_TEST_FOR_EXCEPTION(line_defined,std::runtime_error,"Error, only one line can be defined");
        line_defined = true;
        scalar_t ax = strtod(tokens[0].c_str(),NULL);
        scalar_t ay = strtod(tokens[1].c_str(),NULL);
        scalar_t bx = strtod(tokens[2].c_str(),NULL);
        scalar_t by = strtod(tokens[3].c_str(),NULL);
        DEBUG_MSG("Live_Plot_Post_Processor::pre_execution_tasks(): adding line from " << ax << ","<< ay << " to " << bx << "," << by);
        // add the origin of the line
        pts_x_.push_back(ax);
        pts_y_.push_back(ay);
        //DEBUG_MSG("Live_Plot_Post_Processor::pre_execution_tasks(): adding line point " << pts_x_[pts_x_.size()-1] << " " << pts_y_[pts_y_.size()-1]);
        // unit vector
        scalar_t dx = bx - ax;
        scalar_t dy = by - ay;
        scalar_t mag = std::sqrt(dx*dx+dy*dy);
        TEUCHOS_TEST_FOR_EXCEPTION(mag==0,std::runtime_error,"");
        scalar_t nx = mag==0.0?0.0:dx/mag;
        scalar_t ny = mag==0.0?0.0:dy/mag;
        scalar_t arc_length = 2.0; // pixels
        scalar_t px = ax;
        scalar_t py = ay;
        int_t s = 1;
        while(true){
          px += arc_length*nx;
          py += arc_length*ny;
          scalar_t len = ((px-ax)*(px-ax) + (py-ay)*(py-ay));
          if(len >= mag*mag) break;
          pts_x_.push_back(px);
          pts_y_.push_back(py);
          //DEBUG_MSG("Live_Plot_Post_Processor::pre_execution_tasks(): adding line point " << pts_x_[pts_x_.size()-1] << " " << pts_y_[pts_y_.size()-1]);
          s++;
          if(s>=10000)break;
        }
      }
    }
  }
  livePlotDataFile.close();

  const int_t num_pts = pts_x_.size();
  std::vector<scalar_t> closest_distances_this_proc(num_pts,0.0);

  DEBUG_MSG("Live_Plot_Post_Processor::pre_execution_tasks(): coord x field " << coords_x_name_ << " y " << coords_y_name_);
  // do a nearest neighbor search to see which processor this point is on:
  const int_t spa_dim = mesh_->spatial_dimension();
  DICe::field_enums::Field_Spec coords_x_spec = mesh_->get_field_spec(coords_x_name_);
  DICe::field_enums::Field_Spec coords_y_spec = mesh_->get_field_spec(coords_y_name_);
  Teuchos::RCP<MultiField> coords;
  if(coords_x_spec.get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE){
    Teuchos::RCP<MultiField> coords_x = mesh_->get_field(coords_x_spec);
    Teuchos::RCP<MultiField> coords_y = mesh_->get_field(coords_y_spec);
    Teuchos::RCP<MultiField_Map> coords_map = mesh_->get_vector_node_dist_map();
    coords = Teuchos::rcp( new MultiField(coords_map,1,true));
    for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
      coords->local_value(i*spa_dim+0) = coords_x->local_value(i);
      coords->local_value(i*spa_dim+1) = coords_y->local_value(i);
    }
  }else{
    // note assumes that the same vector field spec was given for x and y
    coords = mesh_->get_field(coords_x_spec);
  }
  Teuchos::RCP<MultiField> overlap_coords;
  if(coords_x_spec.get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE){
    Teuchos::RCP<MultiField> overlap_coords_x = mesh_->get_overlap_field(coords_x_spec);
    Teuchos::RCP<MultiField> overlap_coords_y = mesh_->get_overlap_field(coords_y_spec);
    Teuchos::RCP<MultiField_Map> coords_map = mesh_->get_vector_node_overlap_map();
    overlap_coords = Teuchos::rcp( new MultiField(coords_map,1,true));
    for(int_t i=0;i<mesh_->get_scalar_node_overlap_map()->get_num_local_elements();++i){
      overlap_coords->local_value(i*spa_dim+0) = overlap_coords_x->local_value(i);
      overlap_coords->local_value(i*spa_dim+1) = overlap_coords_y->local_value(i);
    }
  }else{
    // note assumes that the same vector field spec was given for x and y
    overlap_coords = mesh_->get_overlap_field(coords_x_spec);
  }

  // create neighborhood lists using nanoflann:
  DEBUG_MSG("creating the point cloud of local points using nanoflann");
  Teuchos::RCP<Point_Cloud_2D<scalar_t> > local_point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
  local_point_cloud->pts.resize(local_num_points_);
  for(int_t i=0;i<local_num_points_;++i){
    local_point_cloud->pts[i].x = coords->local_value(i*spa_dim+0);
    local_point_cloud->pts[i].y = coords->local_value(i*spa_dim+1);
  }
  DEBUG_MSG("building the local kd-tree");
  Teuchos::RCP<kd_tree_2d_t> local_kd_tree = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *local_point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  local_kd_tree->buildIndex();
  DEBUG_MSG("local kd-tree completed");
  scalar_t query_pt[2];
  const int_t nearest_num_neigh = 1;
  DEBUG_MSG("performing k-nearest neighbors search for to see which procesors own each live point");
  std::vector<size_t> local_ret_index(nearest_num_neigh);
  std::vector<scalar_t> local_out_dist_sqr(nearest_num_neigh);
  for(int_t i=0;i<num_pts;++i){
    query_pt[0] = pts_x_[i];
    query_pt[1] = pts_y_[i];
    local_kd_tree->knnSearch(&query_pt[0], nearest_num_neigh, &local_ret_index[0], &local_out_dist_sqr[0]);
    // distance to closest node in the mesh on this processor
    closest_distances_this_proc[i] = std::sqrt(local_out_dist_sqr[0]);
  }
  const int_t proc_id = mesh_->get_comm()->get_rank();
  const int_t num_procs = mesh_->get_comm()->get_size();
  Teuchos::Array<int_t> all_zero_owned_ids;
  if(proc_id==0){
    for(int_t i=0;i<num_procs*num_pts;++i)
      all_zero_owned_ids.push_back(i);
  }
  Teuchos::Array<int_t> all_dist_ids;
  for(int_t i=proc_id*num_pts;i<proc_id*num_pts+num_pts;++i)
    all_dist_ids.push_back(i);
  Teuchos::RCP<MultiField_Map> all_zero_map = Teuchos::rcp (new MultiField_Map(-1, all_zero_owned_ids,0,*mesh_->get_comm()));
  Teuchos::RCP<MultiField> all_zero_data = Teuchos::rcp(new MultiField(all_zero_map,1,true));
  Teuchos::RCP<MultiField_Map> all_dist_map = Teuchos::rcp (new MultiField_Map(-1, all_dist_ids,0,*mesh_->get_comm()));
  Teuchos::RCP<MultiField> all_dist_data = Teuchos::rcp(new MultiField(all_dist_map,1,true));

  // all procs populate their piece:
  TEUCHOS_TEST_FOR_EXCEPTION(all_dist_map->get_num_local_elements()!=num_pts,std::runtime_error,"");
  for(int_t i=0;i<num_pts;++i){
    all_dist_data->local_value(i) = closest_distances_this_proc[i];
  }

  // now that all the distances are populated on each proc, send them all to zero
  MultiField_Exporter exporter(*all_zero_map,*all_dist_map);
  all_zero_data->do_import(all_dist_data,exporter,INSERT);

  //for(int_t i=0;i<zero_map->get_num_local_elements();++i){
  //  std::cout << "PROCESSOR " << proc_id << " has element " << i << " value " << zero_data->local_value(i) << std::endl;
  //}

  // figure out the "winners"
  if(proc_id==0){
    std::vector<int_t> owners(num_pts,0);
    for(int_t i=0;i<num_pts;++i){
      int_t owning_proc = 0;
      scalar_t min_dist = std::numeric_limits<scalar_t>::max();
      for(int_t proc=0;proc<num_procs;++proc){
        if(all_zero_data->local_value(proc*num_pts+i)<=min_dist){
          min_dist = all_zero_data->local_value(proc*num_pts+i);
          owning_proc = proc;
        }
      }
      // store the owning proc for each point
      owners[i] = owning_proc;
    }
    // replace the distance in each processor with a 1 for owned or 0 for not owned
    all_zero_data->put_scalar(0.0);
    for(int_t i=0;i<num_pts;++i){
      all_zero_data->local_value(owners[i]*num_pts+i) = 1.0;
    }
  }
  // now export the zero owned values to dist
  MultiField_Exporter exporter_rev(*all_dist_map,*all_zero_map);
  all_dist_data->do_import(all_zero_data,exporter_rev,INSERT);
  for(int_t i=0;i<num_pts;++i){
    if(all_dist_data->local_value(i)>0.0)
      local_indices_.push_back(i);
  }
  //for(size_t i=0;i<local_indices_.size();++i){
  //  DEBUG_MSG("Live_Plot_Post_Processor::pre_execution_tasks(): processor " << proc_id << " owns point " << local_indices_[i]);
  //}

  // create a new dist map with the local indices and store it
  Teuchos::Array<int_t> zero_owned_ids;
  if(proc_id==0){
    for(int_t i=0;i<num_pts;++i)
      zero_owned_ids.push_back(i);
  }
  Teuchos::Array<int_t> dist_ids;
  for(size_t i=0;i<local_indices_.size();++i)
    dist_ids.push_back(local_indices_[i]);
  // count up the number of fields (note some are vector fields so can't just use field_specs_.size())
  num_field_entries_ = 0;
  for(size_t field_it=0;field_it<field_specs_.size();++field_it){
    if(field_specs_[field_it].get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE){
      num_field_entries_++;
    }else if(field_specs_[field_it].get_field_type()==DICe::field_enums::VECTOR_FIELD_TYPE){
      num_field_entries_+=spa_dim;
    }else{
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid field type.");
    }
  }
  zero_map_  = Teuchos::rcp (new MultiField_Map(-1, zero_owned_ids,0,*mesh_->get_comm()));
  zero_data_ = Teuchos::rcp(new MultiField(zero_map_,num_field_entries_,true));
  dist_map_  = Teuchos::rcp (new MultiField_Map(-1, dist_ids,0,*mesh_->get_comm()));
  dist_data_ = Teuchos::rcp(new MultiField(dist_map_,num_field_entries_,true));

  //dist_map_->describe();

  // perform a pass to size the neighbor lists
  neighbor_list_.resize(local_indices_.size());
  neighbor_dist_x_.resize(local_indices_.size());
  neighbor_dist_y_.resize(local_indices_.size());

  // create neighborhood lists using nanoflann:
  DEBUG_MSG("creating the point cloud of overlap points using nanoflann");
  Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud_2D<scalar_t>());
  point_cloud->pts.resize(overlap_num_points_);
  for(int_t i=0;i<overlap_num_points_;++i){
    point_cloud->pts[i].x = overlap_coords->local_value(i*spa_dim+0);
    point_cloud->pts[i].y = overlap_coords->local_value(i*spa_dim+1);
  }
  DEBUG_MSG("building the kd-tree");
  Teuchos::RCP<kd_tree_2d_t> kd_tree = Teuchos::rcp(new kd_tree_2d_t(2 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
  kd_tree->buildIndex();
  DEBUG_MSG("kd-tree completed");
  DEBUG_MSG("performing k-nearest neighbors search");
  std::vector<size_t> ret_index(num_neigh_);
  std::vector<scalar_t> out_dist_sqr(num_neigh_);
  for(size_t i=0;i<local_indices_.size();++i){
    query_pt[0] = pts_x_[local_indices_[i]];
    query_pt[1] = pts_y_[local_indices_[i]];
    kd_tree->knnSearch(&query_pt[0], num_neigh_, &ret_index[0], &out_dist_sqr[0]);
    for(int_t j=0;j<num_neigh_;++j){
      const int_t neigh_olid = ret_index[j];
      neighbor_list_[i].push_back(neigh_olid);
      neighbor_dist_x_[i].push_back(overlap_coords->local_value(neigh_olid*spa_dim+0) - pts_x_[local_indices_[i]]);
      neighbor_dist_y_[i].push_back(overlap_coords->local_value(neigh_olid*spa_dim+1) - pts_y_[local_indices_[i]]);
    }
  }
  neighborhood_initialized_ = true;

  // create output files for each point:
  if(proc_id==0){
    for(int_t i=0;i<num_individual_pts_;++i){
      std::stringstream fileName;
      fileName << "live_plot_pt_" << i << ".txt";
      std::FILE * filePtr = fopen(fileName.str().c_str(),"w");
      fprintf(filePtr,"# point coordinates: %f %f\n",pts_x_[i],pts_y_[i]);
      for(size_t field_it=0;field_it<field_specs_.size();++field_it){
        if(field_specs_[field_it].get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE)
          fprintf(filePtr,"%s,",field_specs_[field_it].get_name_label().c_str());
        else{
          fprintf(filePtr,"%s",field_specs_[field_it].get_name_label().c_str());
          fprintf(filePtr,"%s,","_X");
          fprintf(filePtr,"%s",field_specs_[field_it].get_name_label().c_str());
          fprintf(filePtr,"%s,","_Y");
        }
      }
      fprintf(filePtr,"\n");
      fclose(filePtr);
    }
  }
  DEBUG_MSG("Live_Plot_Post_Processor::pre_execution_tasks(): end");
}

void
Live_Plot_Post_Processor::execute(){
  DEBUG_MSG("Live_Plot_Post_Processor execute() begin");
  if(!neighborhood_initialized_) pre_execution_tasks();

  if(local_indices_.size()==0)return;

  const int_t spa_dim = mesh_->spatial_dimension();
  const bool has_sigma = mesh_->has_field(DICe::field_enums::SIGMA);
  Teuchos::RCP<MultiField> sigma;
  if(has_sigma) sigma = mesh_->get_overlap_field(DICe::field_enums::SIGMA_FS);
  // set up the fields
  std::vector<Teuchos::RCP<MultiField> > field_vec;
  for(size_t i=0;i<field_specs_.size();++i){
    field_vec.push_back(mesh_->get_overlap_field(field_specs_[i]));
  }
  assert(field_vec.size()==field_specs_.size());

  const int_t N = 3;
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  int INFO = 0;
  double *WORK = new double[LWORK];
  // Note, LAPACK does not allow templating on long int or scalar_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;

  assert(neighbor_list_.size()==local_indices_.size());
  assert((int_t)neighbor_list_[0].size()==num_neigh_);
  std::vector<bool> neigh_valid(num_neigh_,true);
  int_t neigh_id = 0;
  for(size_t pt=0;pt<local_indices_.size();++pt){
    //const int_t pt_id = local_indices_[pt];
    //DEBUG_MSG("Processing live plot point " << pts_x_[pt] << " " << pts_y_[pt] );
    // search the neighbors to see how many valid neighbors exist:
    int_t num_valid_neigh = 0;
     for(int_t j=0;j<num_neigh_;++j){
       if(has_sigma){
         if(sigma->local_value(neighbor_list_[pt][j])>=0.0){
           neigh_valid[j] = true;
           num_valid_neigh++;
         }else{
           neigh_valid[j] = false;
         }
       }else{
         neigh_valid[j] = true;
         num_valid_neigh++;
       }
     }
    //DEBUG_MSG("Live plot point num valid neighbors: " << num_valid_neigh);
    if(num_valid_neigh < 3){
      DEBUG_MSG("Live plot point " << local_indices_[pt] << " does not have enough neighbors to calculate values.");
      continue;
    }else{
      std::vector<Teuchos::ArrayRCP<double> > u;
      std::vector<Teuchos::ArrayRCP<double> > X_t_u;
      std::vector<Teuchos::ArrayRCP<double> > coeffs;
      for(int_t i=0;i<num_field_entries_;++i){
        u.push_back(Teuchos::ArrayRCP<double>(num_valid_neigh,0.0));
        X_t_u.push_back(Teuchos::ArrayRCP<double>(N,0.0));
        coeffs.push_back(Teuchos::ArrayRCP<double>(N,0.0));
      }
      Teuchos::SerialDenseMatrix<int_t,double> X_t(N,num_valid_neigh, true);
      Teuchos::SerialDenseMatrix<int_t,double> X_t_X(N,N,true);

      // gather the displacements of the neighbors
      int_t valid_id = 0;
      for(int_t j=0;j<num_neigh_;++j){
        if(!neigh_valid[j])continue;
        neigh_id = neighbor_list_[pt][j];
        int_t field_id = 0;
        for(size_t field_it=0;field_it<field_specs_.size();++field_it){
          if(field_specs_[field_it].get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE){
            u[field_id][valid_id] = field_vec[field_it]->local_value(neigh_id);
            field_id++;
          }
          else{
            u[field_id][valid_id] = field_vec[field_it]->local_value(neigh_id*spa_dim+0);
            field_id++;
            u[field_id][valid_id] = field_vec[field_it]->local_value(neigh_id*spa_dim+1);
            field_id++;
          }
        }
        // set up the X^T matrix
        X_t(0,valid_id) = 1.0;
        X_t(1,valid_id) = neighbor_dist_x_[pt][j];
        X_t(2,valid_id) = neighbor_dist_y_[pt][j];
        valid_id++;
      }
      // set up X^T*X
      for(int_t k=0;k<N;++k){
        for(int_t m=0;m<N;++m){
          for(int_t j=0;j<num_valid_neigh;++j){
            X_t_X(k,m) += X_t(k,j)*X_t(m,j);
          }
        }
      }
      // Invert X^T*X
      lapack.GETRF(X_t_X.numRows(),X_t_X.numCols(),X_t_X.values(),X_t_X.numRows(),IPIV,&INFO);
      lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),IPIV,WORK,LWORK,&INFO);

      // compute X^T*u
      for(int_t i=0;i<N;++i){
        for(int_t j=0;j<num_valid_neigh;++j){
          for(int_t field_it=0;field_it<num_field_entries_;++field_it){
            X_t_u[field_it][i] += X_t(i,j)*u[field_it][j];
          }
        }
      }
      // compute the coeffs
      for(int_t i=0;i<N;++i){
        for(int_t j=0;j<N;++j){
          for(int_t field_it=0;field_it<num_field_entries_;++field_it){
            coeffs[field_it][i] += X_t_X(i,j)*X_t_u[field_it][j];
          }
        }
      }
      assert(dist_map_->is_node_global_elem(local_indices_[pt]));
      const int_t lid = dist_map_->get_local_element(local_indices_[pt]);
      for(int_t field_it=0;field_it<num_field_entries_;++field_it){
        dist_data_->local_value(lid,field_it) = coeffs[field_it][0];
      }
    } // end has enough valid neighbors to compute a value
  } // end point loop

  delete [] WORK;
  delete [] IPIV;

  MultiField_Exporter exporter(*zero_map_,*dist_map_);
  zero_data_->do_import(dist_data_,exporter,INSERT);

  const int_t proc_id = mesh_->get_comm()->get_rank();
  if(proc_id==0){
    // append the file for individual files for individual points
    for(int_t i=0;i<num_individual_pts_;++i){
      std::stringstream fileName;
      fileName << "live_plot_pt_" << i << ".txt";
      std::FILE * filePtr = fopen(fileName.str().c_str(),"a");
      for(int_t field_it=0;field_it<num_field_entries_;++field_it)
        fprintf(filePtr,"%f,",zero_data_->local_value(i,field_it));
      fprintf(filePtr,"\n");
      fclose(filePtr);
    }
    // print the line data for this step (create a new line file for each step)
    std::stringstream fileName;
    fileName << "live_plot_line_step_" << frame_index_++ << ".txt";
    std::FILE * filePtr = fopen(fileName.str().c_str(),"w");
    fprintf(filePtr,"X,Y,");

    for(size_t field_it=0;field_it<field_specs_.size();++field_it){
      if(field_specs_[field_it].get_field_type()==DICe::field_enums::SCALAR_FIELD_TYPE)
        fprintf(filePtr,"%s,",field_specs_[field_it].get_name_label().c_str());
      else{
        fprintf(filePtr,"%s",field_specs_[field_it].get_name_label().c_str());
        fprintf(filePtr,"%s,","_X");
        fprintf(filePtr,"%s",field_specs_[field_it].get_name_label().c_str());
        fprintf(filePtr,"%s,","_Y");
      }
    }
    fprintf(filePtr,"\n");
    for(size_t pt=num_individual_pts_;pt<pts_x_.size();++pt){
      fprintf(filePtr,"%f,%f,",pts_x_[pt],pts_y_[pt]);
      for(int_t field_it=0;field_it<num_field_entries_;++field_it)
        fprintf(filePtr,"%f,",zero_data_->local_value(pt,field_it));
      fprintf(filePtr,"\n");
    }
    fclose(filePtr);
  }

  DEBUG_MSG("Live_Plot_Post_Processor execute() end");
}

}// End DICe Namespace
