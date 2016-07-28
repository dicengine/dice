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

#include <DICe_BCManager.h>
#include <DICe_GlobalUtils.h>
#include <DICe_Global.h>

namespace DICe {

namespace global{

BC_Manager::BC_Manager(Global_Algorithm * alg) :
    spa_dim_(alg->mesh()->spatial_dimension()),
    row_bc_register_(0),
    col_bc_register_(0),
    mixed_bc_register_(0),
    row_bc_register_size_(0),
    col_bc_register_size_(0),
    mixed_bc_register_size_(0),
    bc_register_initialized_(false),
    mesh_(alg->mesh()),
    //l_mesh_(alg->l_mesh()),
    is_mixed_(false),
    alg_(alg)
{
  is_mixed_ = alg->is_mixed_formulation();

  const int_t lagrange_size = is_mixed_ ? mesh_->get_scalar_node_dist_map()->get_num_local_elements() : 0;
  initialize_bc_register(mesh_->get_vector_node_dist_map()->get_num_local_elements(),
    mesh_->get_vector_node_overlap_map()->get_num_local_elements(),lagrange_size);

  DEBUG_MSG("BC_Manager::BC_Manager(): setting the boundary condition nodes");

  // positive number is dirichlet node set
  // 0 is a neumann boundary
  // negative number is lagrange multiplier node set
  DICe::mesh::bc_set * bc_set = mesh_->get_node_bc_sets();
  DICe::mesh::bc_set::iterator it = bc_set->begin();
  DICe::mesh::bc_set::iterator it_end = bc_set->end();
  for(;it!=it_end;++it){
    const int_t boundary_node_set_id = it->first;
    DEBUG_MSG("Setting up bc register flags for node set id " << boundary_node_set_id);
    if(boundary_node_set_id==0)
      continue; // neumann boundary is 0 id
    // lagrange multiplier boundary is negative id
    if(boundary_node_set_id<0 && is_mixed_){
      const int_t num_bc_nodes = bc_set->find(boundary_node_set_id)->second.size();
      for(int_t i=0;i<num_bc_nodes;++i){
        const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
        DEBUG_MSG(" lagrange condition on node " << node_gid);
        bool is_local_node = mesh_->get_scalar_node_dist_map()->is_node_global_elem(node_gid);
        if(is_local_node){
          const int_t row_id = mesh_->get_scalar_node_dist_map()->get_local_element(node_gid);
          register_mixed_bc(row_id);
        }
      }
    }
    else{
      // make sure that a bc def exists for this nodes set
      TEUCHOS_TEST_FOR_EXCEPTION(boundary_node_set_id <=0, std::runtime_error,"bc set id should be positive here");
      TEUCHOS_TEST_FOR_EXCEPTION((int_t)mesh_->bc_defs()->size() < boundary_node_set_id,std::runtime_error,
        "bc_def does not exist for this node set id " << boundary_node_set_id);
      const int_t comp = (*mesh_->bc_defs())[boundary_node_set_id-1].comp_;
      const int_t num_bc_nodes = bc_set->find(boundary_node_set_id)->second.size();
      for(int_t i=0;i<num_bc_nodes;++i){
        const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
        bool is_local_node = mesh_->get_vector_node_dist_map()->is_node_global_elem((node_gid-1)*spa_dim_+1);
        int_t col_id = mesh_->get_vector_node_overlap_map()->get_local_element((node_gid-1)*spa_dim_+1);
        if(comp==0||comp==2){
          DEBUG_MSG("disp x condition on node " << node_gid);
          if(is_local_node){
            const int_t row_id = mesh_->get_vector_node_dist_map()->get_local_element((node_gid-1)*spa_dim_+1);
            register_row_bc(row_id);
          }
          register_col_bc(col_id);
        }
        if(comp==1||comp==2){
          DEBUG_MSG("disp y condition on node " << node_gid);
          is_local_node = mesh_->get_vector_node_dist_map()->is_node_global_elem((node_gid-1)*spa_dim_+2);
          if(is_local_node){
            const int_t row_id = mesh_->get_vector_node_dist_map()->get_local_element((node_gid-1)*spa_dim_+2);
            register_row_bc(row_id);
          }
          col_id = mesh_->get_vector_node_overlap_map()->get_local_element((node_gid-1)*spa_dim_+2);
          register_col_bc(col_id);
        }
      }
    } // end bcs from node sets
  }
}

void
BC_Manager::initialize_bc_register(const int_t row_register_size,
  const int_t col_register_size,
  const int_t mixed_register_size)
{
  row_bc_register_size_ = row_register_size;
  row_bc_register_ = new bool[row_bc_register_size_];
  for(int_t i=0;i<row_bc_register_size_;++i)
    row_bc_register_[i] = false;

  col_bc_register_size_ = col_register_size;
  col_bc_register_ = new bool[col_bc_register_size_];
  for(int_t i=0;i<col_bc_register_size_;++i)
    col_bc_register_[i] = false;

  mixed_bc_register_size_ = mixed_register_size;
  mixed_bc_register_ = new bool[mixed_bc_register_size_];
  for(int_t i=0;i<mixed_bc_register_size_;++i)
    mixed_bc_register_[i] = false;

  bc_register_initialized_ = true;
}

void
BC_Manager::clear_bc_register()
{
  for(int_t i=0;i<row_bc_register_size_;++i)
    row_bc_register_[i] = false;
  for(int_t i=0;i<col_bc_register_size_;++i)
    col_bc_register_[i] = false;
  for(int_t i=0;i<mixed_bc_register_size_;++i)
    mixed_bc_register_[i] = false;
}

void
BC_Manager::create_bc(Global_EQ_Term eq_term,
  const bool is_mixed){
  if(eq_term==DIRICHLET_DISPLACEMENT_BC){
    Teuchos::RCP<Boundary_Condition> bc_rcp = Teuchos::rcp(new Dirichlet_BC(mesh_,is_mixed));
    bcs_.push_back(bc_rcp);
  }
  else if(eq_term==SUBSET_DISPLACEMENT_BC){
    Teuchos::RCP<Boundary_Condition> bc_rcp = Teuchos::rcp(new Subset_BC(alg_,mesh_,is_mixed));
    bcs_.push_back(bc_rcp);
  }
  else if(eq_term==LAGRANGE_BC){
    Teuchos::RCP<Boundary_Condition> bc_rcp = Teuchos::rcp(new Lagrange_BC(mesh_,is_mixed));
    bcs_.push_back(bc_rcp);
  }
  else if(eq_term==MMS_DIRICHLET_DISPLACEMENT_BC){
    Teuchos::RCP<Boundary_Condition> bc_rcp = Teuchos::rcp(new MMS_BC(alg_,mesh_,is_mixed));
    bcs_.push_back(bc_rcp);
  }
  else if(eq_term==MMS_LAGRANGE_BC){
    Teuchos::RCP<Boundary_Condition> bc_rcp = Teuchos::rcp(new MMS_Lagrange_BC(alg_,mesh_,is_mixed));
    bcs_.push_back(bc_rcp);
  }
  else if(eq_term==CORNER_BC){
    Teuchos::RCP<Boundary_Condition> bc_rcp = Teuchos::rcp(new Corner_BC(mesh_,is_mixed));
    bcs_.push_back(bc_rcp);
  }
  else if(eq_term==CONSTANT_IC){
    Teuchos::RCP<Boundary_Condition> bc_rcp = Teuchos::rcp(new Constant_IC(alg_,mesh_,is_mixed));
    ics_.push_back(bc_rcp);
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid bc type");
  }
}

void
BC_Manager::apply_bcs(const bool first_iteration){
  DEBUG_MSG("BC_Manager::apply_bcs(): first iteration: " << first_iteration);
  for(size_t i=0;i<bcs_.size();++i)
    bcs_[i]->apply(first_iteration);
}

void
BC_Manager::apply_ics(const bool first_iteration){
  DEBUG_MSG("BC_Manager::apply_ics(): first iteration: " << first_iteration);
  for(size_t i=0;i<ics_.size();++i)
    ics_[i]->apply(first_iteration);
}

void
Lagrange_BC::apply(const bool first_iteration){
  if(!is_mixed_) return;
  // get the residual field
  Teuchos::RCP<MultiField> residual = is_mixed_ ? mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS) :
      mesh_->get_field(mesh::field_enums::RESIDUAL_FS);

  const int_t node_set_id = -1; // lagrangian nodes are all in set -1
  const int_t mixed_global_offset = mesh_->get_vector_node_dist_map()->get_num_global_elements();
  DEBUG_MSG("Dirichlet_BC::apply(): applying a 0.0 Lagrange multiplier bc to node set id " << node_set_id);
  DICe::mesh::bc_set * bc_sets = mesh_->get_node_bc_sets();
  if(bc_sets->find(node_set_id)==bc_sets->end())return;
  const int_t num_bc_nodes = bc_sets->find(node_set_id)->second.size();
  for(int_t i=0;i<num_bc_nodes;++i){
    const int_t node_gid = bc_sets->find(node_set_id)->second[i];
    residual->global_value(node_gid+mixed_global_offset) = 0.0;
  } // bc node
}

void
Dirichlet_BC::apply(const bool first_iteration){
  // get the residual field
  Teuchos::RCP<MultiField> residual = is_mixed_ ? mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS) :
      mesh_->get_field(mesh::field_enums::RESIDUAL_FS);

  const int_t spa_dim = mesh_->spatial_dimension();
  // iterate the boudnary condition def
  for(size_t i=0;i<mesh_->bc_defs()->size();++i){
    const int_t node_set_id = i+1;  // +1 because bc ids are one-based
    if(!(*mesh_->bc_defs())[i].has_value_) continue;
    const scalar_t value_ = (*mesh_->bc_defs())[i].value_;
    const int_t comp = (*mesh_->bc_defs())[i].comp_;
    DEBUG_MSG("Dirichlet_BC::apply(): applying a Dirichlet bc to node set id " << node_set_id
      << " comp " << comp << " value " << value_);
    DICe::mesh::bc_set * bc_sets = mesh_->get_node_bc_sets();
    TEUCHOS_TEST_FOR_EXCEPTION(bc_sets->find(node_set_id)==bc_sets->end(),std::runtime_error,
      "Error, invalid node set id: " << node_set_id);
    const int_t num_bc_nodes = bc_sets->find(node_set_id)->second.size();
    DEBUG_MSG("Dirichlet_BC::apply(): number of nodes in set " << num_bc_nodes);
    for(int_t i=0;i<num_bc_nodes;++i){
      const int_t node_gid = bc_sets->find(node_set_id)->second[i];
      const int_t ix = (node_gid-1)*spa_dim + 1;
      const int_t iy = (node_gid-1)*spa_dim + 2;
      if(comp==0||comp==2)
        residual->global_value(ix) = first_iteration ? value_ : 0.0;
      if(comp==1||comp==2)
        residual->global_value(iy) = first_iteration ? value_ : 0.0;
    }
  }
}

void
Subset_BC::apply(const bool first_iteration){
  // get the residual field
  Teuchos::RCP<MultiField> residual = is_mixed_ ? mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS) :
      mesh_->get_field(mesh::field_enums::RESIDUAL_FS);
  Teuchos::RCP<MultiField> coords = mesh_->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);

  const int_t spa_dim = mesh_->spatial_dimension();
  // iterate the boudnary condition def
  for(size_t i=0;i<mesh_->bc_defs()->size();++i){
    const int_t node_set_id = i+1;  // +1 because bc ids are one-based
    if(!(*mesh_->bc_defs())[i].use_subsets_) continue;
    const int_t comp = (*mesh_->bc_defs())[i].comp_;
    const int_t subset_size = (*mesh_->bc_defs())[i].subset_size_;
    DEBUG_MSG("Subset_BC::apply(): applying a subset bc to node set id " << node_set_id
      << " subset size " << subset_size);
    DICe::mesh::bc_set * bc_sets = mesh_->get_node_bc_sets();
    TEUCHOS_TEST_FOR_EXCEPTION(bc_sets->find(node_set_id)==bc_sets->end(),std::runtime_error,
      "Error, invalid node set id: " << node_set_id);
    const int_t num_bc_nodes = bc_sets->find(node_set_id)->second.size();
    DEBUG_MSG("Subset_BC::apply(): number of nodes in set " << num_bc_nodes);
    for(int_t i=0;i<num_bc_nodes;++i){
      const int_t node_gid = bc_sets->find(node_set_id)->second[i];
      const int_t ix = (node_gid-1)*spa_dim + 1;
      const int_t iy = (node_gid-1)*spa_dim + 2;
      if(!first_iteration){
        if(comp==0||comp==2)
          residual->global_value(ix) = 0.0;
        if(comp==1||comp==2)
          residual->global_value(iy) = 0.0;
      }
      else{
        scalar_t b_x = 0.0;
        scalar_t b_y = 0.0;
        const scalar_t x = coords->global_value(ix);
        const scalar_t y = coords->global_value(iy);
        // get the closest pixel to x and y
        int_t px = (int_t)x;
        if(x - (int_t)x >= 0.5) px++;
        int_t py = (int_t)y;
        if(y - (int_t)y >= 0.5) py++;
        DICe::global::subset_velocity(alg_,px,py,subset_size,b_x,b_y);
        if(comp==0||comp==2)
          residual->global_value(ix) = b_x;
        if(comp==1||comp==2)
          residual->global_value(iy) = b_y;
      }
    }
  }
}

void
MMS_BC::apply(const bool first_iteration){
  // get the residual field
  Teuchos::RCP<MultiField> residual = is_mixed_ ? mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS) :
      mesh_->get_field(mesh::field_enums::RESIDUAL_FS);
  Teuchos::RCP<MultiField> coords = mesh_->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  Teuchos::RCP<MultiField> exact_sol = mesh_->get_field(mesh::field_enums::EXACT_SOL_VECTOR_FS);
  Teuchos::RCP<MultiField> image_phi = mesh_->get_field(mesh::field_enums::IMAGE_PHI_FS);
  Teuchos::RCP<MultiField> image_grad_phi = mesh_->get_field(mesh::field_enums::IMAGE_GRAD_PHI_FS);
  const int_t spa_dim = mesh_->spatial_dimension();

  TEUCHOS_TEST_FOR_EXCEPTION(alg_->mms_problem()==Teuchos::null,std::runtime_error,"Error, mms_problem should not be null");

  // populate the image and solution fields
  if(first_iteration){
    for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
      int_t ix = i*2+0;
      int_t iy = i*2+1;
      scalar_t b_x = 0.0;
      scalar_t b_y = 0.0;
      const scalar_t x = coords->local_value(ix);
      const scalar_t y = coords->local_value(iy);
      alg_->mms_problem()->velocity(x,y,b_x,b_y);
      scalar_t phi = 0.0,d_phi_dt=0.0,grad_phi_x=0.0,grad_phi_y=0.0;
      alg_->mms_problem()->phi(x,y,phi);
      image_phi->local_value(i) = phi;
      alg_->mms_problem()->phi_derivatives(x,y,d_phi_dt,grad_phi_x,grad_phi_y);
      image_grad_phi->local_value(ix) = grad_phi_x;
      image_grad_phi->local_value(iy) = grad_phi_y;
      exact_sol->local_value(ix) = b_x;
      exact_sol->local_value(iy) = b_y;
    }
    if(is_mixed_){
      Teuchos::RCP<MultiField> exact_lag = alg_->mesh()->get_field(mesh::field_enums::EXACT_LAGRANGE_MULTIPLIER_FS);
      for(int_t i=0;i<alg_->mesh()->get_scalar_node_dist_map()->get_num_local_elements();++i){
        int_t ix = i*2+0;
        int_t iy = i*2+1;
        const scalar_t x = coords->local_value(ix);
        const scalar_t y = coords->local_value(iy);
        scalar_t l_out = 0.0;
        alg_->mms_problem()->lagrange(x,y,l_out);
        exact_lag->local_value(i) = l_out;
      }
    }
  }
  for(size_t i=0;i<mesh_->bc_defs()->size();++i){
    const int_t node_set_id = i+1;
    DEBUG_MSG("MMS_BC::apply(): applying a MMS Dirichlet bc to node set id " << node_set_id);
    DICe::mesh::bc_set * bc_sets = mesh_->get_node_bc_sets();
    TEUCHOS_TEST_FOR_EXCEPTION(bc_sets->find(node_set_id)==bc_sets->end(),std::runtime_error,
      "Error, invalid node set id");
    TEUCHOS_TEST_FOR_EXCEPTION((int_t)mesh_->bc_defs()->size()<node_set_id,std::runtime_error,
      "Error, invalid node set id");
    const int_t num_bc_nodes = bc_sets->find(node_set_id)->second.size();
    const int_t comp = (*mesh_->bc_defs())[node_set_id-1].comp_;
    DEBUG_MSG("MMS_BC::apply(): number of nodes in set " << num_bc_nodes);
    for(int_t i=0;i<num_bc_nodes;++i){
      const int_t node_gid = bc_sets->find(node_set_id)->second[i];
      const int_t ix = (node_gid-1)*spa_dim + 1;
      const int_t iy = (node_gid-1)*spa_dim + 2;
      if(!first_iteration){
        if(comp==0||comp==2)
          residual->global_value(ix) = 0.0;
        if(comp==1||comp==2)
          residual->global_value(iy) = 0.0;
      }
      else{
        const scalar_t x = coords->global_value(ix);
        const scalar_t y = coords->global_value(iy);
        scalar_t b_x = 0.0, b_y = 0.0;
        alg_->mms_problem()->velocity(x,y,b_x,b_y);
        if(comp==0||comp==2)
          residual->global_value(ix) = b_x;
        if(comp==1||comp==2)
          residual->global_value(iy) = b_y;
      }
    }
  }
}

void
MMS_Lagrange_BC::apply(const bool first_iteration){
  // get the residual field
  Teuchos::RCP<MultiField> residual = is_mixed_ ? mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS) :
      mesh_->get_field(mesh::field_enums::RESIDUAL_FS);
  Teuchos::RCP<MultiField> coords = mesh_->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  const int_t spa_dim = mesh_->spatial_dimension();
  TEUCHOS_TEST_FOR_EXCEPTION(alg_->mms_problem()==Teuchos::null,std::runtime_error,"Error, mms_problem should not be null");
  //TEUCHOS_TEST_FOR_EXCEPTION(alg_->mesh()==Teuchos::null,std::runtime_error,
  //  "Error, mesh pointer should not be null");
  const int_t mixed_global_offset = mesh_->get_vector_node_dist_map()->get_num_global_elements();
//  if(first_iteration){ // populate the exact solution fields
//    Teuchos::RCP<MultiField> exact_lag = alg_->l_mesh()->get_field(mesh::field_enums::EXACT_LAGRANGE_MULTIPLIER_FS);
//    for(int_t i=0;i<alg_->l_mesh()->get_scalar_node_dist_map()->get_num_local_elements();++i){
//      int_t ix = i*2+0;
//      int_t iy = i*2+1;
//      const scalar_t x = coords->local_value(ix);
//      const scalar_t y = coords->local_value(iy);
//      scalar_t l_out = 0.0;
//      alg_->mms_problem()->lagrange(x,y,l_out);
//      exact_lag->local_value(i) = l_out;
//    }
//  }
  const int_t node_set_id = -1;  // mms bcs are hard coded to node set id -1 for the lagrange multiplier
  DEBUG_MSG("MMS_BC::apply(): applying a Lagrange Multiplier BC to node set id " << node_set_id);
  DICe::mesh::bc_set * bc_sets = mesh_->get_node_bc_sets();
  if(bc_sets->find(node_set_id)==bc_sets->end())return;
  const int_t num_bc_nodes = bc_sets->find(node_set_id)->second.size();
  DEBUG_MSG("MMS_BC::apply(): number of nodes in set " << num_bc_nodes);
  for(int_t i=0;i<num_bc_nodes;++i){
    const int_t node_gid = bc_sets->find(node_set_id)->second[i];
    const int_t ix = (node_gid-1)*spa_dim + 1;
    const int_t iy = (node_gid-1)*spa_dim + 2;
    if(!first_iteration){
      residual->global_value(node_gid + mixed_global_offset) = 0.0;
    }
    else{
      const scalar_t x = coords->global_value(ix);
      const scalar_t y = coords->global_value(iy);
      scalar_t l_out = 0.0;
      alg_->mms_problem()->lagrange(x,y,l_out);
      residual->global_value(node_gid + mixed_global_offset) = l_out;
    }
  }
}

void
Corner_BC::apply(const bool first_iteration){
  // get the residual field
  Teuchos::RCP<MultiField> residual = is_mixed_ ? mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS) :
      mesh_->get_field(mesh::field_enums::RESIDUAL_FS);
  const int_t mixed_global_offset = mesh_->get_vector_node_dist_map()->get_num_global_elements();
  DEBUG_MSG("Corner_BC::apply(): applying 0.0 constraint to node 1");
  const int_t node_gid = 1;
  residual->global_value(node_gid + mixed_global_offset) = 0.0;
}

void
Constant_IC::apply(const bool first_iteration){
  if(!first_iteration) return; // only apply this ic on the first it
  // get the residual field
  Teuchos::RCP<MultiField> lhs = is_mixed_ ? mesh_->get_field(mesh::field_enums::MIXED_LHS_FS):
      mesh_->get_field(mesh::field_enums::LHS_FS);
  Teuchos::RCP<MultiField> disp_nm1 = mesh_->get_field(mesh::field_enums::DISPLACEMENT_NM1_FS);
  const int_t spa_dim = mesh_->spatial_dimension();

  // populate the image and solution fields
  for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
    int_t ix = i*spa_dim+0;
    int_t iy = i*spa_dim+1;
    lhs->local_value(ix) = alg_->mesh()->ic_value_x();
    lhs->local_value(iy) = alg_->mesh()->ic_value_y();
    disp_nm1->local_value(ix) = alg_->mesh()->ic_value_x();
    disp_nm1->local_value(iy) = alg_->mesh()->ic_value_y();
  }
}


} // end namespace global

}// End DICe Namespace

//  /// neighbor ids for dirichlet boundary (used for filtering bcs)
//  std::map<int_t,std::vector<int_t> > bc_neighbors_;
//  /// neighbor distances^2 for dirichlet boundary (used for filtering bcs)
//  std::map<int_t,std::vector<scalar_t> > bc_neighbor_distances_;
//    // set up the neighbor list for each boundary node
//    DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): creating the point cloud");
//    /// pointer to the point cloud used for the neighbor searching
//    MultiField & coords = *mesh_->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
//    Teuchos::RCP<Point_Cloud<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud<scalar_t>());
//    std::vector<int_t> node_map(num_bc_nodes);
//    point_cloud->pts.resize(num_bc_nodes);
//    for(int_t i=0;i<num_bc_nodes;++i){
//      const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
//      point_cloud->pts[i].x = coords.global_value((node_gid-1)*spa_dim + 1);
//      point_cloud->pts[i].y = coords.global_value((node_gid-1)*spa_dim + 2);
//      point_cloud->pts[i].z = 0.0;
//      node_map[i] = node_gid;
//    }
//    DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): building the kd-tree");
//    /// pointer to the kd-tree used for searching
//    Teuchos::RCP<my_kd_tree_t> kd_tree =
//        Teuchos::rcp(new my_kd_tree_t(3 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
//    kd_tree->buildIndex();
//    DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): setting up neighbors");
//    const int_t num_neighbors = 5;
//    scalar_t query_pt[3];
//    for(int_t i=0;i<num_bc_nodes;++i){
//      std::vector<size_t> ret_index(num_neighbors);
//      std::vector<int_t> neighbors(num_neighbors);
//      std::vector<scalar_t> out_dist_sqr(num_neighbors);
//      const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
//      query_pt[0] = point_cloud->pts[i].x;
//      query_pt[1] = point_cloud->pts[i].y;
//      query_pt[2] = point_cloud->pts[i].z;
//      kd_tree->knnSearch(&query_pt[0], num_neighbors, &ret_index[0], &out_dist_sqr[0]);
//      for(int_t j=0;j<num_neighbors;++j)
//        neighbors[j] = node_map[ret_index[j]];
//      bc_neighbors_.insert(std::pair<int_t,std::vector<int_t> >(node_gid,neighbors));
//      bc_neighbor_distances_.insert(std::pair<int_t,std::vector<scalar_t> >(node_gid,out_dist_sqr));
//    }
//    // describe the neighbors of each point:
//    for(int_t i=0;i<num_bc_nodes;++i){
//      const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
//      DEBUG_MSG("boundary node " << node_gid << " at " << point_cloud->pts[i].x << " " << point_cloud->pts[i].y);
//      DEBUG_MSG("  has neighbors and distances: ");
//      for(int_t j=0;j<num_neighbors;++j){
//        TEUCHOS_TEST_FOR_EXCEPTION(bc_neighbors_.find(node_gid)==bc_neighbors_.end(),std::runtime_error,
//          "Error, neighbors invalid node gid");
//        TEUCHOS_TEST_FOR_EXCEPTION(bc_neighbor_distances_.find(node_gid)==bc_neighbor_distances_.end(),std::runtime_error,
//          "Error, neighbor distances invalid node gid");
//        DEBUG_MSG("  " << bc_neighbors_.find(node_gid)->second[j] << " " << bc_neighbor_distances_.find(node_gid)->second[j]);
//      }
//    }
//  }

