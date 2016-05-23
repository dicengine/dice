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

#include <DICe_Global.h>
#include <DICe_Schema.h>
#include <DICe_TriangleUtils.h>
#include <DICe_MeshIO.h>
#include <DICe_ParameterUtilities.h>
#include <DICe_Preconditioner.h>

namespace DICe {

namespace global{

Global_Algorithm::Global_Algorithm(Schema * schema,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  schema_(schema),
  mesh_size_(1000.0),
  alpha2_(1.0),
  mms_problem_(Teuchos::null),
  is_initialized_(false),
  global_formulation_(NO_SUCH_GLOBAL_FORMULATION),
  global_solver_(CG_SOLVER)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!schema,std::runtime_error,"Error, cannot have null schema in this constructor");
  default_constructor_tasks(params);
}

Global_Algorithm::Global_Algorithm(const Teuchos::RCP<Teuchos::ParameterList> & params):
  schema_(NULL),
  mesh_size_(1000.0),
  alpha2_(1.0),
  is_initialized_(false),
  global_formulation_(NO_SUCH_GLOBAL_FORMULATION),
  global_solver_(CG_SOLVER)
{
  default_constructor_tasks(params);
}

void
Global_Algorithm::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){

  TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,
    "Error, params must be defined");

  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::global_formulation),std::runtime_error,"Error, parameter: global_formulation must be defined");
  global_formulation_ = params->get<Global_Formulation>(DICe::global_formulation);

  global_solver_ = params->get<Global_Solver>(DICe::global_solver,CG_SOLVER);
  DEBUG_MSG("Global_Algorithm::default_constructor_tasks(): global solver type: " << to_string(global_solver_));

  /// get the mesh size from the params
  if(params->isParameter(DICe::mesh_size))
    mesh_size_ = params->get<double>(DICe::mesh_size);
  DEBUG_MSG("Global_Algorithm::default_constructor_tasks(): Mesh size " << mesh_size_);

  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::output_prefix),std::runtime_error,
    "Error, output_prefix must be defined");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::output_folder),std::runtime_error,
    "Error, output_folder must be defined");
  const std::string output_folder = params->get<std::string>(DICe::output_folder);
  const std::string output_prefix = params->get<std::string>(DICe::output_prefix);
  std::stringstream output_file_name_ss;
  std::stringstream linear_output_file_name_ss;
  output_file_name_ss << output_prefix << ".e";
  linear_output_file_name_ss << output_prefix << "_linear.e";
  output_file_name_ = output_file_name_ss.str();
  linear_output_file_name_ = linear_output_file_name_ss.str();
  DEBUG_MSG("Global_Algorithm::default_constructor_tasks(): output file name: " << output_file_name_);

  if(params->isParameter(DICe::mms_spec)){
    TEUCHOS_TEST_FOR_EXCEPTION(params->isParameter(DICe::subset_file),std::runtime_error,"Error, subset file cannot be defined"
      " in the parameters file for an mms problem");
    // carve off the mms_spec params:
    Teuchos::ParameterList mms_sublist = params->sublist(DICe::mms_spec);
    Teuchos::RCP<Teuchos::ParameterList> mms_params = Teuchos::rcp( new Teuchos::ParameterList());
    // iterate the sublist and add the params to the output params:
    for(Teuchos::ParameterList::ConstIterator it=mms_sublist.begin();it!=mms_sublist.end();++it){
      mms_params->setEntry(it->first,it->second);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(!mms_params->isParameter(DICe::problem_name),std::runtime_error,
      "Error, the problem_name must be defined in the mms_spec");

    MMS_Problem_Factory mms_factory;
    mms_problem_ = mms_factory.create(mms_params);

    // determine the problem extents:
    Teuchos::ArrayRCP<scalar_t> points_x(4);
    Teuchos::ArrayRCP<scalar_t> points_y(4);
    points_x[0] = 0.0; points_y[0] = 0.0;
    points_x[1] = mms_problem_->dim_x(); points_y[1] = 0.0;
    points_x[2] = mms_problem_->dim_x(); points_y[2] = mms_problem_->dim_y();
    points_x[3] = 0.0; points_y[3] = mms_problem_->dim_y();
    mesh_ = DICe::generate_tri6_mesh(points_x,points_y,mesh_size_,output_file_name_);
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(schema_==NULL,std::runtime_error,"If not an mms problem, schema must not be null.");
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::subset_file),std::runtime_error,"Error, subset file must be defined");
    const std::string & subset_file = params->get<std::string>(DICe::subset_file);
    mesh_ = DICe::generate_tri6_mesh(subset_file,mesh_size_,output_file_name_);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(mesh_==Teuchos::null,std::runtime_error,"Error, mesh should not be a null pointer here.");
  DICe::mesh::create_output_exodus_file(mesh_,output_folder);
  // if this is a mixed formulation, generate the lagrange multiplier mesh
  if(is_mixed_formulation()){//||global_formulation_==LEVENBERG_MARQUARDT){
    l_mesh_ = DICe::mesh::create_tri3_exodus_mesh_from_tri6(mesh_,linear_output_file_name_);
    TEUCHOS_TEST_FOR_EXCEPTION(l_mesh_==Teuchos::null,std::runtime_error,"Error, mesh should not be a null pointer here.");
    DICe::mesh::create_output_exodus_file(l_mesh_,output_folder);
    // create the mixed field maps for the master mesh
    mesh_->create_mixed_node_field_maps(l_mesh_);
  }
  // create the necessary fields:
  mesh_->create_field(mesh::field_enums::DISPLACEMENT_FS);
  mesh_->create_field(mesh::field_enums::RESIDUAL_FS);
  mesh_->create_field(mesh::field_enums::LHS_FS);
  if(mms_problem_!=Teuchos::null){
    mesh_->create_field(mesh::field_enums::EXACT_SOL_VECTOR_FS);
    mesh_->create_field(mesh::field_enums::IMAGE_PHI_FS);
    mesh_->create_field(mesh::field_enums::IMAGE_GRAD_PHI_FS);
  }
  mesh_->print_field_info();
  DICe::mesh::create_exodus_output_variable_names(mesh_);
  if(is_mixed_formulation()){//||global_formulation_==LEVENBERG_MARQUARDT){
    mesh_->create_field(mesh::field_enums::MIXED_LHS_FS);
    mesh_->create_field(mesh::field_enums::MIXED_RESIDUAL_FS);
    l_mesh_->create_field(mesh::field_enums::LAGRANGE_MULTIPLIER_FS);
    l_mesh_->create_field(mesh::field_enums::DISPLACEMENT_FS);
    if(mms_problem_!=Teuchos::null){
      l_mesh_->create_field(mesh::field_enums::EXACT_LAGRANGE_MULTIPLIER_FS);
    }
    l_mesh_->print_field_info();
    DICe::mesh::create_exodus_output_variable_names(l_mesh_);
  }

  DEBUG_MSG("Global_Algorithm::default_constructor_tasks(): using global formulation: " << to_string(global_formulation_));
  if(global_formulation_==HORN_SCHUNCK||global_formulation_==MIXED_HORN_SCHUNCK){
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::global_regularization_alpha),std::runtime_error,
      "Error, global_regularization_alpha must be defined");
    const scalar_t alpha = params->get<double>(DICe::global_regularization_alpha);
    alpha2_ = alpha*alpha;
    add_term(DIV_SYMMETRIC_STRAIN_REGULARIZATION);
    if(global_formulation_==MIXED_HORN_SCHUNCK){
      add_term(GRAD_LAGRANGE_MULTIPLIER);
      add_term(DIV_VELOCITY);
    }
    if(mms_problem_!=Teuchos::null){
        add_term(MMS_IMAGE_GRAD_TENSOR);
        add_term(MMS_IMAGE_TIME_FORCE);
        add_term(MMS_FORCE);
        add_term(DIRICHLET_DISPLACEMENT_BC);
        if(global_formulation_==MIXED_HORN_SCHUNCK){
          add_term(MMS_GRAD_LAGRANGE_MULTIPLIER);
        }
    }
    else{
      add_term(IMAGE_TIME_FORCE);
      add_term(IMAGE_GRAD_TENSOR);
      if(global_solver_==CG_SOLVER)
        add_term(SUBSET_DISPLACEMENT_IC);
      add_term(SUBSET_DISPLACEMENT_BC);
      //add_term(OPTICAL_FLOW_DISPLACEMENT_BC);
    }
  }
  else if(global_formulation_==LEHOUCQ_TURNER){
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::global_regularization_alpha),std::runtime_error,
      "Error, global_regularization_alpha must be defined");
    const scalar_t alpha = params->get<double>(DICe::global_regularization_alpha);
    alpha2_ = alpha*alpha;
    add_term(TIKHONOV_REGULARIZATION);
    add_term(GRAD_LAGRANGE_MULTIPLIER);
    add_term(DIV_VELOCITY);
    if(mms_problem_!=Teuchos::null){
        add_term(MMS_IMAGE_GRAD_TENSOR);
        add_term(MMS_IMAGE_TIME_FORCE);
        add_term(MMS_FORCE);
        add_term(DIRICHLET_DISPLACEMENT_BC);
    }
    else{
      add_term(IMAGE_TIME_FORCE);
      add_term(IMAGE_GRAD_TENSOR);
      if(global_solver_==CG_SOLVER){
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"CG solver is not appropriate for LEHOUCQ_TURNER");
      }
      add_term(SUBSET_DISPLACEMENT_BC);
    }
  }
  else if(global_formulation_==UNREGULARIZED){
    // TODO read regularizer lambda from the params to use in LSQR solver
//    const scalar_t alpha = params->get<double>(DICe::global_regularization_alpha);
//    alpha2_ = alpha; // NOTE not squared for this formulation
    if(mms_problem_!=Teuchos::null){
        add_term(MMS_IMAGE_GRAD_TENSOR);
        add_term(MMS_IMAGE_TIME_FORCE);
        add_term(MMS_FORCE);
        add_term(DIRICHLET_DISPLACEMENT_BC);
    }
    else{
      add_term(IMAGE_TIME_FORCE);
      add_term(IMAGE_GRAD_TENSOR);
      if(global_solver_==CG_SOLVER){
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"CG solver is not appropriate for UNREGULARIZED");
      }
    }
  }
  else if(global_formulation_==LEVENBERG_MARQUARDT){
    //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, this formulation has been deactivated because it is not stable");
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::global_regularization_alpha),std::runtime_error,
      "Error, global_regularization_alpha must be defined");
    const scalar_t alpha = params->get<double>(DICe::global_regularization_alpha);
    alpha2_ = alpha*alpha;
    add_term(TIKHONOV_REGULARIZATION);
    if(mms_problem_!=Teuchos::null){
        add_term(MMS_IMAGE_GRAD_TENSOR);
        add_term(MMS_IMAGE_TIME_FORCE);
        add_term(MMS_FORCE);
        add_term(DIRICHLET_DISPLACEMENT_BC);
    }
    else{
      add_term(IMAGE_TIME_FORCE);
      add_term(IMAGE_GRAD_TENSOR);
      if(global_solver_==CG_SOLVER){
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"CG solver is not appropriate for LEVENBERG_MARQUARDT");
      }
      add_term(SUBSET_DISPLACEMENT_BC);
      //add_term(OPTICAL_FLOW_DISPLACEMENT_BC);
    }
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid global_formulation " + to_string(global_formulation_));
  }
  std::set<Global_EQ_Term>::iterator it=eq_terms_.begin();
  std::set<Global_EQ_Term>::iterator it_end=eq_terms_.end();
  for(;it!=it_end;++it)
    DEBUG_MSG("Global_Algorithm::default_constructor_tasks(): active EQ term " << to_string(*it));
  DEBUG_MSG("Global_Algorithm::default_constructor_tasks(): alpha^2: " << alpha2_);
}

void
Global_Algorithm::pre_execution_tasks(){

  // initialize the solver:
  // set up the solver
  Teuchos::ParameterList belos_list;
  const int_t maxiters = 5000; // these are the max iterations of the belos solver not the nonlinear iterations
  const int_t numblk = (maxiters > 500) ? 500 : maxiters;
  const int_t maxrestarts = 10;
  const double conv_tol = 1.0E-8;
  std::string ortho("DGKS");
  belos_list.set( "Num Blocks", numblk);
  belos_list.set( "Maximum Iterations", maxiters);
  belos_list.set( "Convergence Tolerance", conv_tol); // Relative convergence tolerance requested
  belos_list.set( "Maximum Restarts", maxrestarts );  // Maximum number of restarts allowed
  int verbosity = Belos::Errors + Belos::Warnings + Belos::Debug + Belos::TimingDetails + Belos::FinalSummary + Belos::StatusTestDetails;
  belos_list.set( "Verbosity", verbosity );
  if(global_solver_!=LSQR_SOLVER)
    belos_list.set( "Output Style", Belos::Brief );
  belos_list.set( "Output Frequency", 1 );
// TODO  if(global_formulation_==UNREGULARIZED)
//    belos_list.set( "Lambda", alpha2_ );
  belos_list.set( "Orthogonalization", ortho); // Orthogonalization type

  /// linear problem for solve
  linear_problem_ = Teuchos::rcp(new Belos::LinearProblem<mv_scalar_type,vec_type,operator_type>());
  /// Belos solver
  if(global_solver_==CG_SOLVER){ // use Gmres for mms problems
    belos_solver_ = Teuchos::rcp( new Belos::BlockCGSolMgr<mv_scalar_type,vec_type,operator_type>
    (linear_problem_,Teuchos::rcp(&belos_list,false)));
  }
  else if(global_solver_==GMRES_SOLVER){
    belos_solver_ = Teuchos::rcp( new Belos::BlockGmresSolMgr<mv_scalar_type,vec_type,operator_type>
    (linear_problem_,Teuchos::rcp(&belos_list,false)));
  }
  else if(global_solver_==LSQR_SOLVER){
    belos_solver_ = Teuchos::rcp( new Belos::LSQRSolMgr<mv_scalar_type,vec_type,operator_type>
    (linear_problem_,Teuchos::rcp(&belos_list,false)));
  }
  else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, Unknown solver type.");
  }

  DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): Solver and linear problem have been initialized.");

  const int_t spa_dim = mesh_->spatial_dimension();
  matrix_service_ = Teuchos::rcp(new DICe::Matrix_Service(spa_dim));
  const int_t lagrange_size = is_mixed_formulation() ? l_mesh_->get_scalar_node_dist_map()->get_num_local_elements() : 0;
  matrix_service_->initialize_bc_register(mesh_->get_vector_node_dist_map()->get_num_local_elements(),
    mesh_->get_vector_node_overlap_map()->get_num_local_elements(),lagrange_size);

  if(has_term(DIRICHLET_DISPLACEMENT_BC)||
     has_term(OPTICAL_FLOW_DISPLACEMENT_BC)||
     has_term(SUBSET_DISPLACEMENT_BC)){
    DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): setting the boundary condition nodes");
    // set up the boundary condition nodes: FIXME this assumes there is only one node set
    //mesh->get_vector_node_dist_map()->describe();
    DICe::mesh::bc_set * bc_set = mesh_->get_node_bc_sets();
    const int_t boundary_node_set_id = 0; // dirichlet nodes
    const int_t num_bc_nodes = bc_set->find(boundary_node_set_id)->second.size();
    for(int_t i=0;i<num_bc_nodes;++i){
      const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
      //std::cout << " disp x condition on node " << node_gid << std::endl;
      bool is_local_node = mesh_->get_vector_node_dist_map()->is_node_global_elem((node_gid-1)*spa_dim+1);
      if(is_local_node){
        const int_t row_id = mesh_->get_vector_node_dist_map()->get_local_element((node_gid-1)*spa_dim+1);
        matrix_service_->register_row_bc(row_id);
      }
      int_t col_id = mesh_->get_vector_node_overlap_map()->get_local_element((node_gid-1)*spa_dim+1);
      matrix_service_->register_col_bc(col_id);
      //std::cout << " disp y condition on node " << node_gid << std::endl;
      is_local_node = mesh_->get_vector_node_dist_map()->is_node_global_elem((node_gid-1)*spa_dim+2);
      if(is_local_node){
        const int_t row_id = mesh_->get_vector_node_dist_map()->get_local_element((node_gid-1)*spa_dim+2);
        matrix_service_->register_row_bc(row_id);
      }
      col_id = mesh_->get_vector_node_overlap_map()->get_local_element((node_gid-1)*spa_dim+2);
      matrix_service_->register_col_bc(col_id);
    }
    if(is_mixed_formulation()){
      const int_t lagrange_node_set_id = 2; // lagrange multiplier nodes
      if(bc_set->find(lagrange_node_set_id)!=bc_set->end()){
        const int_t num_lagrange_nodes = bc_set->find(lagrange_node_set_id)->second.size();
        for(int_t i=0;i<num_lagrange_nodes;++i){
          const int_t node_gid = bc_set->find(lagrange_node_set_id)->second[i];
          //std::cout << " lagrange condition on node " << node_gid << std::endl;
          bool is_local_node = l_mesh_->get_scalar_node_dist_map()->is_node_global_elem(node_gid);
          if(is_local_node){
            const int_t row_id = l_mesh_->get_scalar_node_dist_map()->get_local_element(node_gid);
            matrix_service_->register_mixed_bc(row_id);
          }
        }
      }
    }

    // set up the neighbor list for each boundary node
    DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): creating the point cloud");
    /// pointer to the point cloud used for the neighbor searching
    MultiField & coords = *mesh_->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
    Teuchos::RCP<Point_Cloud<scalar_t> > point_cloud = Teuchos::rcp(new Point_Cloud<scalar_t>());
    std::vector<int_t> node_map(num_bc_nodes);
    point_cloud->pts.resize(num_bc_nodes);
    for(int_t i=0;i<num_bc_nodes;++i){
      const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
      point_cloud->pts[i].x = coords.global_value((node_gid-1)*spa_dim + 1);
      point_cloud->pts[i].y = coords.global_value((node_gid-1)*spa_dim + 2);
      point_cloud->pts[i].z = 0.0;
      node_map[i] = node_gid;
    }
    DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): building the kd-tree");
    /// pointer to the kd-tree used for searching
    Teuchos::RCP<my_kd_tree_t> kd_tree =
        Teuchos::rcp(new my_kd_tree_t(3 /*dim*/, *point_cloud.get(), nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ) );
    kd_tree->buildIndex();
    DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): setting up neighbors");
    const int_t num_neighbors = 5;
    scalar_t query_pt[3];
    for(int_t i=0;i<num_bc_nodes;++i){
      std::vector<size_t> ret_index(num_neighbors);
      std::vector<int_t> neighbors(num_neighbors);
      std::vector<scalar_t> out_dist_sqr(num_neighbors);
      const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
      query_pt[0] = point_cloud->pts[i].x;
      query_pt[1] = point_cloud->pts[i].y;
      query_pt[2] = point_cloud->pts[i].z;
      kd_tree->knnSearch(&query_pt[0], num_neighbors, &ret_index[0], &out_dist_sqr[0]);
      for(int_t j=0;j<num_neighbors;++j)
        neighbors[j] = node_map[ret_index[j]];
      bc_neighbors_.insert(std::pair<int_t,std::vector<int_t> >(node_gid,neighbors));
      bc_neighbor_distances_.insert(std::pair<int_t,std::vector<scalar_t> >(node_gid,out_dist_sqr));
    }
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
  }

  DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): Matrix service has been initialized.");
  is_initialized_ = true;

  // if this is not an mms problem, set up the images
  if(schema_){
    initialize_ref_image();
    set_def_image();
  }
}

Status_Flag
Global_Algorithm::execute(){
  if(!is_initialized_) pre_execution_tasks();

  const int_t spa_dim = mesh_->spatial_dimension();
  const int_t p_rank = mesh_->get_comm()->get_rank();
  const int_t relations_size = mesh_->max_num_node_relations();
  Teuchos::RCP<DICe::MultiField_Matrix> tangent;
  if(is_mixed_formulation()){
    tangent = Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_mixed_vector_node_dist_map(),relations_size));
  }
  else{
    tangent = Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_vector_node_dist_map(),relations_size));
  }

  DEBUG_MSG("Global_Algorithm::execute(): Tangent has been allocated.");

  Teuchos::RCP<MultiField> mixed_residual;
  Teuchos::RCP<MultiField> mixed_lhs;
  Teuchos::RCP<MultiField> lagrange_multiplier;
  Teuchos::RCP<MultiField> residual;
  Teuchos::RCP<MultiField> lhs;
  Teuchos::RCP<MultiField> exact_sol;
  Teuchos::RCP<MultiField> exact_lag;
  Teuchos::RCP<MultiField> image_phi;
  Teuchos::RCP<MultiField> image_grad_phi;
  const int_t mixed_global_offset = mesh_->get_vector_node_dist_map()->get_num_global_elements();
//  DICe::mesh::bc_set * bc_set = mesh_->get_node_bc_sets();
//  int_t p_node_gid = -1;
//  int_t p_node_row = -1;
//  if(bc_set->find(0)!=mesh_->get_node_bc_sets()->end()){
//    p_node_gid = bc_set->find(0)->second[0];
//    p_node_row = p_node_gid + mixed_global_offset;
//  }
//  DEBUG_MSG("Global_Algorithm::execute(): lagrange bc on node " << p_node_gid << " global row " << p_node_row);
  if(is_mixed_formulation()){
    DEBUG_MSG("Global_Algorithm::execute(): gathering mixed fields.");
    mixed_residual = mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS);
    mixed_lhs = mesh_->get_field(mesh::field_enums::MIXED_LHS_FS);
    lagrange_multiplier = l_mesh_->get_field(mesh::field_enums::LAGRANGE_MULTIPLIER_FS);
    if(mms_problem_!=Teuchos::null){
      exact_lag = l_mesh_->get_field(mesh::field_enums::EXACT_LAGRANGE_MULTIPLIER_FS);
    }
  }
  else{
    residual = mesh_->get_field(mesh::field_enums::RESIDUAL_FS);
    residual->put_scalar(0.0);
    lhs = mesh_->get_field(mesh::field_enums::LHS_FS);
  }
  MultiField & disp = *mesh_->get_field(mesh::field_enums::DISPLACEMENT_FS);
  disp.put_scalar(0.0);
  MultiField & coords = *mesh_->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  if(mms_problem_!=Teuchos::null){
    DEBUG_MSG("Global_Algorithm::execute(): gathering exact solution fields.");
    exact_sol = mesh_->get_field(mesh::field_enums::EXACT_SOL_VECTOR_FS);
    image_phi = mesh_->get_field(mesh::field_enums::IMAGE_PHI_FS);
    image_grad_phi = mesh_->get_field(mesh::field_enums::IMAGE_GRAD_PHI_FS);
  }
  DEBUG_MSG("Global_Algorithm::execute(): Assembling the linear system...");
  DEBUG_MSG("Global_Algorithm::execute(): mixed global offset: " << mixed_global_offset);

  // global iteration loop:
//  const int_t max_its = 1;
//  if(p_rank==0){
//    DEBUG_MSG("    iteration  residual_norm");
//  }
//  for(int_t iteration=0;iteration<max_its;++iteration){
//    if(p_rank==0){
//      DEBUG_MSG("    " << iteration << "      " << residual_norm);
//    }

    // assemble the distributed tangent matrix
    Teuchos::RCP<DICe::MultiField_Matrix> tangent_overlap;
    if(is_mixed_formulation()){
      tangent_overlap = Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_mixed_vector_node_overlap_map(),relations_size));
    }
    else{
      tangent_overlap = Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_vector_node_overlap_map(),relations_size));
    }
    // clear the jacobian values
    tangent_overlap->put_scalar(0.0);
    tangent->put_scalar(0.0);
    // allocate simple temp arrays for holding values of the local jacobain contributions
    // this will hopefully limit the number of calls to crsmatrix.insertGlobalValues
    std::map<int_t,Teuchos::Array<int_t> > col_id_array_map;
    std::map<int_t,Teuchos::Array<mv_scalar_type> > values_array_map;
    for(int_t i=0;i<tangent_overlap->num_local_rows();++i)
    {
      Teuchos::Array<int_t> id_array;
      Teuchos::Array<int_t> local_id_array;
      const int_t row_gid = is_mixed_formulation() ? mesh_->get_mixed_vector_node_overlap_map()->get_global_element(i) :
          mesh_->get_vector_node_overlap_map()->get_global_element(i);
      col_id_array_map.insert(std::pair<int_t,Teuchos::Array<int_t> >(row_gid,id_array));
      Teuchos::Array<mv_scalar_type> value_array;
      values_array_map.insert(std::pair<int_t,Teuchos::Array<mv_scalar_type> >(row_gid,value_array));
    }
    // establish the shape functions (using P2-P1 element for velocity pressure, or P2 velocity if no constraint):
    DICe::mesh::Shape_Function_Evaluator_Factory shape_func_eval_factory;
    Teuchos::RCP<DICe::mesh::Shape_Function_Evaluator> tri3_shape_func_evaluator =
        shape_func_eval_factory.create(DICe::mesh::TRI3);
    const int_t tri3_num_funcs = tri3_shape_func_evaluator->num_functions();
    scalar_t N3[tri3_num_funcs];
    scalar_t DN3[tri3_num_funcs*spa_dim];
    Teuchos::RCP<DICe::mesh::Shape_Function_Evaluator> tri6_shape_func_evaluator =
        shape_func_eval_factory.create(DICe::mesh::TRI6);
    const int_t tri6_num_funcs = tri6_shape_func_evaluator->num_functions();
    scalar_t N6[tri6_num_funcs];
    scalar_t DN6[tri6_num_funcs*spa_dim];
    int_t node_ids[tri6_num_funcs];
    scalar_t nodal_coords[tri6_num_funcs*spa_dim];
    scalar_t jac[spa_dim*spa_dim];
    scalar_t inv_jac[spa_dim*spa_dim];
    scalar_t J =0.0;
    //const int_t B_dim = 2*spa_dim - 1;
    //scalar_t B[B_dim*tri6_num_funcs*spa_dim];
    scalar_t elem_stiffness[tri6_num_funcs*spa_dim*tri6_num_funcs*spa_dim];
    scalar_t elem_div_stiffness[tri3_num_funcs*spa_dim*tri6_num_funcs];
    scalar_t elem_force[tri6_num_funcs*spa_dim];
    //scalar_t grad_phi[spa_dim];
    scalar_t x=0.0,y=0.0;

    // get the natural integration points for this element:
    const int_t integration_order = 6;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > gp_locs;
    Teuchos::ArrayRCP<scalar_t> gp_weights;
    int_t num_integration_points = -1;
    tri6_shape_func_evaluator->get_natural_integration_points(integration_order,gp_locs,gp_weights,num_integration_points);
    const int_t natural_coord_dim = gp_locs[0].size();
    scalar_t natural_coords[natural_coord_dim];

    // gather the OVERLAP fields
    Teuchos::RCP<MultiField> overlap_residual_ptr = is_mixed_formulation() ? mesh_->get_overlap_field(mesh::field_enums::MIXED_RESIDUAL_FS):
        mesh_->get_overlap_field(mesh::field_enums::RESIDUAL_FS);
    MultiField & overlap_residual = *overlap_residual_ptr;
    overlap_residual.put_scalar(0.0);
    Teuchos::RCP<MultiField> overlap_coords_ptr = mesh_->get_overlap_field(mesh::field_enums::INITIAL_COORDINATES_FS);
    MultiField & overlap_coords = *overlap_coords_ptr;
    Teuchos::ArrayRCP<const scalar_t> coords_values = overlap_coords.get_1d_view();

    scalar_t max_diag = 0.0;

    // element loop
    DICe::mesh::element_set::iterator elem_it = mesh_->get_element_set()->begin();
    DICe::mesh::element_set::iterator elem_end = mesh_->get_element_set()->end();
    for(;elem_it!=elem_end;++elem_it)
    {
      //std::cout << "ELEM: " << elem_it->get()->global_id() << std::endl;
      const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
      // compute the shape functions and derivatives for this element:
      for(int_t nd=0;nd<tri6_num_funcs;++nd){
        node_ids[nd] = connectivity[nd]->global_id();
        //std::cout << " gid " << node_ids[nd] << std::endl;
        for(int_t dim=0;dim<spa_dim;++dim){
          const int_t stride = nd*spa_dim + dim;
          nodal_coords[stride] = coords_values[connectivity[nd]->overlap_local_id()*spa_dim + dim];
          //std::cout << " node coords " << nodal_coords[stride] << std::endl;
        }
      }
      // clear the elem stiffness
      for(int_t i=0;i<tri6_num_funcs*spa_dim*tri6_num_funcs*spa_dim;++i)
        elem_stiffness[i] = 0.0;
      // clear the elem force
      for(int_t i=0;i<tri6_num_funcs*spa_dim;++i)
        elem_force[i] = 0.0;
      // clear the div stiffness storage
      for(int_t i=0;i<tri3_num_funcs*spa_dim*tri6_num_funcs;++i)
        elem_div_stiffness[i] = 0.0;

      // low-order gauss point loop:
      for(int_t gp=0;gp<num_integration_points;++gp){

        // isoparametric coords of the gauss point
        for(int_t dim=0;dim<natural_coord_dim;++dim){
          natural_coords[dim] = gp_locs[gp][dim];
          //std::cout << " natural coords " << dim << " " << natural_coords[dim] << std::endl;
        }
        // evaluate the shape functions and derivatives:
        tri6_shape_func_evaluator->evaluate_shape_functions(natural_coords,N6);
        tri6_shape_func_evaluator->evaluate_shape_function_derivatives(natural_coords,DN6);

        // evaluate the shape functions and derivatives:
        tri3_shape_func_evaluator->evaluate_shape_functions(natural_coords,N3);
        tri3_shape_func_evaluator->evaluate_shape_function_derivatives(natural_coords,DN3);

        // physical gp location
        x = 0.0; y=0.0;
        for(int_t i=0;i<tri6_num_funcs;++i){
          x += nodal_coords[i*spa_dim+0]*N6[i];
          y += nodal_coords[i*spa_dim+1]*N6[i];
        }
        //std::cout << " physical coords " << x << " " << y << std::endl;

        // compute the jacobian for this element:
        DICe::global::calc_jacobian(nodal_coords,DN6,jac,inv_jac,J,tri6_num_funcs,spa_dim);

        // stiffness terms

        // grad(phi) tensor_prod grad(phi)
        if(has_term(MMS_IMAGE_GRAD_TENSOR))
          mms_image_grad_tensor(mms_problem_,spa_dim,tri6_num_funcs,x,y,J,gp_weights[gp],N6,elem_stiffness);

        // alpha^2 * div(0.5*(grad(b) + grad(b)^T))
        if(has_term(DIV_SYMMETRIC_STRAIN_REGULARIZATION))
          div_symmetric_strain(spa_dim,tri6_num_funcs,alpha2_,J,gp_weights[gp],inv_jac,DN6,elem_stiffness);

        // alpha^2 * b
        if(has_term(TIKHONOV_REGULARIZATION))
          tikhonov_tensor(this,spa_dim,tri6_num_funcs,J,gp_weights[gp],N6,elem_stiffness);
          //lumped_tikhonov_tensor(this,spa_dim,tri6_num_funcs,J,gp_weights[gp],N6,elem_stiffness);

        // grad(phi) tensor_prod grad(phi)
        if(has_term(IMAGE_GRAD_TENSOR))
          image_grad_tensor(this,spa_dim,tri6_num_funcs,x,y,J,gp_weights[gp],N6,elem_stiffness);

        //std::cout << " elem stiffness " << std::endl;
        //for(int_t i=0;i<tri6_num_funcs*spa_dim;++i){
        //  for(int_t j=0;j<tri6_num_funcs*spa_dim;++j){
        //    std::cout << elem_stiffness[i*tri6_num_funcs*spa_dim + j] << " ";
        //  }
        //  std::cout << std::endl;
        //}

        // mixed formulation stiffness terms

        // grad(lambda)
        if(has_term(DIV_VELOCITY))
          div_velocity(spa_dim,tri3_num_funcs,tri6_num_funcs,J,gp_weights[gp],inv_jac,DN6,N3,elem_div_stiffness);

        //std::cout << "div elem stiffness " << std::endl;
        //for(int_t i=0;i<tri3_num_funcs;++i){
        //  for(int_t j=0;j<tri6_num_funcs*spa_dim;++j){
        //    std::cout << elem_div_stiffness[i*tri6_num_funcs*spa_dim + j] << " ";
        //  }
        //  std::cout << std::endl;
        //}

        // force terms

        // mms force
        if(has_term(MMS_FORCE))
          mms_force(mms_problem_,spa_dim,tri6_num_funcs,x,y,alpha2_,J,gp_weights[gp],N6,this->eq_terms(),elem_force);

        // d_dt(phi) * grad(phi)
        if(has_term(MMS_IMAGE_TIME_FORCE))
          mms_image_time_force(mms_problem_,spa_dim,tri6_num_funcs,x,y,J,gp_weights[gp],N6,elem_force);

        // d_dt(phi) * grad(phi)
        if(has_term(IMAGE_TIME_FORCE))
          image_time_force(this,spa_dim,tri6_num_funcs,x,y,J,gp_weights[gp],N6,elem_force);

      } // gp loop

//      if(has_term(IMAGE_TIME_FORCE)||has_term(IMAGE_GRAD_TENSOR)){
//        // get the high order natural integration points for this element:
//        const int_t high_order_integration_order = 6;
//        Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > high_order_gp_locs;
//        Teuchos::ArrayRCP<scalar_t> high_order_gp_weights;
//        int_t high_order_num_integration_points = -1;
//        tri6_shape_func_evaluator->get_natural_integration_points(high_order_integration_order,
//          high_order_gp_locs,
//          high_order_gp_weights,
//          high_order_num_integration_points);
//
//        // high-order gauss point loop:
//        for(int_t gp=0;gp<high_order_num_integration_points;++gp){
//
//          // isoparametric coords of the gauss point
//          for(int_t dim=0;dim<natural_coord_dim;++dim)
//            natural_coords[dim] = high_order_gp_locs[gp][dim];
//          //std::cout << " natural coords " << natural_coords[0] << " " << natural_coords[1] << std::endl;
//
//          // evaluate the shape functions and derivatives:
//          tri6_shape_func_evaluator->evaluate_shape_functions(natural_coords,N6);
//          tri6_shape_func_evaluator->evaluate_shape_function_derivatives(natural_coords,DN6);
//
//          // physical gp location
//          x = 0.0; y=0.0;
//          for(int_t i=0;i<tri6_num_funcs;++i){
//            x += nodal_coords[i*spa_dim+0]*N6[i];
//            y += nodal_coords[i*spa_dim+1]*N6[i];
//          }
//          //std::cout << " physical coords " << x << " " << y << std::endl;
//
//          // compute the jacobian for this element:
//          DICe::global::calc_jacobian(nodal_coords,DN6,jac,inv_jac,J,tri6_num_funcs,spa_dim);
//
//          // stiffness terms
//
//          // grad(phi) tensor-prod grad(phi)
//          if(has_term(IMAGE_GRAD_TENSOR))
//            image_grad_tensor(this,spa_dim,tri6_num_funcs,x,y,J,high_order_gp_weights[gp],N6,elem_stiffness);
//
//          // force terms
//
//          // d_dt(phi) * grad(phi)
//          if(has_term(IMAGE_TIME_FORCE))
//            image_time_force(this,spa_dim,tri6_num_funcs,x,y,J,high_order_gp_weights[gp],N6,elem_force);
//        }
//      }

      // assemble the global stiffness matrix
      for(int_t i=0;i<tri6_num_funcs;++i){
        for(int_t m=0;m<spa_dim;++m){
          // assemble the lagrange multiplier degrees of freedom
          if(is_mixed_formulation()){
            for(int_t j=0;j<tri3_num_funcs;++j){
              const int_t row = (node_ids[i]-1)*spa_dim + m + 1;
              const int_t col = node_ids[j] + mixed_global_offset;
              //std::cout << "row " << row << " col " << col << std::endl;
              const scalar_t value = elem_div_stiffness[j*tri6_num_funcs*spa_dim + i*spa_dim+m];
              const bool is_local_row_node =  mesh_->get_vector_node_dist_map()->is_node_global_elem(row); // using the non-mixed map because the row is a velocity row
              const bool row_is_bc_node = is_local_row_node ?
                  matrix_service_->is_row_bc(mesh_->get_vector_node_dist_map()->get_local_element(row)) : false; // same rationalle here
              const bool is_local_mixed_row_node =  l_mesh_->get_scalar_node_dist_map()->is_node_global_elem(node_ids[i]); // using the non-mixed map because the row is a velocity row
              const bool is_p_row = is_local_mixed_row_node ?
                  matrix_service_->is_mixed_bc(l_mesh_->get_scalar_node_dist_map()->get_local_element(node_ids[i])) : false;
              if(!row_is_bc_node&&!is_p_row){// && !col_is_bc_node){
                col_id_array_map.find(row)->second.push_back(col);
                values_array_map.find(row)->second.push_back(value);
                // transpose should be the same value
                const bool is_local_mixed_col_node =  l_mesh_->get_scalar_node_dist_map()->is_node_global_elem(node_ids[j]); // using the non-mixed map because the row is a velocity row
                const bool is_p_col = is_local_mixed_col_node ?
                    matrix_service_->is_mixed_bc(l_mesh_->get_scalar_node_dist_map()->get_local_element(node_ids[j])) : false;
                if(!is_p_col){
                  col_id_array_map.find(col)->second.push_back(row);
                  values_array_map.find(col)->second.push_back(value);
                }
              }
            } // tri3_num_funcs
          }
          // assemble the velocity degrees of freedom
          for(int_t j=0;j<tri6_num_funcs;++j){
            for(int_t n=0;n<spa_dim;++n){
              const int_t row = (node_ids[i]-1)*spa_dim + m + 1;
              const int_t col = (node_ids[j]-1)*spa_dim + n + 1;
              const scalar_t value = elem_stiffness[(i*spa_dim + m)*tri6_num_funcs*spa_dim + (j*spa_dim + n)];
              const bool is_local_row_node = mesh_->get_vector_node_dist_map()->is_node_global_elem(row);
              const bool row_is_bc_node = is_local_row_node ?
                  matrix_service_->is_row_bc(mesh_->get_vector_node_dist_map()->get_local_element(row)) : false;
              //const bool col_is_bc_node = matrix_service->is_col_bc(mesh_->get_vector_node_overlap_map()->get_local_element(col));
              //std::cout << " row " << row << " col " << col << " row_is_bc " << row_is_bc_node << " col_is_bc " << col_is_bc_node << std::endl;
              if(!row_is_bc_node){// && !col_is_bc_node){
                if(row==col && value > max_diag) max_diag=value;
                TEUCHOS_TEST_FOR_EXCEPTION(col_id_array_map.find(row)==col_id_array_map.end(),std::runtime_error,"Error, invalid col id");
                col_id_array_map.find(row)->second.push_back(col);
                values_array_map.find(row)->second.push_back(value);
              }
            } // spa dim
          } // num_funcs
        } // spa dim
      } // num_funcs

      // assemble the force terms
      for(int_t i=0;i<tri6_num_funcs;++i){
        //int_t nodex_local_id = connectivity[i]->overlap_local_id()*spa_dim;
        //int_t nodey_local_id = nodex_local_id + 1;
        //overlap_residual.local_value(nodex_local_id) += elem_force[i*spa_dim+0];
        //overlap_residual.local_value(nodey_local_id) += elem_force[i*spa_dim+1];
        int_t nodex_id = (connectivity[i]->global_id()-1)*spa_dim+1;
        int_t nodey_id = nodex_id + 1;
        if(is_mixed_formulation()){
          mixed_residual->global_value(nodex_id) += elem_force[i*spa_dim+0];
          mixed_residual->global_value(nodey_id) += elem_force[i*spa_dim+1];
        }
        else{
          residual->global_value(nodex_id) += elem_force[i*spa_dim+0];
          residual->global_value(nodey_id) += elem_force[i*spa_dim+1];
        }
      } // num_funcs
    }  // elem

    // export the overlap residual to the dist vector
    //mesh_->field_overlap_export(overlap_residual_ptr, mesh_::field_enums::RESIDUAL_FS, ADD);

    // add ones to the diagonal for kinematic velocity bc nodes:
    for(int_t i=0;i<mesh_->get_vector_node_overlap_map()->get_num_local_elements();++i)
    {
      if(matrix_service_->is_col_bc(i))
      {
        const int_t global_id = mesh_->get_vector_node_overlap_map()->get_global_element(i);
        col_id_array_map.find(global_id)->second.push_back(global_id);
        values_array_map.find(global_id)->second.push_back(1.0);//max_diag);
      }
    }
    /// lagrange multiplier bc
    if(is_mixed_formulation()){
      for(int_t i=0;i<l_mesh_->get_scalar_node_overlap_map()->get_num_local_elements();++i)
      {
        if(matrix_service_->is_mixed_bc(i))
        {
          const int_t global_id = l_mesh_->get_scalar_node_overlap_map()->get_global_element(i) + mixed_global_offset;
          col_id_array_map.find(global_id)->second.push_back(global_id);
          values_array_map.find(global_id)->second.push_back(1.0);//max_diag);
        }
      }
    }

    std::map<int_t,Teuchos::Array<int_t> >::iterator cmap_it = col_id_array_map.begin();
    std::map<int_t,Teuchos::Array<int_t> >::iterator cmap_end = col_id_array_map.end();
    for(;cmap_it!=cmap_end;++cmap_it)
    {
      int_t global_row = cmap_it->first;
      const Teuchos::Array<int_t> ids_array = cmap_it->second;
      const Teuchos::Array<mv_scalar_type> values_array = values_array_map.find(cmap_it->first)->second;
      if(ids_array.empty()) continue;
      // now do the insertGlobalValues calls:
      tangent_overlap->insert_global_values(global_row,ids_array,values_array);
    }
    tangent_overlap->fill_complete();

    if(is_mixed_formulation()){
      MultiField_Exporter exporter (*mesh_->get_mixed_vector_node_overlap_map(),*mesh_->get_mixed_vector_node_dist_map());
      tangent->do_export(tangent_overlap, exporter, ADD);
    }
    else{
      MultiField_Exporter exporter (*mesh_->get_vector_node_overlap_map(),*mesh_->get_vector_node_dist_map());
      tangent->do_export(tangent_overlap, exporter, ADD);
    }
    tangent->fill_complete();
    //tangent->describe();

    if(is_mixed_formulation())
      mixed_lhs->put_scalar(0.0);
    else
      lhs->put_scalar(0.0);

    // apply IC first so that it is overwritten by BCs
    if(has_term(SUBSET_DISPLACEMENT_IC)){
      TEUCHOS_TEST_FOR_EXCEPTION(is_mixed_formulation(),std::runtime_error,"Error ICs have not been implemented for mixed methods");
      for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
        int_t ix = i*2+0;
        int_t iy = i*2+1;
        scalar_t b_x = 0.0;
        scalar_t b_y = 0.0;
        const scalar_t x = coords.local_value(ix);
        const scalar_t y = coords.local_value(iy);
        // get the closest pixel to x and y
        int_t px = (int_t)x;
        if(x - (int_t)x >= 0.5) px++;
        int_t py = (int_t)y;
        if(y - (int_t)y >= 0.5) py++;
        const int_t subset_size = 39;
        subset_velocity(this,px,py,subset_size,b_x,b_y);
        lhs->local_value(ix) = b_x;
        lhs->local_value(iy) = b_y;
      }
    }
    if(has_term(DIRICHLET_DISPLACEMENT_BC)){
      if(mms_problem_!=Teuchos::null){
        // enforce the dirichlet boundary conditions
        for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
          int_t ix = i*2+0;
          int_t iy = i*2+1;
          scalar_t b_x = 0.0;
          scalar_t b_y = 0.0;
          const scalar_t x = coords.local_value(ix);
          const scalar_t y = coords.local_value(iy);
          mms_problem_->velocity(x,y,b_x,b_y);
          scalar_t phi = 0.0,d_phi_dt=0.0,grad_phi_x=0.0,grad_phi_y=0.0;
          mms_problem_->phi(x,y,phi);
          image_phi->local_value(i) = phi;
          mms_problem_->phi_derivatives(x,y,d_phi_dt,grad_phi_x,grad_phi_y);
          image_grad_phi->local_value(ix) = grad_phi_x;
          image_grad_phi->local_value(iy) = grad_phi_y;
          exact_sol->local_value(ix) = b_x;
          exact_sol->local_value(iy) = b_y;
          if(matrix_service_->is_col_bc(ix)){
            if(is_mixed_formulation()){
              mixed_residual->local_value(ix) = b_x;//*max_diag;
            }
            else{
              residual->local_value(ix) = b_x;//*max_diag;
            }
          }
          if(matrix_service_->is_col_bc(iy)){
            if(is_mixed_formulation()){
              mixed_residual->local_value(iy) = b_y;//*max_diag;
            }
            else{
              residual->local_value(iy) = b_y;//*max_diag;
            }
          }
        }// local element loop
        if(is_mixed_formulation()){
          for(int_t i=0;i<l_mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
            int_t ix = i*2+0;
            int_t iy = i*2+1;
            const scalar_t x = coords.local_value(ix);
            const scalar_t y = coords.local_value(iy);
            scalar_t l_out = 0.0;
            mms_problem_->lagrange(x,y,l_out);
            exact_lag->local_value(i) = l_out;
            if(matrix_service_->is_mixed_bc(i)){
              mixed_residual->local_value(i+mixed_global_offset) = l_out;
            }
          }
        }
      }
    }
    if(has_term(SUBSET_DISPLACEMENT_BC)){
      // compute all the displacements then average them to filter
      std::map<int_t,scalar_t> disp_x_bcs;
      std::map<int_t,scalar_t> disp_y_bcs;
      for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
        int_t ix = i*2+0;
        int_t iy = i*2+1;
        if(matrix_service_->is_col_bc(ix)||matrix_service_->is_col_bc(iy)){
          const int_t node_gid = mesh_->get_scalar_node_dist_map()->get_global_element(i);
          scalar_t b_x = 0.0;
          scalar_t b_y = 0.0;
          const scalar_t x = coords.local_value(ix);
          const scalar_t y = coords.local_value(iy);
          // get the closest pixel to x and y
          int_t px = (int_t)x;
          if(x - (int_t)x >= 0.5) px++;
          int_t py = (int_t)y;
          if(y - (int_t)y >= 0.5) py++;
          const int_t subset_size = 39;
          subset_velocity(this,px,py,subset_size,b_x,b_y);
          disp_x_bcs.insert(std::pair<int_t,scalar_t>(node_gid,b_x));
          disp_y_bcs.insert(std::pair<int_t,scalar_t>(node_gid,b_y));
        }
        // enforce lagrange multiplier bc
        if(is_mixed_formulation()){
          for(int_t i=0;i<l_mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
            if(matrix_service_->is_mixed_bc(i)){
              mixed_residual->local_value(i+mixed_global_offset) = 0.0;
            }
          }
        }
      }
      // now that all the displacements have been computed, average them
      // across neighbors to filter
      for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
        int_t ix = i*2+0;
        int_t iy = i*2+1;
        if(matrix_service_->is_col_bc(ix)||matrix_service_->is_col_bc(iy)){
          const int_t node_gid = mesh_->get_scalar_node_dist_map()->get_global_element(i);
          TEUCHOS_TEST_FOR_EXCEPTION(bc_neighbors_.find(node_gid)==bc_neighbors_.end(),std::runtime_error,
            "Error, missing node gid in neighbor map");
          const int_t num_neighbors = bc_neighbors_.find(node_gid)->second.size();
          scalar_t disp_x = 0.0;
          scalar_t disp_y = 0.0;
          for(int_t j=0;j<num_neighbors;++j){
            const int_t neighbor_id = bc_neighbors_.find(node_gid)->second[j];
            TEUCHOS_TEST_FOR_EXCEPTION(disp_x_bcs.find(neighbor_id)==disp_x_bcs.end(),std::runtime_error,
              "Error, missing neighbor id in set of displacement bcs");
            disp_x += disp_x_bcs.find(neighbor_id)->second;
            disp_y += disp_y_bcs.find(neighbor_id)->second;
          }
          disp_x /= num_neighbors;
          disp_y /= num_neighbors;
          if(matrix_service_->is_col_bc(ix)){
            if(is_mixed_formulation()){
              mixed_residual->local_value(ix) = disp_x;
            }
            else{
              residual->local_value(ix) = disp_x;
            }
          }
          if(matrix_service_->is_col_bc(iy)){
            if(is_mixed_formulation()){
              mixed_residual->local_value(iy) = disp_y;
            }
            else{
              residual->local_value(iy) = disp_y;
            }
          }
        } // is a col bc in x or y
      } // loop over all local nodes
    } // has term SUBSET_DISPLACEMENT_BC
    if(has_term(OPTICAL_FLOW_DISPLACEMENT_BC)){
      TEUCHOS_TEST_FOR_EXCEPTION(is_mixed_formulation(),std::runtime_error,"Error, optical flow bc has not been implemented for mixed formulations");
      for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
        int_t ix = i*2+0;
        int_t iy = i*2+1;
        if(matrix_service_->is_col_bc(ix)||matrix_service_->is_col_bc(iy)){
          scalar_t b_x = 0.0;
          scalar_t b_y = 0.0;
          const scalar_t x = coords.local_value(ix);
          const scalar_t y = coords.local_value(iy);
          // get the closest pixel to x and y
          int_t px = (int_t)x;
          if(x - (int_t)x >= 0.5) px++;
          int_t py = (int_t)y;
          if(y - (int_t)y >= 0.5) py++;
          // compute the optical flow here
          optical_flow_velocity(this,px,py,b_x,b_y);
          if(matrix_service_->is_col_bc(ix)){
            // get the coordinates of the node
            residual->local_value(ix) = b_x;
          }
          if(matrix_service_->is_col_bc(iy)){
            // get the coordinates of the node
            residual->local_value(iy) = b_y;
          }
        }
      }
    } // end has_term OPTICAL_FLOW_DISPLACEMENT_BC

//    if(is_mixed_formulation())
//      mixed_residual->describe();
//    else
//      residual->describe();

    DEBUG_MSG("Global_Algorithm::execute(): Solving the linear system...");

    // solve:
    linear_problem_->setHermitian(true);
    linear_problem_->setOperator(tangent->get());

    DEBUG_MSG("Global_Algorithm::execute(): Preconditioning");
    Preconditioner_Factory factory;
    Teuchos::RCP<Teuchos::ParameterList> plist = factory.parameter_list_for_ifpack();
    Teuchos::RCP<Ifpack_Preconditioner> Prec = factory.create (tangent->get(), plist);
    Teuchos::RCP<Belos::EpetraPrecOp> belosPrec = Teuchos::rcp( new Belos::EpetraPrecOp( Prec ) );
    linear_problem_->setLeftPrec( belosPrec );

    bool is_set = is_mixed_formulation() ? linear_problem_->setProblem(mixed_lhs->get(), mixed_residual->get()):
        linear_problem_->setProblem(lhs->get(), residual->get());
    TEUCHOS_TEST_FOR_EXCEPTION(!is_set, std::logic_error,
      "Error: Belos::LinearProblem::setProblem() failed to set up correctly.\n");
    Belos::ReturnType ret = belos_solver_->solve();
    if(ret != Belos::Converged && p_rank==0)
      std::cout << "*** WARNING: Belos linear solver did not converge!" << std::endl;
 // } // end iteration loop


  // if this is a mixed form split the lhs into displacement and lagrange multiplier fields
  if(is_mixed_formulation()){
    for(int_t i=0;i<mesh_->get_vector_node_dist_map()->get_num_local_elements();++i){
      const int_t gid = mesh_->get_vector_node_dist_map()->get_global_element(i);
      disp.global_value(gid) = mixed_lhs->global_value(gid);
    }
    for(int_t i=0;i<l_mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
      const int_t gid = l_mesh_->get_scalar_node_dist_map()->get_global_element(i);
      lagrange_multiplier->global_value(gid) = mixed_lhs->global_value(gid+mixed_global_offset);
    }
  }
  else{
    // upate the displacements
    disp.update(1.0,*lhs,1.0);
  }
  //disp.describe();

  // for levenberg marquart, copy the quadratic displacement field to the linear mesh
  if(is_mixed_formulation()){//||global_formulation_==LEVENBERG_MARQUARDT){
    TEUCHOS_TEST_FOR_EXCEPTION(l_mesh_==Teuchos::null,std::runtime_error,"Error, pointer should not be null here");
    Teuchos::RCP<MultiField> linear_disp = l_mesh_->get_field(mesh::field_enums::DISPLACEMENT_FS);
    Teuchos::RCP<MultiField> master_gids = l_mesh_->get_field(mesh::field_enums::MASTER_NODE_ID_FS);
    for(int_t i=0;i<l_mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
      // get the gid from the
      const int_t linear_gid = l_mesh_->get_scalar_node_dist_map()->get_global_element(i);
      const int_t ux_id = (linear_gid-1)*spa_dim + 1;
      const int_t uy_id = ux_id + 1;
      const int_t node_gid = master_gids->local_value(i);
      const int_t ux_master_id = (node_gid-1)*spa_dim + 1;
      const int_t uy_master_id = ux_master_id + 1;
      linear_disp->global_value(ux_id) = disp.global_value(ux_master_id);
      linear_disp->global_value(uy_id) = disp.global_value(uy_master_id);
    }
  }
  mesh_->print_field_stats();
  if(is_mixed_formulation())
    l_mesh_->print_field_stats();
  return CORRELATION_SUCCESSFUL;
}


void
Global_Algorithm::initialize_ref_image(){
  TEUCHOS_TEST_FOR_EXCEPTION(!schema_,std::runtime_error,"Error, schema must not be null for this method");
  //schema_->ref_img()->write("pre_ref_img.tif");

  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::compute_image_gradients,true);
  imgParams->set(DICe::gradient_method,FINITE_DIFFERENCE);
  ref_img_ = schema_->ref_img()->normalize(imgParams);
  //ref_img_->write("global_normalized_image.tif");
  // take the gradienst from the normalized image and make grad images for interpolation
  const int_t w = ref_img_->width();
  const int_t h = ref_img_->height();
  Teuchos::ArrayRCP<intensity_t> grad_ref_x(ref_img_->width()*ref_img_->height(),0.0);
  Teuchos::ArrayRCP<intensity_t> grad_ref_y(ref_img_->width()*ref_img_->height(),0.0);
  for(int_t y=0;y<h;++y){
    for(int_t x=0;x<w;++x){
      grad_ref_x[y*w+x] = ref_img_->grad_x(x,y);
      grad_ref_y[y*w+x] = ref_img_->grad_y(x,y);
    }
  }
  grad_x_img_ = Teuchos::rcp(new Image(w,h,grad_ref_x));
  //grad_x_img_->write("grad_x_img.tif");
  grad_y_img_ = Teuchos::rcp(new Image(w,h,grad_ref_y));
  //grad_y_img_->write("grad_y_img.tif");
}

void
Global_Algorithm::set_def_image(){
  //schema_->def_img()->write("pre_def_img.tif");

  TEUCHOS_TEST_FOR_EXCEPTION(!schema_,std::runtime_error,"Error, schema must not be null for this method");
  def_img_ = schema_->def_img()->normalize();
  //def_img_->write("global_normalized_def_image.tif");
}

void
Global_Algorithm::post_execution_tasks(const scalar_t & time_stamp){
  TEUCHOS_TEST_FOR_EXCEPTION(!is_initialized_,std::runtime_error,"Error, must be initialized");

  DEBUG_MSG("Writing the output file with time stamp: " << time_stamp);
  DICe::mesh::exodus_output_dump(mesh_,1,time_stamp);
  if(is_mixed_formulation()){//||global_formulation_==LEVENBERG_MARQUARDT){
    DEBUG_MSG("Writing the lagrange multiplier output file with time stamp: " << time_stamp);
    DICe::mesh::exodus_output_dump(l_mesh_,1,time_stamp);
  }
}

void
Global_Algorithm::evaluate_mms_error(scalar_t & error_bx,
  scalar_t & error_by,
  scalar_t & error_lambda){
  if(mms_problem_==Teuchos::null) return;

  error_bx = 0.0;
  error_by = 0.0;
  error_lambda = 0.0;

  MultiField & disp = *mesh_->get_field(mesh::field_enums::DISPLACEMENT_FS);
  MultiField & exact_sol = *mesh_->get_field(mesh::field_enums::EXACT_SOL_VECTOR_FS);

  for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
    int_t ix = i*2+0;
    int_t iy = i*2+1;
    const scalar_t b_x = disp.local_value(ix);
    const scalar_t b_y = disp.local_value(iy);
    const scalar_t b_exact_x = exact_sol.local_value(ix);
    const scalar_t b_exact_y = exact_sol.local_value(iy);
    error_bx += (b_exact_x - b_x)*(b_exact_x - b_x);
    error_by += (b_exact_y - b_y)*(b_exact_y - b_y);
  }
  error_bx = std::sqrt(error_bx);
  error_by = std::sqrt(error_by);

  if(is_mixed_formulation()){
    MultiField & l_exact_sol = *l_mesh_->get_field(mesh::field_enums::EXACT_LAGRANGE_MULTIPLIER_FS);
    MultiField & l_field = *l_mesh_->get_field(mesh::field_enums::LAGRANGE_MULTIPLIER_FS);
    for(int_t i=0;i<l_mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
      const scalar_t l = l_field.local_value(i);
      const scalar_t l_exact = l_exact_sol.local_value(i);
      error_lambda += (l_exact - l)*(l_exact - l);
    }
    error_lambda = std::sqrt(error_lambda);
  }

  DEBUG_MSG("MMS Error bx: " << error_bx << " by: " << error_by << " lambda: " << error_lambda);
}


}// end global namespace

}// End DICe Namespace
