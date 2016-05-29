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
  global_solver_(CG_SOLVER),
  num_image_integration_points_(20)
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
  global_solver_(CG_SOLVER),
  num_image_integration_points_(20)
{
  default_constructor_tasks(params);
}

void
Global_Algorithm::default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params){

  TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,
    "Error, params must be defined");

  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::global_formulation),std::runtime_error,"Error, parameter: global_formulation must be defined");
  global_formulation_ = params->get<Global_Formulation>(DICe::global_formulation);

  num_image_integration_points_ = params->get<int_t>(DICe::num_image_integration_points,20);

  DEBUG_MSG("Global_Algorithm::default_constructor_tasks(): num image integration points: " << num_image_integration_points_);
  TEUCHOS_TEST_FOR_EXCEPTION(num_image_integration_points_<=0,std::runtime_error,"Error, invalid num integration points");

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
  mesh_->create_field(mesh::field_enums::DISPLACEMENT_NM1_FS);
  mesh_->create_field(mesh::field_enums::RESIDUAL_FS);
  mesh_->create_field(mesh::field_enums::LHS_FS);
  // create the strain fields
  mesh_->create_field(mesh::field_enums::DU_DX_FS);
  mesh_->create_field(mesh::field_enums::DU_DY_FS);
  mesh_->create_field(mesh::field_enums::DV_DX_FS);
  mesh_->create_field(mesh::field_enums::DV_DY_FS);
  mesh_->create_field(mesh::field_enums::STRAIN_CONTRIBS_FS);
  mesh_->create_field(mesh::field_enums::GREEN_LAGRANGE_STRAIN_XX_FS);
  mesh_->create_field(mesh::field_enums::GREEN_LAGRANGE_STRAIN_YY_FS);
  mesh_->create_field(mesh::field_enums::GREEN_LAGRANGE_STRAIN_XY_FS);
  mesh_->create_field(mesh::field_enums::EXACT_SOL_VECTOR_FS); // zeros if not an mms problem
  // create the MMS fields if necessary
  if(mms_problem_!=Teuchos::null){
    mesh_->create_field(mesh::field_enums::DU_DX_EXACT_FS);
    mesh_->create_field(mesh::field_enums::DU_DY_EXACT_FS);
    mesh_->create_field(mesh::field_enums::DV_DX_EXACT_FS);
    mesh_->create_field(mesh::field_enums::DV_DY_EXACT_FS);
    mesh_->create_field(mesh::field_enums::IMAGE_PHI_FS);
    mesh_->create_field(mesh::field_enums::IMAGE_GRAD_PHI_FS);
  }
  mesh_->print_field_info();
  // create the mixed fields for the mixed formulations
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

  if(mms_problem_!=Teuchos::null){
    add_term(MMS_IMAGE_GRAD_TENSOR);
    add_term(MMS_IMAGE_TIME_FORCE);
    add_term(MMS_FORCE);
    add_term(MMS_DIRICHLET_DISPLACEMENT_BC);
  }
  else{
    add_term(IMAGE_TIME_FORCE);
    add_term(IMAGE_GRAD_TENSOR);
    add_term(DIRICHLET_DISPLACEMENT_BC);
    add_term(SUBSET_DISPLACEMENT_BC);
  }

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
    if(global_formulation_==MIXED_HORN_SCHUNCK && mms_problem_!=Teuchos::null){
      add_term(MMS_GRAD_LAGRANGE_MULTIPLIER);
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
    if(global_solver_==CG_SOLVER){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"CG solver is not appropriate for LEHOUCQ_TURNER");
    }
  }
  else if(global_formulation_==UNREGULARIZED){
    // TODO read regularizer lambda from the params to use in LSQR solver
    //      const scalar_t alpha = params->get<double>(DICe::global_regularization_alpha);
    //      alpha2_ = alpha; // NOTE not squared for this formulation
    if(global_solver_==CG_SOLVER){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"CG solver is not appropriate for UNREGULARIZED");
    }
  }
  else if(global_formulation_==LEVENBERG_MARQUARDT){
    //TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, this formulation has been deactivated because it is not stable");
    TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::global_regularization_alpha),std::runtime_error,
      "Error, global_regularization_alpha must be defined");
    const scalar_t alpha = params->get<double>(DICe::global_regularization_alpha);
    alpha2_ = alpha*alpha;
    add_term(TIKHONOV_REGULARIZATION);
    if(global_solver_==CG_SOLVER){
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"CG solver is not appropriate for LEVENBERG_MARQUARDT");
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

//  const int_t spa_dim = mesh_->spatial_dimension();
  bc_manager_ = Teuchos::rcp(new BC_Manager(this));

  if(has_term(DIRICHLET_DISPLACEMENT_BC)){
    if(mesh_->bc_defs()->size()>0){
      bc_manager_->create_bc(DIRICHLET_DISPLACEMENT_BC,is_mixed_formulation());
    }
  }
  if(has_term(SUBSET_DISPLACEMENT_BC)){
    if(mesh_->bc_defs()->size()>0){
      bc_manager_->create_bc(SUBSET_DISPLACEMENT_BC,is_mixed_formulation());
    }
  }
  if(has_term(MMS_DIRICHLET_DISPLACEMENT_BC)){
    bc_manager_->create_bc(MMS_DIRICHLET_DISPLACEMENT_BC,is_mixed_formulation());
  }

  DEBUG_MSG("Global_Algorithm::pre_execution_tasks(): BC_Manager has been initialized.");
  is_initialized_ = true;

  // if this is not an mms problem, set up the images
  if(schema_){
    initialize_ref_image();
    set_def_image();
  }
}

Teuchos::RCP<DICe::MultiField_Matrix>
Global_Algorithm::compute_tangent(){

  DEBUG_MSG("Global_Algorithm::compute_tangent(): Computing the tangent matrix");
  const int_t spa_dim = mesh_->spatial_dimension();
  const int_t relations_size = mesh_->max_num_node_relations();
  Teuchos::RCP<DICe::MultiField_Matrix> tangent;
  if(is_mixed_formulation())
    tangent = Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_mixed_vector_node_dist_map(),relations_size));
  else
    tangent = Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_vector_node_dist_map(),relations_size));
  DEBUG_MSG("Global_Algorithm::execute(): Tangent has been allocated.");
  const int_t mgo = mixed_global_offset();
  DEBUG_MSG("Global_Algorithm::compute_tangent(): mixed global offset: " << mgo);

  // assemble the distributed tangent matrix
  Teuchos::RCP<DICe::MultiField_Matrix> tangent_overlap;
  if(is_mixed_formulation())
    tangent_overlap = Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_mixed_vector_node_overlap_map(),relations_size));
  else
    tangent_overlap = Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_vector_node_overlap_map(),relations_size));

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
  scalar_t elem_stiffness[tri6_num_funcs*spa_dim*tri6_num_funcs*spa_dim];
  scalar_t elem_div_stiffness[tri3_num_funcs*spa_dim*tri6_num_funcs];
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

  const int_t image_integration_order = num_image_integration_points_;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > image_gp_locs;
  Teuchos::ArrayRCP<scalar_t> image_gp_weights;
  int_t num_image_integration_points = -1;
  tri2d_nonexact_integration_points(image_integration_order,image_gp_locs,image_gp_weights,num_image_integration_points);

  // gather the OVERLAP fields
  Teuchos::RCP<MultiField> overlap_coords_ptr = mesh_->get_overlap_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  MultiField & overlap_coords = *overlap_coords_ptr;
  Teuchos::ArrayRCP<const scalar_t> coords_values = overlap_coords.get_1d_view();

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

      // physical gp location
      x = 0.0; y=0.0;
      for(int_t i=0;i<tri6_num_funcs;++i){
        x += nodal_coords[i*spa_dim+0]*N6[i];
        y += nodal_coords[i*spa_dim+1]*N6[i];
      }
      //std::cout << " physical coords " << x << " " << y << std::endl;

      // compute the jacobian for this element:
      DICe::global::calc_jacobian(nodal_coords,DN6,jac,inv_jac,J,tri6_num_funcs,spa_dim);

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

      // mixed formulation stiffness terms

      // grad(lambda)
      if(has_term(DIV_VELOCITY))
        div_velocity(spa_dim,tri3_num_funcs,tri6_num_funcs,J,gp_weights[gp],inv_jac,DN6,N3,elem_div_stiffness);

    } // gp loop

    // low-order gauss point loop:
    for(int_t gp=0;gp<num_image_integration_points;++gp){

      // isoparametric coords of the gauss point
      for(int_t dim=0;dim<natural_coord_dim;++dim){
        natural_coords[dim] = image_gp_locs[gp][dim];
        //std::cout << " natural coords " << dim << " " << natural_coords[dim] << std::endl;
      }
      // evaluate the shape functions and derivatives:
      tri6_shape_func_evaluator->evaluate_shape_functions(natural_coords,N6);
      tri6_shape_func_evaluator->evaluate_shape_function_derivatives(natural_coords,DN6);

      // physical gp location
      x = 0.0; y=0.0;
      for(int_t i=0;i<tri6_num_funcs;++i){
        x += nodal_coords[i*spa_dim+0]*N6[i];
        y += nodal_coords[i*spa_dim+1]*N6[i];
      }
      //std::cout << " physical coords " << x << " " << y << std::endl;

      // compute the jacobian for this element:
      DICe::global::calc_jacobian(nodal_coords,DN6,jac,inv_jac,J,tri6_num_funcs,spa_dim);

      // grad(phi) tensor_prod grad(phi)
      if(has_term(IMAGE_GRAD_TENSOR))
        image_grad_tensor(this,spa_dim,tri6_num_funcs,x,y,J,image_gp_weights[gp],N6,elem_stiffness);

    } // image gp loop

    //DEBUG_MSG("Global_Algorithm::compute_tangent(): Assembling the tangent matrix");
    // assemble the global stiffness matrix
    for(int_t i=0;i<tri6_num_funcs;++i){
      for(int_t m=0;m<spa_dim;++m){
        // assemble the lagrange multiplier degrees of freedom
        if(is_mixed_formulation()){
          for(int_t j=0;j<tri3_num_funcs;++j){
            const int_t row = (node_ids[i]-1)*spa_dim + m + 1;
            const int_t col = node_ids[j] + mgo;
            //std::cout << "row " << row << " col " << col << std::endl;
            const scalar_t value = elem_div_stiffness[j*tri6_num_funcs*spa_dim + i*spa_dim+m];
            const bool is_local_row_node =  mesh_->get_vector_node_dist_map()->is_node_global_elem(row); // using the non-mixed map because the row is a velocity row
            const bool row_is_bc_node = is_local_row_node ?
                bc_manager_->is_row_bc(mesh_->get_vector_node_dist_map()->get_local_element(row)) : false; // same rationalle here
            const bool is_local_mixed_row_node =  l_mesh_->get_scalar_node_dist_map()->is_node_global_elem(node_ids[i]); // using the non-mixed map because the row is a velocity row
            const bool is_p_row = is_local_mixed_row_node ?
                bc_manager_->is_mixed_bc(l_mesh_->get_scalar_node_dist_map()->get_local_element(node_ids[i])) : false;
            if(!row_is_bc_node&&!is_p_row){// && !col_is_bc_node){
              col_id_array_map.find(row)->second.push_back(col);
              values_array_map.find(row)->second.push_back(value);
              // transpose should be the same value
              const bool is_local_mixed_col_node =  l_mesh_->get_scalar_node_dist_map()->is_node_global_elem(node_ids[j]); // using the non-mixed map because the row is a velocity row
              const bool is_p_col = is_local_mixed_col_node ?
                  bc_manager_->is_mixed_bc(l_mesh_->get_scalar_node_dist_map()->get_local_element(node_ids[j])) : false;
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
                bc_manager_->is_row_bc(mesh_->get_vector_node_dist_map()->get_local_element(row)) : false;
            if(!row_is_bc_node){// && !col_is_bc_node){
              TEUCHOS_TEST_FOR_EXCEPTION(col_id_array_map.find(row)==col_id_array_map.end(),std::runtime_error,"Error, invalid col id");
              col_id_array_map.find(row)->second.push_back(col);
              values_array_map.find(row)->second.push_back(value);
            }
          } // spa dim
        } // num_funcs
      } // spa dim
    } // num_funcs
  }  // elem
  // add ones to the diagonal for kinematic velocity bc nodes:
  for(int_t i=0;i<mesh_->get_vector_node_overlap_map()->get_num_local_elements();++i){
    if(bc_manager_->is_col_bc(i)){
      const int_t global_id = mesh_->get_vector_node_overlap_map()->get_global_element(i);
      col_id_array_map.find(global_id)->second.push_back(global_id);
      values_array_map.find(global_id)->second.push_back(1.0);
    }
  }
  /// lagrange multiplier bc
  if(is_mixed_formulation()){
    for(int_t i=0;i<l_mesh_->get_scalar_node_overlap_map()->get_num_local_elements();++i){
      if(bc_manager_->is_mixed_bc(i)){
        const int_t global_id = l_mesh_->get_scalar_node_overlap_map()->get_global_element(i) + mgo;
        col_id_array_map.find(global_id)->second.push_back(global_id);
        values_array_map.find(global_id)->second.push_back(1.0);
      }
    }
  }
  std::map<int_t,Teuchos::Array<int_t> >::iterator cmap_it = col_id_array_map.begin();
  std::map<int_t,Teuchos::Array<int_t> >::iterator cmap_end = col_id_array_map.end();
  for(;cmap_it!=cmap_end;++cmap_it){
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
  return tangent;
}

scalar_t
Global_Algorithm::compute_residual(const bool use_fixed_point){

  DEBUG_MSG("Global_Algorithm::compute_residual(): computing the residual.");
  const int_t spa_dim = mesh_->spatial_dimension();

  Teuchos::RCP<MultiField> residual;
  if(is_mixed_formulation())
    residual = mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS);
  else
    residual = mesh_->get_field(mesh::field_enums::RESIDUAL_FS);
  residual->put_scalar(0.0);

  // establish the shape functions (using P2-P1 element for velocity pressure, or P2 velocity if no constraint):
  DICe::mesh::Shape_Function_Evaluator_Factory shape_func_eval_factory;
  Teuchos::RCP<DICe::mesh::Shape_Function_Evaluator> tri3_shape_func_evaluator =
      shape_func_eval_factory.create(DICe::mesh::TRI3);
  const int_t tri3_num_funcs = tri3_shape_func_evaluator->num_functions();
  scalar_t N3[tri3_num_funcs];
  Teuchos::RCP<DICe::mesh::Shape_Function_Evaluator> tri6_shape_func_evaluator =
      shape_func_eval_factory.create(DICe::mesh::TRI6);
  const int_t tri6_num_funcs = tri6_shape_func_evaluator->num_functions();
  scalar_t N6[tri6_num_funcs];
  scalar_t DN6[tri6_num_funcs*spa_dim];
  scalar_t nodal_coords[tri6_num_funcs*spa_dim];
  scalar_t nodal_disp[tri6_num_funcs*spa_dim];
  scalar_t jac[spa_dim*spa_dim];
  scalar_t inv_jac[spa_dim*spa_dim];
  scalar_t J =0.0;
  scalar_t elem_force[tri6_num_funcs*spa_dim];
  scalar_t x=0.0,y=0.0,bx=0.0,by=0.0;

  // get the natural integration points for this element:
  const int_t integration_order = 6;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > gp_locs;
  Teuchos::ArrayRCP<scalar_t> gp_weights;
  int_t num_integration_points = -1;
  tri6_shape_func_evaluator->get_natural_integration_points(integration_order,gp_locs,gp_weights,num_integration_points);
  const int_t natural_coord_dim = gp_locs[0].size();
  scalar_t natural_coords[natural_coord_dim];

  const int_t image_integration_order = num_image_integration_points_;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > image_gp_locs;
  Teuchos::ArrayRCP<scalar_t> image_gp_weights;
  int_t num_image_integration_points = -1;
  tri2d_nonexact_integration_points(image_integration_order,image_gp_locs,image_gp_weights,num_image_integration_points);

  // gather the OVERLAP fields
//  Teuchos::RCP<MultiField> overlap_residual_ptr = is_mixed_formulation() ? mesh_->get_overlap_field(mesh::field_enums::MIXED_RESIDUAL_FS):
//      mesh_->get_overlap_field(mesh::field_enums::RESIDUAL_FS);
//  MultiField & overlap_residual = *overlap_residual_ptr;
//  overlap_residual.put_scalar(0.0);
  Teuchos::RCP<MultiField> overlap_coords_ptr = mesh_->get_overlap_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  MultiField & overlap_coords = *overlap_coords_ptr;
  Teuchos::ArrayRCP<const scalar_t> coords_values = overlap_coords.get_1d_view();
  Teuchos::RCP<MultiField> overlap_disp_ptr = mesh_->get_overlap_field(mesh::field_enums::DISPLACEMENT_FS);
  MultiField & overlap_disp = *overlap_disp_ptr;
  Teuchos::ArrayRCP<const scalar_t> disp_values = overlap_disp.get_1d_view();

  // element loop
  DICe::mesh::element_set::iterator elem_it = mesh_->get_element_set()->begin();
  DICe::mesh::element_set::iterator elem_end = mesh_->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it)
  {
    //std::cout << "ELEM: " << elem_it->get()->global_id() << std::endl;
    const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
    // compute the shape functions and derivatives for this element:
    for(int_t nd=0;nd<tri6_num_funcs;++nd){
      for(int_t dim=0;dim<spa_dim;++dim){
        const int_t stride = nd*spa_dim + dim;
        nodal_coords[stride] = coords_values[connectivity[nd]->overlap_local_id()*spa_dim + dim];
        nodal_disp[stride] = disp_values[connectivity[nd]->overlap_local_id()*spa_dim + dim];
        //std::cout << " node coords " << nodal_coords[stride] << std::endl;
      }
    }
    // clear the elem force
    for(int_t i=0;i<tri6_num_funcs*spa_dim;++i)
      elem_force[i] = 0.0;

    if(mms_problem_!=Teuchos::null){
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

        // physical gp location
        x = 0.0; y=0.0;
        for(int_t i=0;i<tri6_num_funcs;++i){
          x += nodal_coords[i*spa_dim+0]*N6[i];
          y += nodal_coords[i*spa_dim+1]*N6[i];
        }
        //std::cout << " physical coords " << x << " " << y << std::endl;

        // compute the jacobian for this element:
        DICe::global::calc_jacobian(nodal_coords,DN6,jac,inv_jac,J,tri6_num_funcs,spa_dim);

        // mms force
        if(has_term(MMS_FORCE))
          mms_force(mms_problem_,spa_dim,tri6_num_funcs,x,y,alpha2_,J,gp_weights[gp],N6,this->eq_terms(),elem_force);

        // d_dt(phi) * grad(phi)
        if(has_term(MMS_IMAGE_TIME_FORCE))
          mms_image_time_force(mms_problem_,spa_dim,tri6_num_funcs,x,y,J,gp_weights[gp],N6,elem_force);

      } // gp loop
    } // has mms_problem

    // low-order gauss point loop:
    for(int_t gp=0;gp<num_image_integration_points;++gp){

      // isoparametric coords of the gauss point
      for(int_t dim=0;dim<natural_coord_dim;++dim){
        natural_coords[dim] = image_gp_locs[gp][dim];
        //std::cout << " natural coords " << dim << " " << natural_coords[dim] << std::endl;
      }
      // evaluate the shape functions and derivatives:
      tri6_shape_func_evaluator->evaluate_shape_functions(natural_coords,N6);
      tri6_shape_func_evaluator->evaluate_shape_function_derivatives(natural_coords,DN6);

      // physical gp location
      x = 0.0; y=0.0;
      bx = 0.0; by=0.0;
      for(int_t i=0;i<tri6_num_funcs;++i){
        x += nodal_coords[i*spa_dim+0]*N6[i];
        y += nodal_coords[i*spa_dim+1]*N6[i];
        if(use_fixed_point){
          bx += nodal_disp[i*spa_dim+0]*N6[i];
          by += nodal_disp[i*spa_dim+1]*N6[i];
        }
      }
      //std::cout << " x " << x << " y " << y <<  " bx " << bx << " by " << by << std::endl;
      //std::cout << " physical coords " << x << " " << y << std::endl;

      // compute the jacobian for this element:
      DICe::global::calc_jacobian(nodal_coords,DN6,jac,inv_jac,J,tri6_num_funcs,spa_dim);

      // d_dt(phi) * grad(phi)
      if(has_term(IMAGE_TIME_FORCE))
        image_time_force(this,spa_dim,tri6_num_funcs,x,y,bx,by,J,image_gp_weights[gp],N6,elem_force);

    } // image gp loop

    // assemble the force terms
    // (note: no force terms for lagrange multiplier...so assembly is the same if mixed or not)
    for(int_t i=0;i<tri6_num_funcs;++i){
      //int_t nodex_local_id = connectivity[i]->overlap_local_id()*spa_dim;
      //int_t nodey_local_id = nodex_local_id + 1;
      //overlap_residual.local_value(nodex_local_id) += elem_force[i*spa_dim+0];
      //overlap_residual.local_value(nodey_local_id) += elem_force[i*spa_dim+1];
      for(int_t dim=0;dim<spa_dim;++dim){
        int_t row = (connectivity[i]->global_id()-1)*spa_dim+dim+1;
        //const bool is_local_row_node =  mesh_->get_vector_node_dist_map()->is_node_global_elem(row); // using the non-mixed map because the row is a velocity row
        //const bool row_is_bc_node = is_local_row_node ?
        //    bc_manager_->is_row_bc(mesh_->get_vector_node_dist_map()->get_local_element(row)) : false; // same rationalle here
        residual->global_value(row) += elem_force[i*spa_dim+dim];
      }
    } // num_funcs
  }  // elem

  // export the overlap residual to the dist vector
  //mesh_->field_overlap_export(overlap_residual_ptr, mesh_::field_enums::RESIDUAL_FS, ADD);
  return residual->norm();
  //residual->describe();
}


Status_Flag
Global_Algorithm::execute(){
  DEBUG_MSG("Global_Algorithm::execute(): method called");
  if(!is_initialized_) pre_execution_tasks();

  const int_t p_rank = mesh_->get_comm()->get_rank();

  Teuchos::RCP<MultiField> residual;
  Teuchos::RCP<MultiField> lhs;
  Teuchos::RCP<MultiField> lagrange_multiplier;
  if(is_mixed_formulation()){
    residual = mesh_->get_field(mesh::field_enums::MIXED_RESIDUAL_FS);
    lhs = mesh_->get_field(mesh::field_enums::MIXED_LHS_FS);
    lagrange_multiplier = l_mesh_->get_field(mesh::field_enums::LAGRANGE_MULTIPLIER_FS);
    lagrange_multiplier->put_scalar(0.0);
  }
  else{
    residual = mesh_->get_field(mesh::field_enums::RESIDUAL_FS);
    lhs = mesh_->get_field(mesh::field_enums::LHS_FS);
  }
  Teuchos::RCP<MultiField> disp = mesh_->get_field(mesh::field_enums::DISPLACEMENT_FS);
  Teuchos::RCP<MultiField> disp_nm1 = mesh_->get_field(mesh::field_enums::DISPLACEMENT_NM1_FS);
  disp->put_scalar(0.0);
  disp_nm1->put_scalar(0.0);

  Teuchos::RCP<DICe::MultiField_Matrix> tangent = compute_tangent();
  linear_problem_->setHermitian(true);
  linear_problem_->setOperator(tangent->get());

  // if this is a real problem (not MMS) use the lagrangian coordinates
  // and use a fixed point iteration loop rather than a direct solve
  const bool use_fixed_point = mms_problem_==Teuchos::null;
  DEBUG_MSG("Global_Algorithm::execute: use_fixed_point: " << use_fixed_point);
  const int_t max_its = 25;
  const scalar_t update_tol = 1.0E-8;
  const scalar_t residual_tol = 1.0E-8;
  int_t it=0;
  for(;it<=max_its;++it){
    // clear the left hand side
    lhs->put_scalar(0.0);

    const scalar_t resid_norm = compute_residual(use_fixed_point);

    // apply the boundary conditions
    bc_manager_->apply_bcs(it==0);

    if(resid_norm < residual_tol){
      DEBUG_MSG("Iteration: " << it << " residual norm: " << resid_norm);
      DEBUG_MSG("Global_Algorithm::execute(): * * * convergence successful * * *");
      DEBUG_MSG("Global_Algorithm::execute(): criteria: residual_norm < tol (" << residual_tol << ")");
      break;
    }

    // solve:
    DEBUG_MSG("Global_Algorithm::execute(): Solving the linear system...");
    DEBUG_MSG("Global_Algorithm::execute(): Preconditioning");
    Preconditioner_Factory factory;
    Teuchos::RCP<Teuchos::ParameterList> plist = factory.parameter_list_for_ifpack();
    Teuchos::RCP<Ifpack_Preconditioner> Prec = factory.create (tangent->get(), plist);
    Teuchos::RCP<Belos::EpetraPrecOp> belosPrec = Teuchos::rcp( new Belos::EpetraPrecOp( Prec ) );
    linear_problem_->setLeftPrec( belosPrec );
    bool is_set = linear_problem_->setProblem(lhs->get(), residual->get());
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
        disp->global_value(gid) += lhs->global_value(gid);
      }
      for(int_t i=0;i<l_mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
        const int_t gid = l_mesh_->get_scalar_node_dist_map()->get_global_element(i);
        lagrange_multiplier->global_value(gid) += lhs->global_value(gid+mixed_global_offset());
      }
    }
    else{
      // upate the displacements
      disp->update(1.0,*lhs,1.0);
    }

    //disp.describe();
    //write out the stats on the displacement field
    scalar_t lhs_min_x = 0.0, lhs_min_y = 0.0;
    scalar_t lhs_max_x = 0.0, lhs_max_y = 0.0;
    scalar_t lhs_avg_x = 0.0, lhs_avg_y = 0.0;
    scalar_t lhs_std_dev_x = 0.0, lhs_std_dev_y = 0.0;
    mesh_->field_stats(DICe::mesh::field_enums::LHS_FS,lhs_min_x,lhs_max_x,lhs_avg_x,lhs_std_dev_x,0);
    mesh_->field_stats(DICe::mesh::field_enums::LHS_FS,lhs_min_y,lhs_max_y,lhs_avg_y,lhs_std_dev_y,1);
    DEBUG_MSG("DISPLACEMENT UPDATE: min_x " << lhs_min_x << " max_x " << lhs_max_x << " avg_x " << lhs_avg_x << " std_dev_x " << lhs_std_dev_x);
    DEBUG_MSG("DISPLACEMENT UPDATE: min_y " << lhs_min_y << " max_y " << lhs_max_y << " avg_y " << lhs_avg_y << " std_dev_y " << lhs_std_dev_y);
    scalar_t disp_min_x = 0.0, disp_min_y = 0.0;
    scalar_t disp_max_x = 0.0, disp_max_y = 0.0;
    scalar_t disp_avg_x = 0.0, disp_avg_y = 0.0;
    scalar_t disp_std_dev_x = 0.0, disp_std_dev_y = 0.0;
    mesh_->field_stats(DICe::mesh::field_enums::DISPLACEMENT_FS,disp_min_x,disp_max_x,disp_avg_x,disp_std_dev_x,0);
    mesh_->field_stats(DICe::mesh::field_enums::DISPLACEMENT_FS,disp_min_y,disp_max_y,disp_avg_y,disp_std_dev_y,1);
    DEBUG_MSG("DISPLACEMENT: min_x " << disp_min_x << " max_x " << disp_max_x << " avg_x " << disp_avg_x << " std_dev_x " << disp_std_dev_x);
    DEBUG_MSG("DISPLACEMENT: min_y " << disp_min_y << " max_y " << disp_max_y << " avg_y " << disp_avg_y << " std_dev_y " << disp_std_dev_y);

    if(!use_fixed_point){
      DEBUG_MSG("Global_Algorithm::execute(): * * * convergence successful * * *");
      DEBUG_MSG("Global_Algorithm::execute(): criteria: single iteration required");
      break;
    }

    // compute the change in disp from the last solution:
    const scalar_t disp_norm = disp->norm();
    const scalar_t delta_disp_norm = disp->norm(disp_nm1);
    DEBUG_MSG("Iteration: " << it << " residual norm: " << resid_norm
      << " disp norm: " << disp_norm << " disp update norm: " << delta_disp_norm);
    if(delta_disp_norm < update_tol){
      DEBUG_MSG("Global_Algorithm::execute(): * * * convergence successful * * *");
      DEBUG_MSG("Global_Algorithm::execute(): criteria: delta_disp_norm < tol (" << update_tol << ")");
      break;
    }
    // copy the displacement solution to state n-1
    disp_nm1->update(1.0,*disp,0.0);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(it>=max_its,std::runtime_error,"Error, max iterations reached.");

  DEBUG_MSG("Global_Algorithm::execute(): linear solve complete");

  compute_strains();

  if(is_mixed_formulation()){
    const int_t spa_dim = mesh_->spatial_dimension();
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
      linear_disp->global_value(ux_id) = disp->global_value(ux_master_id);
      linear_disp->global_value(uy_id) = disp->global_value(uy_master_id);
    }
  }
  mesh_->print_field_stats();
  if(is_mixed_formulation())
    l_mesh_->print_field_stats();
  return CORRELATION_SUCCESSFUL;
}

void
Global_Algorithm::compute_strains(){
  DEBUG_MSG("Global_Algiorithm::compute_strains(): computing the Green-Lagrange strains");
  Teuchos::RCP<MultiField> du_dx = mesh_->get_field(mesh::field_enums::DU_DX_FS);
  du_dx->put_scalar(0.0);
  Teuchos::RCP<MultiField> du_dy = mesh_->get_field(mesh::field_enums::DU_DY_FS);
  du_dy->put_scalar(0.0);
  Teuchos::RCP<MultiField> dv_dx = mesh_->get_field(mesh::field_enums::DV_DX_FS);
  dv_dx->put_scalar(0.0);
  Teuchos::RCP<MultiField> dv_dy = mesh_->get_field(mesh::field_enums::DV_DY_FS);
  dv_dy->put_scalar(0.0);
  Teuchos::RCP<MultiField> strain_contribs = mesh_->get_field(mesh::field_enums::STRAIN_CONTRIBS_FS);
  Teuchos::RCP<MultiField> gl_xx = mesh_->get_field(mesh::field_enums::GREEN_LAGRANGE_STRAIN_XX_FS);
  Teuchos::RCP<MultiField> gl_yy = mesh_->get_field(mesh::field_enums::GREEN_LAGRANGE_STRAIN_YY_FS);
  Teuchos::RCP<MultiField> gl_xy = mesh_->get_field(mesh::field_enums::GREEN_LAGRANGE_STRAIN_XY_FS);
  Teuchos::RCP<MultiField> coords = mesh_->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);

  Teuchos::RCP<MultiField> overlap_coords_ptr = mesh_->get_overlap_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  MultiField & overlap_coords = *overlap_coords_ptr;
  Teuchos::ArrayRCP<const scalar_t> coords_values = overlap_coords.get_1d_view();
  Teuchos::RCP<MultiField> overlap_disp_ptr = mesh_->get_overlap_field(mesh::field_enums::DISPLACEMENT_FS);
  MultiField & overlap_disp = *overlap_disp_ptr;
  Teuchos::ArrayRCP<const scalar_t> disp_values = overlap_disp.get_1d_view();
  Teuchos::RCP<MultiField> overlap_strain_contribs_ptr = mesh_->get_overlap_field(mesh::field_enums::STRAIN_CONTRIBS_FS);
  overlap_strain_contribs_ptr->put_scalar(0.0);
  //MultiField & overlap_strain_contribs = *overlap_strain_contribs_ptr;
  //Teuchos::ArrayRCP<const scalar_t> strain_contribs_values = overlap_strain_contribs.get_1d_view();
  Teuchos::RCP<MultiField> overlap_dudx_ptr = mesh_->get_overlap_field(mesh::field_enums::DU_DX_FS);
  overlap_dudx_ptr->put_scalar(0.0);
  //MultiField & overlap_dudx = *overlap_dudx_ptr;
  //Teuchos::ArrayRCP<const scalar_t> dudx_values = overlap_dudx.get_1d_view();
  Teuchos::RCP<MultiField> overlap_dudy_ptr = mesh_->get_overlap_field(mesh::field_enums::DU_DY_FS);
  overlap_dudy_ptr->put_scalar(0.0);
  //MultiField & overlap_dudy = *overlap_dudy_ptr;
  //Teuchos::ArrayRCP<const scalar_t> dudy_values = overlap_dudy.get_1d_view();
  Teuchos::RCP<MultiField> overlap_dvdx_ptr = mesh_->get_overlap_field(mesh::field_enums::DV_DX_FS);
  overlap_dvdx_ptr->put_scalar(0.0);
  //MultiField & overlap_dvdx = *overlap_dvdx_ptr;
  //Teuchos::ArrayRCP<const scalar_t> dvdx_values = overlap_dvdx.get_1d_view();
  Teuchos::RCP<MultiField> overlap_dvdy_ptr = mesh_->get_overlap_field(mesh::field_enums::DV_DY_FS);
  overlap_dvdy_ptr->put_scalar(0.0);
  //MultiField & overlap_dvdy = *overlap_dvdy_ptr;
  //Teuchos::ArrayRCP<const scalar_t> dvdy_values = overlap_dvdy.get_1d_view();

  const int_t spa_dim = mesh_->spatial_dimension();
  DICe::mesh::Shape_Function_Evaluator_Factory shape_func_eval_factory;
  Teuchos::RCP<DICe::mesh::Shape_Function_Evaluator> shape_func_evaluator =
      //shape_func_eval_factory.create(DICe::mesh::TRI3); // linear strains
      shape_func_eval_factory.create(DICe::mesh::TRI6); // quadratic strains
  const int_t num_funcs = shape_func_evaluator->num_functions();
  scalar_t DN[num_funcs*spa_dim];
  scalar_t nodal_coords[num_funcs*spa_dim];
  scalar_t nodal_disp[num_funcs*spa_dim];
  scalar_t jac[spa_dim*spa_dim];
  scalar_t inv_jac[spa_dim*spa_dim];
  scalar_t J =0.0;
  scalar_t natural_coords[spa_dim];
  scalar_t node_nat_x[] = {0.0, 1.0, 0.0, 0.5, 0.5, 0.0};
  scalar_t node_nat_y[] = {0.0, 0.0, 1.0, 0.0, 0.5, 0.5};

  // element loop
  DICe::mesh::element_set::iterator elem_it = mesh_->get_element_set()->begin();
  DICe::mesh::element_set::iterator elem_end = mesh_->get_element_set()->end();
  for(;elem_it!=elem_end;++elem_it)
  {
    //std::cout << "ELEM: " << elem_it->get()->global_id() << std::endl;
    const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
    // compute the shape functions and derivatives for this element:
    for(int_t nd=0;nd<num_funcs;++nd){
      for(int_t dim=0;dim<spa_dim;++dim){
        const int_t stride = nd*spa_dim + dim;
        nodal_coords[stride] = coords_values[connectivity[nd]->overlap_local_id()*spa_dim + dim];
        nodal_disp[stride] = disp_values[connectivity[nd]->overlap_local_id()*spa_dim + dim];
        //std::cout << " node coords " << nodal_coords[stride] << std::endl;
      }
    }
    // iterate the nodes for this element and compute the strain at each node
    // sum the contributions
    for(int_t nd=0;nd<num_funcs;++nd){
      // isoparametric coords
      natural_coords[0] = node_nat_x[nd];
      natural_coords[1] = node_nat_y[nd];

      // evaluate the shape functions and derivatives:
      shape_func_evaluator->evaluate_shape_function_derivatives(natural_coords,DN);

      // compute the jacobian for this element:
      DICe::global::calc_jacobian(nodal_coords,DN,jac,inv_jac,J,num_funcs,spa_dim);

      for(int_t i=0;i<num_funcs;++i){
        int_t local_index = connectivity[nd]->overlap_local_id();
        overlap_dudx_ptr->local_value(local_index) += nodal_disp[i*spa_dim + 0]*(inv_jac[0]*DN[i*spa_dim + 0]+inv_jac[2]*DN[i*spa_dim + 1]);
        overlap_dudy_ptr->local_value(local_index) += nodal_disp[i*spa_dim + 0]*(inv_jac[1]*DN[i*spa_dim + 0]+inv_jac[3]*DN[i*spa_dim + 1]);
        overlap_dvdx_ptr->local_value(local_index) += nodal_disp[i*spa_dim + 1]*(inv_jac[0]*DN[i*spa_dim + 0]+inv_jac[2]*DN[i*spa_dim + 1]);
        overlap_dvdy_ptr->local_value(local_index) += nodal_disp[i*spa_dim + 1]*(inv_jac[1]*DN[i*spa_dim + 0]+inv_jac[3]*DN[i*spa_dim + 1]);
      }
      overlap_strain_contribs_ptr->local_value(connectivity[nd]->overlap_local_id()) += 1.0;
    }
  }  // elem

  // export the fields
  mesh_->field_overlap_export(overlap_dudx_ptr,DICe::mesh::field_enums::DU_DX_FS, ADD);
  mesh_->field_overlap_export(overlap_dudy_ptr,DICe::mesh::field_enums::DU_DY_FS, ADD);
  mesh_->field_overlap_export(overlap_dvdx_ptr,DICe::mesh::field_enums::DV_DX_FS, ADD);
  mesh_->field_overlap_export(overlap_dvdy_ptr,DICe::mesh::field_enums::DV_DY_FS, ADD);
  mesh_->field_overlap_export(overlap_strain_contribs_ptr,DICe::mesh::field_enums::STRAIN_CONTRIBS_FS, ADD);

  Teuchos::RCP<MultiField> du_dx_exact;
  Teuchos::RCP<MultiField> du_dy_exact;
  Teuchos::RCP<MultiField> dv_dx_exact;
  Teuchos::RCP<MultiField> dv_dy_exact;
  if(mms_problem_!=Teuchos::null){
    du_dx_exact = mesh_->get_field(mesh::field_enums::DU_DX_EXACT_FS);
    du_dy_exact = mesh_->get_field(mesh::field_enums::DU_DY_EXACT_FS);
    dv_dx_exact = mesh_->get_field(mesh::field_enums::DV_DX_EXACT_FS);
    dv_dy_exact = mesh_->get_field(mesh::field_enums::DV_DY_EXACT_FS);
  }

  for(int_t i=0;i<mesh_->get_scalar_node_dist_map()->get_num_local_elements();++i){
    const scalar_t count = strain_contribs->local_value(i);
    du_dx->local_value(i) /= count;
    du_dy->local_value(i) /= count;
    dv_dx->local_value(i) /= count;
    dv_dy->local_value(i) /= count;
    // compute the green-lagrange strains
    gl_xx->local_value(i) = 0.5*(2.0*du_dx->local_value(i) + du_dx->local_value(i)*du_dx->local_value(i) + dv_dx->local_value(i)*dv_dx->local_value(i));
    gl_yy->local_value(i) = 0.5*(2.0*dv_dy->local_value(i) + du_dy->local_value(i)*du_dy->local_value(i) + dv_dy->local_value(i)*dv_dy->local_value(i));
    gl_xy->local_value(i) = 0.5*(du_dy->local_value(i) + dv_dx->local_value(i) + du_dx->local_value(i)*du_dy->local_value(i) + dv_dx->local_value(i)*dv_dy->local_value(i));
    if(mms_problem_!=Teuchos::null){
      int_t ix = i*spa_dim + 0;
      int_t iy = i*spa_dim + 1;
      scalar_t x = coords->local_value(ix);
      scalar_t y = coords->local_value(iy);
      scalar_t b_x_x=0.0,b_x_y=0.0,b_y_x=0.0,b_y_y=0.0;
      mms_problem_->grad_velocity(x,y,b_x_x,b_x_y,b_y_x,b_y_y);
      du_dx_exact->local_value(i) = b_x_x;
      du_dy_exact->local_value(i) = b_x_y;
      dv_dx_exact->local_value(i) = b_y_x;
      dv_dy_exact->local_value(i) = b_y_y;
    }
  }
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
