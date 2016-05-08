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

namespace DICe {

namespace global{

Global_Algorithm::Global_Algorithm(Schema * schema,
  const Teuchos::RCP<Teuchos::ParameterList> & params):
  schema_(schema),
  mesh_size_(1000.0),
  alpha2_(1.0),
  mms_problem_(Teuchos::null),
  is_initialized_(false)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!schema,std::runtime_error,"Error, cannot have null schema in this constructor");
  TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,"Error, params cannot be null");
  /// get the mesh size from the params
  if(params->isParameter(DICe::mesh_size))
    mesh_size_ = params->get<double>(DICe::mesh_size);
  DEBUG_MSG("Global_Algorithm::Global_Algorithm(): Mesh size " << mesh_size_);

  // determine the image extents:
  const scalar_t buff_size = 20.0; //pixels
  Teuchos::ArrayRCP<scalar_t> points_x(4);
  Teuchos::ArrayRCP<scalar_t> points_y(4);
  points_x[0] = buff_size; points_y[0] = buff_size;
  points_x[1] = schema->ref_img()->width() - buff_size; points_y[1] = buff_size;
  points_x[2] = schema->ref_img()->width() - buff_size; points_y[2] = schema->ref_img()->height() - buff_size;
  points_x[3] = buff_size; points_y[3] = schema->ref_img()->height() - buff_size;

  default_constructor_tasks(points_x,points_y,params);
}

Global_Algorithm::Global_Algorithm(const Teuchos::RCP<Teuchos::ParameterList> & params):
  schema_(NULL),
  mesh_size_(1000.0),
  alpha2_(1.0),
  is_initialized_(false)
{
  TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,"Error, params cannot be null");
  /// get the mesh size from the params
  if(params->isParameter(DICe::mesh_size))
    mesh_size_ = params->get<double>(DICe::mesh_size);

  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::mms_spec),std::runtime_error,"Error, mms_spec must be defined"
      " in the parameters file for this constructor");
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

  default_constructor_tasks(points_x,points_y,params);
}

void
Global_Algorithm::default_constructor_tasks(Teuchos::ArrayRCP<scalar_t> & points_x,
  Teuchos::ArrayRCP<scalar_t> & points_y,
  const Teuchos::RCP<Teuchos::ParameterList> & params){
  TEUCHOS_TEST_FOR_EXCEPTION(params==Teuchos::null,std::runtime_error,
    "Error, params must be defined");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::global_regularization_alpha),std::runtime_error,
    "Error, global_regularization_alpha must be defined");
  const scalar_t alpha = params->get<double>(DICe::global_regularization_alpha);
  alpha2_ = alpha*alpha;

  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::output_prefix),std::runtime_error,
    "Error, output_prefix must be defined");
  TEUCHOS_TEST_FOR_EXCEPTION(!params->isParameter(DICe::output_folder),std::runtime_error,
    "Error, output_folder must be defined");
  const std::string output_folder = params->get<std::string>(DICe::output_folder);
  const std::string output_prefix = params->get<std::string>(DICe::output_prefix);
  std::stringstream output_file_name_ss;
  output_file_name_ss << output_prefix << ".e";
  output_file_name_ = output_file_name_ss.str();
  DEBUG_MSG("Global_Algorithm::default_constructor_tasks(): output file name: " << output_file_name_);

  mesh_ = DICe::generate_tri6_mesh(points_x,points_y,mesh_size_,output_file_name_);

  DICe::mesh::create_output_exodus_file(mesh_,output_folder);
  // create the necessary fields:
  mesh_->create_field(mesh::field_enums::DISPLACEMENT_FS);
  mesh_->create_field(mesh::field_enums::RESIDUAL_FS);
  mesh_->create_field(mesh::field_enums::LHS_FS);
  mesh_->create_field(mesh::field_enums::EXACT_SOL_VECTOR_FS);
  mesh_->create_field(mesh::field_enums::IMAGE_PHI_FS);
  mesh_->create_field(mesh::field_enums::IMAGE_GRAD_PHI_FS);
  DICe::mesh::create_exodus_output_variable_names(mesh_);

//  // for now default terms: TODO TODO TODO manage this another way
//  add_term(DIV_SYMMETRIC_STRAIN_REGULARIZATION);
//  add_term(MMS_GRAD_IMAGE_TENSOR);
//  add_term(MMS_IMAGE_TIME_FORCE);
//  add_term(MMS_FORCE);

  //add_term(IMAGE_TIME_FORCE);
}

void
Global_Algorithm::pre_execution_tasks(){

  // initialize the solver:
  // set up the solver
  Teuchos::ParameterList belos_list;
  const int_t maxiters = 500; // these are the max iterations of the belos solver not the nonlinear iterations
  const int_t numblk = (maxiters > 500) ? 500 : maxiters;
  const int_t maxrestarts = 1;
  const double conv_tol = 1.0E-8;
  std::string ortho("DGKS");
  belos_list.set( "Num Blocks", numblk);
  belos_list.set( "Maximum Iterations", maxiters);
  belos_list.set( "Convergence Tolerance", conv_tol); // Relative convergence tolerance requested
  belos_list.set( "Maximum Restarts", maxrestarts );  // Maximum number of restarts allowed
  int verbosity = Belos::Errors + Belos::Warnings + Belos::Debug + Belos::TimingDetails + Belos::FinalSummary + Belos::StatusTestDetails;
  belos_list.set( "Verbosity", verbosity );
  belos_list.set( "Output Style", Belos::Brief );
  belos_list.set( "Output Frequency", 1 );
  belos_list.set( "Orthogonalization", ortho); // Orthogonalization type

  /// linear problem for solve
  linear_problem_ = Teuchos::rcp(new Belos::LinearProblem<mv_scalar_type,vec_type,operator_type>());
  /// Belos solver
  belos_solver_ = Teuchos::rcp( new Belos::BlockGmresSolMgr<mv_scalar_type,vec_type,operator_type>
                                (linear_problem_,Teuchos::rcp(&belos_list,false)));

  DEBUG_MSG("Solver and linear problem have been initialized.");

  const int_t spa_dim = mesh_->spatial_dimension();
  matrix_service_ = Teuchos::rcp(new DICe::Matrix_Service(spa_dim));
  matrix_service_->initialize_bc_register(mesh_->get_vector_node_dist_map()->get_num_local_elements(),
    mesh_->get_vector_node_overlap_map()->get_num_local_elements());

  // set up the boundary condition nodes: FIXME this assumes there is only one node set
  //mesh->get_vector_node_dist_map()->describe();
  DICe::mesh::bc_set * bc_set = mesh_->get_node_bc_sets();
  const int_t boundary_node_set_id = 0;
  for(size_t i=0;i<bc_set->find(boundary_node_set_id)->second.size();++i){
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

  DEBUG_MSG("Matrix service has been initialized.");
  is_initialized_ = true;

  std::set<Global_EQ_Term>::iterator it=eq_terms_.begin();
  std::set<Global_EQ_Term>::iterator it_end=eq_terms_.end();
  for(;it!=it_end;++it)
    DEBUG_MSG("Global_Algorithm::pre_execution_tasks() adding EQ term " << to_string(*it));

  // if this is not an mms problem, set up the images
  if(schema_){
    initialize_ref_image();
    set_def_image();
  }
}

Status_Flag
Global_Algorithm::execute(){
  DEBUG_MSG("Global_Algorithm::execute() called");
  if(!is_initialized_) pre_execution_tasks();

  const int_t spa_dim = mesh_->spatial_dimension();
  const int_t p_rank = mesh_->get_comm()->get_rank();
  const int_t relations_size = mesh_->max_num_node_relations();
  Teuchos::RCP<DICe::MultiField_Matrix> tangent =
      Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_vector_node_dist_map(),relations_size));

  DEBUG_MSG("Tangent has been allocated.");

  MultiField & residual = *mesh_->get_field(mesh::field_enums::RESIDUAL_FS);
  residual.put_scalar(0.0);
  MultiField & disp = *mesh_->get_field(mesh::field_enums::DISPLACEMENT_FS);
  disp.put_scalar(0.0);
  MultiField & lhs = *mesh_->get_field(mesh::field_enums::LHS_FS);
  MultiField & coords = *mesh_->get_field(mesh::field_enums::INITIAL_COORDINATES_FS);
  MultiField & exact_sol = *mesh_->get_field(mesh::field_enums::EXACT_SOL_VECTOR_FS);
  MultiField & image_phi = *mesh_->get_field(mesh::field_enums::IMAGE_PHI_FS);
  MultiField & image_grad_phi = *mesh_->get_field(mesh::field_enums::IMAGE_GRAD_PHI_FS);

  scalar_t residual_norm = 0.0;

  // global iteration loop:
  const int_t max_its = 1;
  if(p_rank==0){
    DEBUG_MSG("    iteration  residual_norm");
  }
  for(int_t iteration=0;iteration<max_its;++iteration){
    if(p_rank==0){
      DEBUG_MSG("    " << iteration << "      " << residual_norm);
    }

    // assemble the distributed tangent matrix
    Teuchos::RCP<DICe::MultiField_Matrix> tangent_overlap =
        Teuchos::rcp(new DICe::MultiField_Matrix(*mesh_->get_vector_node_overlap_map(),relations_size));
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
      const int_t row_gid = mesh_->get_vector_node_overlap_map()->get_global_element(i);
      col_id_array_map.insert(std::pair<int_t,Teuchos::Array<int_t> >(row_gid,id_array));
      Teuchos::Array<mv_scalar_type> value_array;
      values_array_map.insert(std::pair<int_t,Teuchos::Array<mv_scalar_type> >(row_gid,value_array));
    }

    // establish the shape functions (using P2-P1 element for velocity pressure, or P2 velocity if no constraint):
    DICe::mesh::Shape_Function_Evaluator_Factory shape_func_eval_factory;
//    Teuchos::RCP<DICe::mesh::Shape_Function_Evaluator> tri3_shape_func_evaluator =
//        shape_func_eval_factory.create(DICe::mesh::TRI3);
//    const int_t tri3_num_funcs = tri3_shape_func_evaluator->num_functions();
//    scalar_t N3[tri3_num_funcs];
//    scalar_t DN3[tri3_num_funcs*spa_dim];
    Teuchos::RCP<DICe::mesh::Shape_Function_Evaluator> tri6_shape_func_evaluator =
        shape_func_eval_factory.create(DICe::mesh::TRI6);
    const int_t tri6_num_funcs = tri6_shape_func_evaluator->num_functions();
    scalar_t N6[tri6_num_funcs];
    scalar_t DN6[tri6_num_funcs*spa_dim];
    int_t node_ids[tri6_num_funcs];
    scalar_t nodal_coords[tri6_num_funcs*spa_dim];
    scalar_t natural_coords[spa_dim];
    scalar_t jac[spa_dim*spa_dim];
    scalar_t inv_jac[spa_dim*spa_dim];
    scalar_t J =0.0;
    //const int_t B_dim = 2*spa_dim - 1;
    //scalar_t B[B_dim*tri6_num_funcs*spa_dim];
    scalar_t elem_stiffness[tri6_num_funcs*spa_dim*tri6_num_funcs*spa_dim];
    scalar_t elem_force[tri6_num_funcs*spa_dim];
    //scalar_t grad_phi[spa_dim];
    scalar_t x=0.0,y=0.0;

    // get the natural integration points for this element:
    const int_t integration_order = 3;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > gp_locs;
    Teuchos::ArrayRCP<scalar_t> gp_weights;
    int_t num_integration_points = -1;
    tri6_shape_func_evaluator->get_natural_integration_points(integration_order,gp_locs,gp_weights,num_integration_points);

    // get the high order natural integration points for this element:
    const int_t high_order_integration_order = 6;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > high_order_gp_locs;
    Teuchos::ArrayRCP<scalar_t> high_order_gp_weights;
    int_t high_order_num_integration_points = -1;
    tri6_shape_func_evaluator->get_natural_integration_points(high_order_integration_order,
      high_order_gp_locs,
      high_order_gp_weights,
      high_order_num_integration_points);

    // gather the OVERLAP fields
    Teuchos::RCP<MultiField> overlap_residual_ptr = mesh_->get_overlap_field(mesh::field_enums::RESIDUAL_FS);
    MultiField & overlap_residual = *overlap_residual_ptr;
    overlap_residual.put_scalar(0.0);
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
      // clear the elem force
      for(int_t i=0;i<tri6_num_funcs*spa_dim;++i)
        elem_force[i] = 0.0;

      // low-order gauss point loop:
      for(int_t gp=0;gp<num_integration_points;++gp){

        // isoparametric coords of the gauss point
        for(int_t dim=0;dim<spa_dim;++dim)
          natural_coords[dim] = gp_locs[gp][dim];
        //std::cout << " natural coords " << natural_coords[0] << " " << natural_coords[1] << std::endl;

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

        // stiffness terms

        // alpha * div(0.5*(grad(b) + grad(b)^T))
        if(has_term(DIV_SYMMETRIC_STRAIN_REGULARIZATION))
          div_symmetric_strain(spa_dim,tri6_num_funcs,alpha2_,J,gp_weights[gp],inv_jac,DN6,elem_stiffness);

        // grad(phi) tensor_prod grad(phi)
        if(has_term(MMS_GRAD_IMAGE_TENSOR))
          mms_grad_image_tensor(mms_problem_,spa_dim,tri6_num_funcs,x,y,J,gp_weights[gp],N6,elem_stiffness);

        // RHS terms

        // mms force
        if(has_term(MMS_FORCE))
          mms_force(mms_problem_,spa_dim,tri6_num_funcs,x,y,alpha2_,J,gp_weights[gp],N6,elem_force);

        // d_dt(phi) * grad(phi)
        if(has_term(MMS_IMAGE_TIME_FORCE))
          mms_image_time_force(mms_problem_,spa_dim,tri6_num_funcs,x,y,J,gp_weights[gp],N6,elem_force);

      } // gp loop

//      // high-order gauss point loop:
//      for(int_t gp=0;gp<high_order_num_integration_points;++gp){
//
//        // isoparametric coords of the gauss point
//        for(int_t dim=0;dim<spa_dim;++dim)
//          natural_coords[dim] = gp_locs[gp][dim];
//        //std::cout << " natural coords " << natural_coords[0] << " " << natural_coords[1] << std::endl;
//
//        // evaluate the shape functions and derivatives:
//        tri6_shape_func_evaluator->evaluate_shape_functions(natural_coords,N6);
//        tri6_shape_func_evaluator->evaluate_shape_function_derivatives(natural_coords,DN6);
//
//        // physical gp location
//        x = 0.0; y=0.0;
//        for(int_t i=0;i<tri6_num_funcs;++i){
//          x += nodal_coords[i*spa_dim+0]*N6[i];
//          y += nodal_coords[i*spa_dim+1]*N6[i];
//        }
//        //std::cout << " physical coords " << x << " " << y << std::endl;
//
//        // compute the jacobian for this element:
//        DICe::global::calc_jacobian(nodal_coords,DN6,jac,inv_jac,J,tri6_num_funcs,spa_dim);
//
//        // add the element terms that go outside the standard FEM gp loop:
//        if(has_term(IMAGE_TIME_FORCE))
//          // TODO TODO
//
//
//      }





      // assemble the global stiffness matrix
      for(int_t i=0;i<tri6_num_funcs;++i){
        for(int_t m=0;m<spa_dim;++m){
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
        residual.global_value(nodex_id) += elem_force[i*spa_dim+0];
        residual.global_value(nodey_id) += elem_force[i*spa_dim+1];
      } // num_funcs

    }  // elem
    // export the overlap residual to the dist vector
    //mesh_->field_overlap_export(overlap_residual_ptr, mesh_::field_enums::RESIDUAL_FS, ADD);

    // add ones to the diagonal for kinematic bc nodes:
    for(int_t i=0;i<mesh_->get_vector_node_overlap_map()->get_num_local_elements();++i)
    {
      if(matrix_service_->is_col_bc(i))
      {
        const int_t global_id = mesh_->get_vector_node_overlap_map()->get_global_element(i);
        col_id_array_map.find(global_id)->second.push_back(global_id);
        values_array_map.find(global_id)->second.push_back(1.0);
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

    MultiField_Exporter exporter (*mesh_->get_vector_node_overlap_map(),*mesh_->get_vector_node_dist_map());
    tangent->do_export(tangent_overlap, exporter, ADD);
    tangent->fill_complete();

    //tangent->describe();

    if(mms_problem_!=Teuchos::null){
      // enforce the dirichlet boundary conditions
      //const scalar_t boundary_value = 100.0;
      // add ones to the diagonal for kinematic bc nodes:
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
        image_phi.local_value(i) = phi;
        mms_problem_->phi_derivatives(x,y,d_phi_dt,grad_phi_x,grad_phi_y);
        image_grad_phi.local_value(ix) = grad_phi_x;
        image_grad_phi.local_value(iy) = grad_phi_y;

        //calc_mms_bc_2(x,y,schema->img_width(),b_x,b_y);
        //calc_mms_bc_simple(x,y,b_x,b_y);
        exact_sol.local_value(ix) = b_x;
        exact_sol.local_value(iy) = b_y;
        if(matrix_service_->is_col_bc(ix)){
          // get the coordinates of the node
          residual.local_value(ix) = b_x;
        }
        if(matrix_service_->is_col_bc(iy)){
          // get the coordinates of the node
          residual.local_value(iy) = b_y;
        }
      }
    }

    //residual.describe();

    DEBUG_MSG("Solving the linear system...");

    // solve:
    lhs.put_scalar(0.0);
    linear_problem_->setOperator(tangent->get());
    bool is_set = linear_problem_->setProblem(lhs.get(), residual.get());
    TEUCHOS_TEST_FOR_EXCEPTION(!is_set, std::logic_error,
      "Error: Belos::LinearProblem::setProblem() failed to set up correctly.\n");
    Belos::ReturnType ret = belos_solver_->solve();
    if(ret != Belos::Converged && p_rank==0)
      std::cout << "*** WARNING: Belos linear solver did not converge!" << std::endl;
  } // end iteration loop

  // upate the displacements
  disp.update(1.0,lhs,1.0);

  return CORRELATION_SUCCESSFUL;
}


void
Global_Algorithm::initialize_ref_image(){
  TEUCHOS_TEST_FOR_EXCEPTION(!schema_,std::runtime_error,"Error, schema must not be null for this method");

  schema_->ref_img()->write("pre_ref_img.tif");

  Teuchos::RCP<Teuchos::ParameterList> imgParams = Teuchos::rcp(new Teuchos::ParameterList());
  imgParams->set(DICe::compute_image_gradients,true);
  imgParams->set(DICe::gradient_method,FINITE_DIFFERENCE);
  ref_img_ = schema_->ref_img()->normalize(imgParams);
  ref_img_->write("global_normalized_image.tif");
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
  grad_x_img_->write("grad_x_img.tif");
  grad_y_img_ = Teuchos::rcp(new Image(w,h,grad_ref_y));
  grad_y_img_->write("grad_y_img.tif");
}

void
Global_Algorithm::set_def_image(){
  schema_->def_img()->write("pre_def_img.tif");

  TEUCHOS_TEST_FOR_EXCEPTION(!schema_,std::runtime_error,"Error, schema must not be null for this method");
  def_img_ = schema_->def_img()->normalize();
  def_img_->write("global_normalized_def_image.tif");
}

void
Global_Algorithm::post_execution_tasks(const scalar_t & time_stamp){
  TEUCHOS_TEST_FOR_EXCEPTION(!is_initialized_,std::runtime_error,"Error, must be initialized");

  DEBUG_MSG("Writing the output file with time stamp: " << time_stamp);
  DICe::mesh::exodus_output_dump(mesh_,1,time_stamp);
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

  DEBUG_MSG("MMS Error bx: " << error_bx << " by: " << error_by << " lambda: " << error_lambda);
}


}// end global namespace

}// End DICe Namespace
