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

#include <DICe_GlobalUtils.h>
#include <DICe_MeshIO.h>
#include <DICe_MatrixService.h>

#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosLinearProblem.hpp>
#ifdef DICE_TPETRA
  #include <BelosTpetraAdapter.hpp>
#else
  #include <BelosEpetraAdapter.hpp>
#endif

namespace DICe {

namespace global{

void initialize_exodus_output(Schema * schema,
  const std::string & output_folder){
  Teuchos::RCP<DICe::mesh::Mesh> mesh = schema->mesh();
  DICe::mesh::create_output_exodus_file(mesh,output_folder);
  // create the necessary fields:
  mesh->create_field(mesh::field_enums::DISPLACEMENT_FS);
  mesh->create_field(mesh::field_enums::RESIDUAL_FS);
  mesh->create_field(mesh::field_enums::LHS_FS);
  DICe::mesh::create_exodus_output_variable_names(mesh);
}

// take a schema pointer and create some fields on the mesh
Status_Flag execute_global_step(Schema * schema){
  Teuchos::RCP<DICe::mesh::Mesh> mesh = schema->mesh();

  int_t p_rank = mesh->get_comm()->get_rank();
  const int_t spa_dim = mesh->spatial_dimension();
  //const scalar_t min_value_threshold = 1.0E-10; // smallest value of the stiffness matrix that will actually be inserted

  // square the alpha term
  scalar_t alpha2 = schema->global_constraint_coefficient();
  alpha2*=alpha2;

  // clear the dislacement field: TODO may not want to do this for an image progression
  MultiField & residual = *mesh->get_field(mesh::field_enums::RESIDUAL_FS);
  MultiField & lhs = *mesh->get_field(mesh::field_enums::LHS_FS);
  MultiField & disp = *mesh->get_field(mesh::field_enums::DISPLACEMENT_FS);
  disp.put_scalar(0.0);
  residual.put_scalar(0.0);

  scalar_t residual_norm = 0.0;

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
  Teuchos::RCP< Belos::LinearProblem<mv_scalar_type,vec_type,operator_type> > linear_problem =
      Teuchos::rcp(new Belos::LinearProblem<mv_scalar_type,vec_type,operator_type>());
  /// Belos solver
  Teuchos::RCP< Belos::SolverManager<mv_scalar_type,vec_type,operator_type> > belos_solver  =
      Teuchos::rcp( new Belos::BlockGmresSolMgr<mv_scalar_type,vec_type,operator_type>(linear_problem,Teuchos::rcp(&belos_list,false)));

  DEBUG_MSG("Solver and linear problem have been initialized.");

  Teuchos::RCP<DICe::Matrix_Service> matrix_service;
  matrix_service = Teuchos::rcp(new DICe::Matrix_Service(mesh->spatial_dimension()));
  matrix_service->initialize_bc_register(mesh->get_vector_node_dist_map()->get_num_local_elements(),
    mesh->get_vector_node_overlap_map()->get_num_local_elements());

  // set up the boundary condition nodes: FIXME this assumes there is only one node set
  //mesh->get_vector_node_dist_map()->describe();
  DICe::mesh::bc_set * bc_set = mesh->get_node_bc_sets();
  const int_t boundary_node_set_id = 0;
  for(size_t i=0;i<bc_set->find(boundary_node_set_id)->second.size();++i){
    const int_t node_gid = bc_set->find(boundary_node_set_id)->second[i];
    //std::cout << " disp x condition on node " << node_gid << std::endl;
    bool is_local_node = mesh->get_vector_node_dist_map()->is_node_global_elem((node_gid-1)*spa_dim+1);
    if(is_local_node){
      const int_t row_id = mesh->get_vector_node_dist_map()->get_local_element((node_gid-1)*spa_dim+1);
      matrix_service->register_row_bc(row_id);
    }
    int_t col_id = mesh->get_vector_node_overlap_map()->get_local_element((node_gid-1)*spa_dim+1);
    matrix_service->register_col_bc(col_id);
    //std::cout << " disp y condition on node " << node_gid << std::endl;
    is_local_node = mesh->get_vector_node_dist_map()->is_node_global_elem((node_gid-1)*spa_dim+2);
    if(is_local_node){
      const int_t row_id = mesh->get_vector_node_dist_map()->get_local_element((node_gid-1)*spa_dim+2);
      matrix_service->register_row_bc(row_id);
    }
    col_id = mesh->get_vector_node_overlap_map()->get_local_element((node_gid-1)*spa_dim+2);
    matrix_service->register_col_bc(col_id);
  }

  DEBUG_MSG("Matrix service has been initialized.");

  const int_t relations_size = mesh->max_num_node_relations();
  Teuchos::RCP<DICe::MultiField_Matrix> tangent =
      Teuchos::rcp(new DICe::MultiField_Matrix(*mesh->get_vector_node_dist_map(),relations_size));

  DEBUG_MSG("Tangent has been allocated.");

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
        Teuchos::rcp(new DICe::MultiField_Matrix(*mesh->get_vector_node_overlap_map(),relations_size));
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
      const int_t row_gid = mesh->get_vector_node_overlap_map()->get_global_element(i);
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
    const int_t B_dim = 2*spa_dim - 1;
    scalar_t B[B_dim*tri6_num_funcs*spa_dim];
    scalar_t elem_stiffness[tri6_num_funcs*spa_dim*tri6_num_funcs*spa_dim];

    // get the natural integration points for this element:
    const int_t integration_order = 3;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<scalar_t> > gp_locs;
    Teuchos::ArrayRCP<scalar_t> gp_weights;
    int_t num_integration_points = -1;
    tri6_shape_func_evaluator->get_natural_integration_points(integration_order,gp_locs,gp_weights,num_integration_points);

    // gather the OVERLAP fields
    Teuchos::RCP<MultiField> overlap_residual_ptr = mesh->get_overlap_field(mesh::field_enums::RESIDUAL_FS);
    MultiField & overlap_residual = *overlap_residual_ptr;
    overlap_residual.put_scalar(0.0);
    Teuchos::RCP<MultiField> overlap_coords_ptr = mesh->get_overlap_field(mesh::field_enums::INITIAL_COORDINATES_FS);
    MultiField & overlap_coords = *overlap_coords_ptr;
    Teuchos::ArrayRCP<const scalar_t> coords_values = overlap_coords.get_1d_view();

    // element loop
    DICe::mesh::element_set::iterator elem_it = mesh->get_element_set()->begin();
    DICe::mesh::element_set::iterator elem_end = mesh->get_element_set()->end();
    for(;elem_it!=elem_end;++elem_it)
    {
      //std::cout << "ELEM: " << elem_it->get()->global_id() << std::endl;
      const DICe::mesh::connectivity_vector & connectivity = *elem_it->get()->connectivity();
      // compute the shape functions and derivatives for this element:
      for(int_t nd=0;nd<tri6_num_funcs;++nd){
        node_ids[nd] = connectivity[nd]->global_id();
        for(int_t dim=0;dim<spa_dim;++dim){
          const int_t stride = nd*spa_dim + dim;
          nodal_coords[stride] = coords_values[connectivity[nd]->overlap_local_id()*spa_dim + dim];
        }
      }

      // clear the elem stiffness
      for(int_t i=0;i<tri6_num_funcs*spa_dim*tri6_num_funcs*spa_dim;++i)
        elem_stiffness[i] = 0.0;

      // gauss point loop:
      for(int_t gp=0;gp<num_integration_points;++gp){

        // clear the temp matrices:
        for(int_t i=0;i<B_dim*tri6_num_funcs*spa_dim;++i){
          B[i] = 0.0;
        }

        // isoparametric coords of the gauss point
        for(int_t dim=0;dim<spa_dim;++dim)
          natural_coords[dim] = gp_locs[gp][dim];

        // evaluate the shape functions and derivatives:
        tri6_shape_func_evaluator->evaluate_shape_functions(natural_coords,N6);
        tri6_shape_func_evaluator->evaluate_shape_function_derivatives(natural_coords,DN6);

        //DEBUG_MSG("Calculating the jacobian");

        // compute the jacobian for this element:
        DICe::global::calc_jacobian(nodal_coords,DN6,jac,inv_jac,J,tri6_num_funcs,spa_dim);

        //DEBUG_MSG("Assembling the B matrix");

        // compute the B matrix
        DICe::global::calc_B(DN6,inv_jac,tri6_num_funcs,spa_dim,B);

        //DEBUG_MSG("Assembling B'*B");

        // compute B'*B
        for(int_t i=0;i<tri6_num_funcs*spa_dim;++i){
          for(int_t j=0;j<B_dim;++j){
            for(int_t k=0;k<tri6_num_funcs*spa_dim;++k){
              elem_stiffness[i*tri6_num_funcs*spa_dim + k] +=
                  alpha2*B[j*tri6_num_funcs*spa_dim+i]*B[j*tri6_num_funcs*spa_dim + k] * gp_weights[gp] * J;
            }
          }
        }
      } // gp loop

      // assemble the global stiffness matrix

      //DEBUG_MSG("Preparing global tangent entries for insertion");
      for(int_t i=0;i<tri6_num_funcs;++i){
        for(int_t m=0;m<spa_dim;++m){
          for(int_t j=0;j<tri6_num_funcs;++j){
            for(int_t n=0;n<spa_dim;++n){
              const int_t row = (node_ids[i]-1)*spa_dim + m + 1;
              const int_t col = (node_ids[j]-1)*spa_dim + n + 1;
              const scalar_t value = elem_stiffness[(i*spa_dim + m)*tri6_num_funcs*spa_dim + (j*spa_dim + n)];
              const bool is_local_row_node = mesh->get_vector_node_dist_map()->is_node_global_elem(row);
              const bool row_is_bc_node = is_local_row_node ?
                  matrix_service->is_row_bc(mesh->get_vector_node_dist_map()->get_local_element(row)) : false;
              //const bool col_is_bc_node = matrix_service->is_col_bc(mesh->get_vector_node_overlap_map()->get_local_element(col));
              //std::cout << " row " << row << " col " << col << " row_is_bc " << row_is_bc_node << " col_is_bc " << col_is_bc_node << std::endl;
              if(!row_is_bc_node){// && !col_is_bc_node){
                col_id_array_map.find(row)->second.push_back(col);
                values_array_map.find(row)->second.push_back(value);
              }
            } // spa dim
          } // num_funcs
        } // spa dim
      } // num_funcs
      //DEBUG_MSG("Done preparing values for insertion...");

    }  // elem

    // add ones to the diagonal for kinematic bc nodes:
    for(int_t i=0;i<mesh->get_vector_node_overlap_map()->get_num_local_elements();++i)
    {
      if(matrix_service->is_col_bc(i))
      {
        const int_t global_id = mesh->get_vector_node_overlap_map()->get_global_element(i);
        col_id_array_map.find(global_id)->second.push_back(global_id);
        values_array_map.find(global_id)->second.push_back(1.0);
      }
    }

    DEBUG_MSG("Inserting values into the global tangent...");

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

    DEBUG_MSG("Exporting the distributed tangent...");

    MultiField_Exporter exporter (*mesh->get_vector_node_overlap_map(),*mesh->get_vector_node_dist_map());
    tangent->do_export(tangent_overlap, exporter, ADD);
    tangent->fill_complete();

    //tangent->describe();

    DEBUG_MSG("Enforcing Dirichlet boundary condition");
    // enforce the dirichlet boundary conditions
    const scalar_t boundary_value = 100.0;
    // add ones to the diagonal for kinematic bc nodes:
    for(int_t i=0;i<mesh->get_vector_node_dist_map()->get_num_local_elements();++i){
      if(matrix_service->is_col_bc(i))
        residual.local_value(i) = boundary_value;
    }

    DEBUG_MSG("Solving the linear system...");

    // solve:
    lhs.put_scalar(0.0);
    linear_problem->setOperator(tangent->get());
    bool is_set = linear_problem->setProblem(lhs.get(), residual.get());
    TEUCHOS_TEST_FOR_EXCEPTION(!is_set, std::logic_error,
      "Error: Belos::LinearProblem::setProblem() failed to set up correctly.\n");
    Belos::ReturnType ret = belos_solver->solve();
    if(ret != Belos::Converged && p_rank==0)
      std::cout << "*** WARNING: Belos linear solver did not converge!" << std::endl;
  } // end iteration loop

  // upate the displacements
  disp.update(1.0,lhs,1.0);

  DEBUG_MSG("Writing the output file...");

  scalar_t time = schema->image_frame(); // pseudo time is the frame number
  DICe::mesh::exodus_output_dump(mesh,1,time);

  return CORRELATION_SUCCESSFUL; // TODO change this to success
}


void calc_jacobian(const scalar_t * xcap,
  const scalar_t * DN,
  scalar_t * jacobian,
  scalar_t * inv_jacobian,
  scalar_t & J,
  int_t num_elem_nodes,
  int_t dim ){

  for(int_t i=0;i<dim*dim;++i){
    jacobian[i] = 0.0;
    inv_jacobian[i] = 0.0;
  }
  J = 0.0;

  for(int_t i=0;i<dim;++i)
    for(int_t j=0;j<dim;++j)
      for(int_t k=0;k<num_elem_nodes;++k)
        jacobian[i*dim+j] += xcap[k*dim + i] * DN[k*dim+j];
//  std::cout << " jacobian " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<dim;++j)
//      std::cout << " " << jacobian[i*dim+j];
//    std::cout << std::endl;
//  }

  if(dim==2){
    J = jacobian[0]*jacobian[3] - jacobian[1]*jacobian[2];
    TEUCHOS_TEST_FOR_EXCEPTION(J<=0.0,std::runtime_error,
      "Error: determinant 0.0 encountered or negative det");
    inv_jacobian[0] =  jacobian[3] / J;
    inv_jacobian[1] = -jacobian[1] / J;
    inv_jacobian[2] = -jacobian[2] / J;
    inv_jacobian[3] =  jacobian[0] / J;
  }
  else if(dim==3){
    J =   jacobian[0]*jacobian[4]*jacobian[8] + jacobian[1]*jacobian[5]*jacobian[6] + jacobian[2]*jacobian[3]*jacobian[7]
        - jacobian[6]*jacobian[4]*jacobian[2] - jacobian[7]*jacobian[5]*jacobian[0] - jacobian[8]*jacobian[3]*jacobian[1];
    TEUCHOS_TEST_FOR_EXCEPTION(J<=0.0,std::runtime_error,
      "Error: determinant 0.0 encountered or negative det");
    inv_jacobian[0] = ( -jacobian[5]*jacobian[7] + jacobian[4]*jacobian[8]) /  J;
    inv_jacobian[1] = (  jacobian[2]*jacobian[7] - jacobian[1]*jacobian[8]) /  J;
    inv_jacobian[2] = ( -jacobian[2]*jacobian[4] + jacobian[1]*jacobian[5]) /  J;
    inv_jacobian[3] = (  jacobian[5]*jacobian[6] - jacobian[3]*jacobian[8]) /  J;
    inv_jacobian[4] = ( -jacobian[2]*jacobian[6] + jacobian[0]*jacobian[8]) /  J;
    inv_jacobian[5] = (  jacobian[2]*jacobian[3] - jacobian[0]*jacobian[5]) /  J;
    inv_jacobian[6] = ( -jacobian[4]*jacobian[6] + jacobian[3]*jacobian[7]) /  J;
    inv_jacobian[7] = (  jacobian[1]*jacobian[6] - jacobian[0]*jacobian[7]) /  J;
    inv_jacobian[8] = ( -jacobian[1]*jacobian[3] + jacobian[0]*jacobian[4]) /  J;
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(false,std::runtime_error,"Error, invalid dimension");

//  std::cout << " J " << J << std::endl;
//  std::cout << " inv_jacobian " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<dim;++j)
//      std::cout << " " << inv_jacobian[i*dim+j];
//    std::cout << std::endl;
//  }
};

void calc_B(const scalar_t * DN,
  const scalar_t * inv_jacobian,
  const int_t num_elem_nodes,
  const int_t dim,
  scalar_t * solid_B){

  scalar_t dN[dim*num_elem_nodes];
  for(int_t i=0;i<dim*num_elem_nodes;++i)
    dN[i] = 0.0;

//  std::cout << " DN " << std::endl;
//  for(int_t j=0;j<num_elem_nodes;++j){
//    for(int_t i=0;i<dim;++i)
//      std::cout << " " << DN[j*dim+i];
//    std::cout << std::endl;
//  }

  // compute j_inv_transpose * DN_transpose
  for(int_t i=0;i<dim;++i)
    for(int_t j=0;j<num_elem_nodes;++j)
      for(int_t k=0;k<dim;++k)
        dN[i*num_elem_nodes+j] += inv_jacobian[k*dim + i] * DN[j*dim+k];

//  std::cout << " dN " << std::endl;
//  for(int_t i=0;i<dim;++i){
//    for(int_t j=0;j<num_elem_nodes;++j)
//      std::cout << " " << dN[i*num_elem_nodes+j];
//    std::cout << std::endl;
//  }

  const int_t stride = dim*num_elem_nodes;
  int_t placer = 0;
  if(dim ==3){
    for(int_t i=0;i<6*dim*num_elem_nodes;++i)
      solid_B[i] = 0.0;
    for(int_t i=0;i<num_elem_nodes;i++){
      placer = i*dim;     // hold the place in B for each node
      solid_B[0*stride + placer + 0] = dN[0*num_elem_nodes + i];
      solid_B[1*stride + placer + 1] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 2] = dN[2*num_elem_nodes + i];
      solid_B[3*stride + placer + 0] = dN[1*num_elem_nodes + i];
      solid_B[3*stride + placer + 1] = dN[0*num_elem_nodes + i];
      solid_B[4*stride + placer + 1] = dN[2*num_elem_nodes + i];
      solid_B[4*stride + placer + 2] = dN[1*num_elem_nodes + i];
      solid_B[5*stride + placer + 0] = dN[2*num_elem_nodes + i];
      solid_B[5*stride + placer + 2] = dN[0*num_elem_nodes + i];
    };
  };

  if(dim==2){
    for(int_t i=0;i<3*dim*num_elem_nodes;++i)
      solid_B[i] = 0.0;
    for(int_t i=0;i<num_elem_nodes;i++){
      placer = i*dim;     // hold the place in B for each node
      solid_B[0*stride + placer + 0] = dN[0*num_elem_nodes + i];
      solid_B[1*stride + placer + 1] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 0] = dN[1*num_elem_nodes + i];
      solid_B[2*stride + placer + 1] = dN[0*num_elem_nodes + i];
    };
  };

//  std::cout << " B " << std::endl;
//  for(int_t i=0;i<6;++i){
//    for(int_t j=0;j<num_elem_nodes*dim;++j)
//      std::cout << " " << solid_B[i*(num_elem_nodes*dim)+j];
//    std::cout << std::endl;
//  }

}


}// end global namespace


}// End DICe Namespace
