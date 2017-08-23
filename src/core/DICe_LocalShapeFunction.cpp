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

#include <DICe_LocalShapeFunction.h>
#include <DICe_MeshEnums.h>
#include <DICe_Schema.h>

namespace DICe {

using namespace DICe::mesh::field_enums;

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Local_Shape_Function> shape_function_factory(Schema * schema){
  if(!schema){
    return Teuchos::rcp(new Affine_Shape_Function(schema));
  }
  else if(schema->affine_matrix_enabled()){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, this method is not ready yet!");
  }else{
    return Teuchos::rcp(new Affine_Shape_Function(schema));
  }
}

void
Local_Shape_Function::save_fields(Schema * schema,
  const int_t subset_gid){
  assert(schema);
  // iterate the fields and save off the values
  std::map<Field_Spec,size_t>::const_iterator it = spec_map_.begin();
  const std::map<Field_Spec,size_t>::const_iterator it_end = spec_map_.end();
  for(;it!=it_end;++it)
    schema->global_field_value(subset_gid,it->first) = parameters_[it->second];
}

void
Local_Shape_Function::create_fields(Schema * schema){
  assert(schema);
  // iterate the field specs and create the necessary fields
  std::map<Field_Spec,size_t>::const_iterator it = spec_map_.begin();
  const std::map<Field_Spec,size_t>::const_iterator it_end = spec_map_.end();
  for(;it!=it_end;++it){
    schema->mesh()->create_field(it->first);
    // creat the nm1 field as well
    DICe::mesh::field_enums::Field_Spec fs_nm1(it->first.get_field_type(),it->first.get_name(),it->first.get_rank(),DICe::mesh::field_enums::STATE_N_MINUS_ONE,false,true);
    schema->mesh()->create_field(fs_nm1);
  }
}

void
Local_Shape_Function::update(const std::vector<scalar_t> & update){
  assert(update.size()==parameters_.size());
  for(size_t i=0;i<parameters_.size();++i)
    parameters_[i] += update[i];
}

Affine_Shape_Function::Affine_Shape_Function(const bool enable_rotation,
  const bool enable_normal_strain,
  const bool enable_shear_strain,
  const scalar_t & delta_disp,
  const scalar_t & delta_theta){
  init(enable_rotation,enable_normal_strain,enable_shear_strain,delta_disp,delta_theta);
}

Affine_Shape_Function::Affine_Shape_Function(Schema * schema):
  Local_Shape_Function()
{
  bool rotation_enabled = true;
  bool normal_strain_enabled = true;
  bool shear_strain_enabled = true;
  scalar_t delta_disp = 1.0;
  scalar_t delta_theta = 0.1;
  if(schema){
    rotation_enabled = schema->rotation_enabled();
    normal_strain_enabled = schema->normal_strain_enabled();
    shear_strain_enabled = schema->shear_strain_enabled();
    delta_disp = schema->robust_delta_disp();
    delta_theta = schema->robust_delta_theta();
    // translation is required
    assert(schema->translation_enabled());
  }
  init(rotation_enabled,normal_strain_enabled,shear_strain_enabled,delta_disp,delta_theta);
}

void
Affine_Shape_Function::init(const bool enable_rotation,
  const bool enable_normal_strain,
  const bool enable_shear_strain,
  const scalar_t & delta_disp,
  const scalar_t & delta_theta){
  spec_map_.insert(std::pair<Field_Spec,size_t>(SUBSET_DISPLACEMENT_X_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(SUBSET_DISPLACEMENT_Y_FS,spec_map_.size()));
  if(enable_rotation)
    spec_map_.insert(std::pair<Field_Spec,size_t>(ROTATION_Z_FS,spec_map_.size()));
  if(enable_normal_strain){
    spec_map_.insert(std::pair<Field_Spec,size_t>(NORMAL_STRETCH_XX_FS,spec_map_.size()));
    spec_map_.insert(std::pair<Field_Spec,size_t>(NORMAL_STRETCH_YY_FS,spec_map_.size()));
  }
  if(enable_shear_strain)
    spec_map_.insert(std::pair<Field_Spec,size_t>(SHEAR_STRETCH_XY_FS,spec_map_.size()));
  num_params_ = spec_map_.size();
  parameters_.resize(num_params_);
  deltas_.resize(num_params_);
  // initialize the parameter values
  clear();
  for(int_t i=0;i<num_params_;++i)
    deltas_[i] = delta_theta;
  deltas_[spec_map_.find(SUBSET_DISPLACEMENT_X_FS)->second] = delta_disp;
  deltas_[spec_map_.find(SUBSET_DISPLACEMENT_Y_FS)->second] = delta_disp;
}

void
Affine_Shape_Function::reset_fields(Schema * schema){
  assert(schema);
  // iterate the fields and save off the values
  std::map<Field_Spec,size_t>::const_iterator it = spec_map_.begin();
  const std::map<Field_Spec,size_t>::const_iterator it_end = spec_map_.end();
  for(;it!=it_end;++it)
    schema->mesh()->get_field(it->first)->put_scalar(0.0);
}

void
Affine_Shape_Function::map(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  scalar_t & out_x,
  scalar_t & out_y){

  static scalar_t dx=0.0,dy=0.0;
  static scalar_t Dx=0.0,Dy=0.0;
  static scalar_t cost;
  static scalar_t sint;
  cost = std::cos(parameter(ROTATION_Z_FS));
  sint = std::sin(parameter(ROTATION_Z_FS));
  dx = x - cx;
  dy = y - cy;
  Dx = (1.0+parameter(NORMAL_STRETCH_XX_FS))*dx + parameter(SHEAR_STRETCH_XY_FS)*dy;
  Dy = (1.0+parameter(NORMAL_STRETCH_YY_FS))*dy + parameter(SHEAR_STRETCH_XY_FS)*dx;
  // mapped location
  out_x = cost*Dx - sint*Dy + parameter(SUBSET_DISPLACEMENT_X_FS) + cx;
  out_y = sint*Dx + cost*Dy + parameter(SUBSET_DISPLACEMENT_Y_FS) + cy;
}


void
Affine_Shape_Function::clear(){
  for(int_t i=0;i<num_params_;++i)
    parameters_[i] = 0.0;
}

//FIXME make these check the field exists rather than if the schema has them enabled
void
Affine_Shape_Function::initialize_parameters_from_fields(Schema * schema,
  const int_t subset_gid){
  assert(schema);
  const Projection_Method projection = schema->projection_method();
  if(schema->translation_enabled()){
    DEBUG_MSG("Subset " << subset_gid << " Translation is enabled.");
    if(schema->frame_id() > schema->first_frame_id()+2 && projection == VELOCITY_BASED){
      (*this)(SUBSET_DISPLACEMENT_X_FS) = schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS) +
          (schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS)-schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_NM1_FS));
      (*this)(SUBSET_DISPLACEMENT_Y_FS) = schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS) +
          (schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS)-schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_NM1_FS));

    }
    else{
      (*this)(SUBSET_DISPLACEMENT_X_FS) = schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS);
      (*this)(SUBSET_DISPLACEMENT_Y_FS) = schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS);
    }
  }
  if(schema->rotation_enabled()){
    DEBUG_MSG("Subset " << subset_gid << " Rotation is enabled.");
    if(schema->frame_id() > schema->first_frame_id()+2 && projection == VELOCITY_BASED){
      (*this)(ROTATION_Z_FS) = schema->global_field_value(subset_gid,ROTATION_Z_FS) +
          (schema->global_field_value(subset_gid,ROTATION_Z_FS)-schema->global_field_value(subset_gid,ROTATION_Z_NM1_FS));
    }
    else{
      (*this)(ROTATION_Z_FS) = schema->global_field_value(subset_gid,ROTATION_Z_FS);
    }
  }
  if(schema->normal_strain_enabled()){
    DEBUG_MSG("Subset " << subset_gid << " Normal strain is enabled.");
    (*this)(NORMAL_STRETCH_XX_FS) = schema->global_field_value(subset_gid,NORMAL_STRETCH_XX_FS);
    (*this)(NORMAL_STRETCH_YY_FS) = schema->global_field_value(subset_gid,NORMAL_STRETCH_YY_FS);
  }
  if(schema->shear_strain_enabled()){
    DEBUG_MSG("Subset " << subset_gid << " Shear strain is enabled.");
    (*this)(SHEAR_STRETCH_XY_FS) = schema->global_field_value(subset_gid,SHEAR_STRETCH_XY_FS);
  }
//  if(sid!=subset_gid)
//    DEBUG_MSG("Subset " << subset_gid << " was initialized from the field values of subset " << sid);
//  else{
//    DEBUG_MSG("Projection Method: " << projection);
//    DEBUG_MSG("Subset " << subset_gid << " solution from prev. step: u " << schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS) <<
//      " v " << schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS) <<
//      " theta " << schema->global_field_value(subset_gid,ROTATION_Z_FS) <<
//      " e_x " << schema->global_field_value(subset_gid,NORMAL_STRETCH_XX_FS) <<
//      " e_y " << schema->global_field_value(subset_gid,NORMAL_STRETCH_YY_FS) <<
//      " g_xy " << schema->global_field_value(subset_gid,SHEAR_STRETCH_XY_FS));
//    DEBUG_MSG("Subset " << subset_gid << " solution from nm1 step: u " << schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_NM1_FS) <<
//      " v " << schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_NM1_FS) <<
//      " theta " << schema->global_field_value(subset_gid,ROTATION_Z_NM1_FS) <<
//      " e_x " << schema->global_field_value(subset_gid,NORMAL_STRETCH_XX_NM1_FS) <<
//      " e_y " << schema->global_field_value(subset_gid,NORMAL_STRETCH_YY_NM1_FS) <<
//      " g_xy " << schema->global_field_value(subset_gid,SHEAR_STRETCH_XY_NM1_FS));
//  }
  DEBUG_MSG("Subset initialized from subset gid " << subset_gid << " with values: u " << parameter(SUBSET_DISPLACEMENT_X_FS) <<
    " v " << parameter(SUBSET_DISPLACEMENT_Y_FS) <<
    " theta " << parameter(ROTATION_Z_FS) <<
    " e_x " << parameter(NORMAL_STRETCH_XX_FS) <<
    " e_y " << parameter(NORMAL_STRETCH_YY_FS) <<
    " g_xy " << parameter(SHEAR_STRETCH_XY_FS));
}

void
Affine_Shape_Function::add_translation(const scalar_t & u,
  const scalar_t & v){
  (*this)(SUBSET_DISPLACEMENT_X_FS) += u;
  (*this)(SUBSET_DISPLACEMENT_Y_FS) += v;
}

void
Affine_Shape_Function::insert_motion(const scalar_t & u,
  const scalar_t & v,
  const scalar_t & theta){
  (*this)(SUBSET_DISPLACEMENT_X_FS) = u;
  (*this)(SUBSET_DISPLACEMENT_Y_FS) = v;
  if(spec_map_.find(ROTATION_Z_FS)!=spec_map_.end())
    (*this)(ROTATION_Z_FS) = theta;
}

void
Affine_Shape_Function::insert_motion(const scalar_t & u,
  const scalar_t & v){
  (*this)(SUBSET_DISPLACEMENT_X_FS) = u;
  (*this)(SUBSET_DISPLACEMENT_Y_FS) = v;
}

void
Affine_Shape_Function::map_to_u_v_theta(const scalar_t & x,
  const scalar_t & y,
  scalar_t & out_u,
  scalar_t & out_v,
  scalar_t & out_theta){
  out_u = parameter(SUBSET_DISPLACEMENT_X_FS);
  out_v = parameter(SUBSET_DISPLACEMENT_Y_FS);
  out_theta = parameter(ROTATION_Z_FS);
}

void
Affine_Shape_Function::residuals(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  const scalar_t & gx,
  const scalar_t & gy,
  std::vector<scalar_t> & residuals,
  const bool use_ref_grads){
  assert((int_t)residuals.size()==num_params_);

  static scalar_t dx=0.0,dy=0.0,Dx=0.0,Dy=0.0,delTheta=0.0,delEx=0.0,delEy=0.0,delGxy=0.0;
  static scalar_t Gx=0.0,Gy=0.0;
  static scalar_t theta=0.0,dudx=0.0,dvdy=0.0,gxy=0.0,cosTheta=0.0,sinTheta=0.0;
  theta = parameter(ROTATION_Z_FS);
  dudx  = parameter(NORMAL_STRETCH_XX_FS);
  dvdy  = parameter(NORMAL_STRETCH_YY_FS);
  gxy   = parameter(SHEAR_STRETCH_XY_FS);
  cosTheta = std::cos(theta);
  sinTheta = std::sin(theta);

  dx = x - cx;
  dy = y - cy;
  Dx = (1.0+dudx)*(dx) + gxy*(dy);
  Dy = (1.0+dvdy)*(dy) + gxy*(dx);
  Gx = use_ref_grads ? cosTheta*gx - sinTheta*gy : gx;
  Gy = use_ref_grads ? sinTheta*gx + cosTheta*gy : gy;

  delTheta = Gx*(-sinTheta*Dx - cosTheta*Dy) + Gy*(cosTheta*Dx - sinTheta*Dy);
  //deldelTheta = Gx*(-cosTheta*Dx + sinTheta*Dy) + Gy*(-sinTheta*Dx - cosTheta*Dy);
  delEx = Gx*dx*cosTheta + Gy*dx*sinTheta;
  delEy = -Gx*dy*sinTheta + Gy*dy*cosTheta;
  delGxy = Gx*(cosTheta*dy - sinTheta*dx) + Gy*(sinTheta*dy + cosTheta*dx);

  if(spec_map_.find(SUBSET_DISPLACEMENT_X_FS)!=spec_map_.end())
    residuals[spec_map_.find(SUBSET_DISPLACEMENT_X_FS)->second] = Gx;
  if(spec_map_.find(SUBSET_DISPLACEMENT_Y_FS)!=spec_map_.end())
    residuals[spec_map_.find(SUBSET_DISPLACEMENT_Y_FS)->second] = Gy;
  if(spec_map_.find(ROTATION_Z_FS)!=spec_map_.end())
    residuals[spec_map_.find(ROTATION_Z_FS)->second] = delTheta;
  if(spec_map_.find(NORMAL_STRETCH_XX_FS)!=spec_map_.end())
    residuals[spec_map_.find(NORMAL_STRETCH_XX_FS)->second] = delEx;
  if(spec_map_.find(NORMAL_STRETCH_YY_FS)!=spec_map_.end())
    residuals[spec_map_.find(NORMAL_STRETCH_YY_FS)->second] = delEy;
  if(spec_map_.find(SHEAR_STRETCH_XY_FS)!=spec_map_.end())
    residuals[spec_map_.find(SHEAR_STRETCH_XY_FS)->second] = delGxy;
}


bool
Affine_Shape_Function::test_for_convergence(const std::vector<scalar_t> & old_parameters,
  const scalar_t & tol){
  // only tests the displacement and rotation fields
  assert(old_parameters.size()==parameters_.size());
  assert(spec_map_.find(SUBSET_DISPLACEMENT_X_FS)!=spec_map_.end());
  assert(spec_map_.find(SUBSET_DISPLACEMENT_Y_FS)!=spec_map_.end());
  const int_t u_id = spec_map_.find(SUBSET_DISPLACEMENT_X_FS)->second;
  const int_t v_id = spec_map_.find(SUBSET_DISPLACEMENT_Y_FS)->second;
  assert(u_id<num_params_);
  assert(v_id<num_params_);
  bool converged = true;
  if(std::abs(parameters_[u_id] - old_parameters[u_id]) >= tol){
    converged = false;
  }
  if(std::abs(parameters_[v_id] - old_parameters[v_id]) >= tol){
    converged = false;
  }
  if(spec_map_.find(ROTATION_Z_FS)!=spec_map_.end()){
    const int_t t_id = spec_map_.find(ROTATION_Z_FS)->second;
    assert(t_id<num_params_);
    if(std::abs(parameters_[t_id] - old_parameters[t_id]) >= tol){
      converged = false;
    }
  }
  return converged;
}



// TODO for Affine map, remember to create a virtual function save_fields that calls Local_Shape_Function::save_fields() then sets rotation and displacement as is done below

//(*deltas)[DOF_A] = 0.0001;
//(*deltas)[DOF_B] = 1.0E-5;
//(*deltas)[DOF_C] = 1.0;
//(*deltas)[DOF_D] = 1.0E-5;
//(*deltas)[DOF_E] = 0.0001;
//(*deltas)[DOF_F] = 1.0;
//(*deltas)[DOF_G] = 1.0E-5;
//(*deltas)[DOF_H] = 1.0E-5;
//(*deltas)[DOF_I] = 1.0E-5;


//global_field_value(subset_gid,AFFINE_A_FS) = (*deformation)[DOF_A];
//global_field_value(subset_gid,AFFINE_B_FS) = (*deformation)[DOF_B];
//global_field_value(subset_gid,AFFINE_C_FS) = (*deformation)[DOF_C];
//global_field_value(subset_gid,AFFINE_D_FS) = (*deformation)[DOF_D];
//global_field_value(subset_gid,AFFINE_E_FS) = (*deformation)[DOF_E];
//global_field_value(subset_gid,AFFINE_F_FS) = (*deformation)[DOF_F];
//global_field_value(subset_gid,AFFINE_G_FS) = (*deformation)[DOF_G];
//global_field_value(subset_gid,AFFINE_H_FS) = (*deformation)[DOF_H];
//global_field_value(subset_gid,AFFINE_I_FS) = (*deformation)[DOF_I];
//const scalar_t x = global_field_value(subset_gid,SUBSET_COORDINATES_X_FS);
//const scalar_t y = global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS);
//scalar_t u = 0.0,v=0.0,theta=0.0;
//affine_map_to_motion(x,y,u,v,theta,deformation);
//global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS) = u;
//global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS) = v;
//global_field_value(subset_gid,ROTATION_Z_FS) = theta;

//gx = gradGx[index];
//gy = gradGy[index];
//Gx = gx;
//Gy = gy;
//term_1 = (A*x+B*y+C);
//term_2 = (D*x+E*y+F);
//term_3 = (G*x+H*y+I);
//TEUCHOS_TEST_FOR_EXCEPTION(term_3==0.0,std::runtime_error,"");
//if(use_ref_grads){
//  dwxdx = A/term_3 - G*term_1/(term_3*term_3);
//  dwxdy = B/term_3 - H*term_1/(term_3*term_3);
//  dwydx = D/term_3 - G*term_2/(term_3*term_3);
//  dwydy = E/term_3 - H*term_2/(term_3*term_3);
//  TEUCHOS_TEST_FOR_EXCEPTION(std::abs(dwxdx*dwydy - dwxdy*dwydx) < 1.0E-12,std::runtime_error,"Error, det(dwdx) is zero.");
//  det_w = 1.0/(dwxdx*dwydy - dwxdy*dwydx);
//  dwxdx_i = det_w * dwydy;
//  dwxdy_i = -1.0 * det_w * dwxdy;
//  dwydx_i = -1.0 * det_w * dwydx;
//  dwydy_i = det_w * dwxdx;
//  Gx = dwxdx_i*gx + dwxdy_i*gy;
//  Gy = dwydx_i*gx + dwydy_i*gy;
//}
//resids[0] = Gx*x/term_3;
//resids[1] = Gx*y/term_3;
//resids[2] = Gx/term_3;
//resids[3] = Gy*x/term_3;
//resids[4] = Gy*y/term_3;
//resids[5] = Gy/term_3;
//resids[6] = -1.0*(Gx*term_1/(term_3*term_3) + Gy*term_2/(term_3*term_3))*x;
//resids[7] = -1.0*(Gx*term_1/(term_3*term_3) + Gy*term_2/(term_3*term_3))*y;
//resids[8] = -1.0*(Gx*term_1/(term_3*term_3) + Gy*term_2/(term_3*term_3));
//for(int_t i=0;i<q.size();++i){
//  q[i] += GmF*resids[i];
//  for(int_t j=0;j<q.size();++j)
//    tangent(i,j) += resids[i]*resids[j];
//}
//}

//DICE_LIB_DLL_EXPORT
//void affine_map_to_motion( const scalar_t & x,
//  const scalar_t & y,
//  scalar_t & out_u,
//  scalar_t & out_v,
//  scalar_t & out_theta,
//  const Teuchos::RCP<const std::vector<scalar_t> > & def){
//  if(def->size()==DICE_DEFORMATION_SIZE){
//    out_u = (*def)[DOF_U];
//    out_v = (*def)[DOF_V];
//    out_theta = (*def)[DOF_THETA];
//  }else if(def->size()==DICE_DEFORMATION_SIZE_AFFINE){
//    scalar_t x_prime = 0.0;
//    scalar_t y_prime = 0.0;
//    map_affine(x,y,x_prime,y_prime,def);
//    out_u = x_prime - x;
//    out_v = y_prime - y;
//    // estimate the rotation using the atan2 function (TODO this could be improved)
//    out_theta = std::atan2((*def)[DOF_B],(*def)[DOF_A]);
//  }
//}

//DICE_LIB_DLL_EXPORT
//void affine_add_translation( const scalar_t & u,
//  const scalar_t & v,
//  Teuchos::RCP<std::vector<scalar_t> > & def){
//  assert(def->size()==DICE_DEFORMATION_SIZE_AFFINE);
//  const scalar_t A = (*def)[DOF_A];
//  const scalar_t B = (*def)[DOF_B];
//  const scalar_t C = (*def)[DOF_C];
//  const scalar_t D = (*def)[DOF_D];
//  const scalar_t E = (*def)[DOF_E];
//  const scalar_t F = (*def)[DOF_F];
//  const scalar_t G = (*def)[DOF_G];
//  const scalar_t H = (*def)[DOF_H];
//  const scalar_t I = (*def)[DOF_I];
//  (*def)[DOF_A] = A + u*G;
//  (*def)[DOF_B] = B + u*H;
//  (*def)[DOF_C] = C + u*I;
//  (*def)[DOF_D] = D + v*G;
//  (*def)[DOF_E] = E + v*H;
//  (*def)[DOF_F] = F + v*I;
//}


/// maps an input point to an output point given the affine parameters
/// as defined by the affine matrix
/// \param x input x coord
/// \param y input y coord
/// \param out_x output x coord
/// \param out_y output y coord
/// \param def deformation vector
//DICE_LIB_DLL_EXPORT
//inline void map_affine( const scalar_t & x,
//  const scalar_t & y,
//  scalar_t & out_x,
//  scalar_t & out_y,
//  const Teuchos::RCP<const std::vector<scalar_t> > & def){
//  assert(def->size()==DICE_DEFORMATION_SIZE_AFFINE);
//  assert((*def)[6]*x + (*def)[7]*y + (*def)[8]!=0.0);
//  out_x = ((*def)[0]*x + (*def)[1]*y + (*def)[2])/((*def)[6]*x + (*def)[7]*y + (*def)[8]);
//  out_y = ((*def)[3]*x + (*def)[4]*y + (*def)[5])/((*def)[6]*x + (*def)[7]*y + (*def)[8]);
//}

/// computes u, v, and theta for an affine matrix deformation mapping
/// \param x input x coord
/// \param y input y coord
/// \param out_u output u displacement
/// \param out_v output v displacement
/// \param out_theta output rotation
/// \param def deformation vector
//DICE_LIB_DLL_EXPORT
//void affine_map_to_motion( const scalar_t & x,
//  const scalar_t & y,
//  scalar_t & out_u,
//  scalar_t & out_v,
//  scalar_t & out_theta,
//  const Teuchos::RCP<const std::vector<scalar_t> > & def);
//



}// End DICe Namespace
