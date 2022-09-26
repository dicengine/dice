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

#include <DICe_LocalShapeFunction.h>
#include <DICe_Schema.h>

namespace DICe {

using namespace DICe::field_enums;

DICE_LIB_DLL_EXPORT
Teuchos::RCP<Local_Shape_Function> shape_function_factory(Schema * schema){
  if(!schema){
    return Teuchos::rcp(new Affine_Shape_Function(schema));
  }else if(schema->shape_function_type()==DICe::AFFINE_SF){
    return Teuchos::rcp(new Affine_Shape_Function(schema));
  }else if(schema->shape_function_type()==DICe::QUADRATIC_SF){
    return Teuchos::rcp(new Quadratic_Shape_Function());
  }else if(schema->shape_function_type()==DICe::RIGID_BODY_SF){
    TEUCHOS_TEST_FOR_EXCEPTION(!schema->get_params()->isParameter(DICe::camera_system_file),std::runtime_error,"");
    return Teuchos::rcp(new Rigid_Body_Shape_Function(schema->get_params()->get<std::string>(DICe::camera_system_file)));
  }else{
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Error, invalid shape function type");
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
Local_Shape_Function::update(const std::vector<scalar_t> & update){
  assert(update.size()==parameters_.size());
  for(size_t i=0;i<parameters_.size();++i)
    parameters_[i] += update[i];
}

bool
Local_Shape_Function::test_for_convergence(const std::vector<scalar_t> & old_parameters,
  const scalar_t & tol){
  assert(old_parameters.size()==parameters_.size());
  bool converged = true;
  for(int_t i=0;i<num_params_;++i){
    if(std::abs(parameters_[i] - old_parameters[i]) >= tol){
      converged = false;
    }
  }
  return converged;
}

void
Local_Shape_Function::reset_fields(Schema * schema){
  assert(schema);
  // iterate the fields and save off the values
  std::map<Field_Spec,size_t>::const_iterator it = spec_map_.begin();
  const std::map<Field_Spec,size_t>::const_iterator it_end = spec_map_.end();
  for(;it!=it_end;++it)
    schema->mesh()->get_field(it->first)->put_scalar(0.0);
}

void
Local_Shape_Function::clear(){
  for(int_t i=0;i<num_params_;++i)
    parameters_[i] = 0.0;
}

void
Local_Shape_Function::clone(Teuchos::RCP<Local_Shape_Function> shape_function){
  assert(shape_function->num_params()==num_params_);
  for(int_t i=0;i<num_params_;++i)
    (*this)(i) = (*shape_function)(i);
}

void
Local_Shape_Function::initialize_parameters_from_fields(Schema * schema,
  const int_t subset_gid){
  assert(schema);
  std::map<Field_Spec,size_t>::const_iterator it = spec_map_.begin();
  const std::map<Field_Spec,size_t>::const_iterator it_end = spec_map_.end();
  std::stringstream dbg_str;
  dbg_str << "Subset initialized from subset gid " << subset_gid << " with values:";
  for(;it!=it_end;++it){
    (*this)(it->first) = schema->global_field_value(subset_gid,it->first);
    dbg_str << " " << parameter(it->first);
  }
  DEBUG_MSG(dbg_str.str());
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
  // set the residual indicies
  if(spec_map_.find(SUBSET_DISPLACEMENT_X_FS)!=spec_map_.end()) {
    dx_ind_ = spec_map_.find(SUBSET_DISPLACEMENT_X_FS)->second;
  }
  if(spec_map_.find(SUBSET_DISPLACEMENT_Y_FS)!=spec_map_.end()) {
    dy_ind_ = spec_map_.find(SUBSET_DISPLACEMENT_Y_FS)->second;
  }
  if(spec_map_.find(ROTATION_Z_FS)!=spec_map_.end()) {
    has_rotz_ = true;
    rotz_ind_ = spec_map_.find(ROTATION_Z_FS)->second;
  }
  if(spec_map_.find(NORMAL_STRETCH_XX_FS)!=spec_map_.end()) {
    has_nsxx_ = true;
    nsxx_ind_ = spec_map_.find(NORMAL_STRETCH_XX_FS)->second;
  }
  if(spec_map_.find(NORMAL_STRETCH_YY_FS)!=spec_map_.end()) {
    has_nsyy_ = true;
    nsyy_ind_ = spec_map_.find(NORMAL_STRETCH_YY_FS)->second;
  }
  if(spec_map_.find(SHEAR_STRETCH_XY_FS)!=spec_map_.end()) {
    has_ssxy_ = true;
    ssxy_ind_ = spec_map_.find(SHEAR_STRETCH_XY_FS)->second;
  }
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
  static scalar_t dispx=0.0,dispy=0.0,theta=0.0,dudx=0.0,dvdy=0.0,gxy=0.0;
  static scalar_t cost=0.0,sint=0.0;

  dispx = parameters_[dx_ind_];
  dispy = parameters_[dy_ind_];
  theta = has_rotz_ ? parameters_[rotz_ind_] : 0.0;
  dudx  = has_nsxx_ ? parameters_[nsxx_ind_] : 0.0;
  dvdy  = has_nsyy_ ? parameters_[nsyy_ind_] : 0.0;
  gxy   = has_ssxy_ ? parameters_[ssxy_ind_] : 0.0;
  cost = cos(theta);
  sint = sin(theta);
  dx = x - cx;
  dy = y - cy;
  Dx = (1.0+dudx)*dx + gxy*dy;
  Dy = (1.0+dvdy)*dy + gxy*dx;
  // mapped location
  out_x = cost*Dx - sint*Dy + dispx + cx;
  out_y = sint*Dx + cost*Dy + dispy + cy;
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
Affine_Shape_Function::map_to_u_v_theta(const scalar_t & cx,
  const scalar_t & cy,
  scalar_t & out_u,
  scalar_t & out_v,
  scalar_t & out_theta){
  // note cx and cy not used for this shape function
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
  theta = has_rotz_ ? parameters_[rotz_ind_] : 0.0;
  dudx  = has_nsxx_ ? parameters_[nsxx_ind_] : 0.0;
  dvdy  = has_nsyy_ ? parameters_[nsyy_ind_] : 0.0;
  gxy   = has_ssxy_ ? parameters_[ssxy_ind_] : 0.0;
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

  //if (has_dx)
  residuals[dx_ind_] = Gx;
  //if (has_dy)
  residuals[dy_ind_] = Gy;
  if (has_rotz_)
    residuals[rotz_ind_] = delTheta;
  if (has_nsxx_)
    residuals[nsxx_ind_] = delEx;
  if (has_nsyy_)
    residuals[nsyy_ind_] = delEy;
  if (has_ssxy_)
    residuals[ssxy_ind_] = delGxy;
}

void
Affine_Shape_Function::high_order_terms(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  const scalar_t & gx,
  const scalar_t & gy,
  std::vector<scalar_t> & high_order_terms,
  const bool use_ref_grads){
  assert((int_t)high_order_terms.size()==4);

  static scalar_t dx=0.0,dy=0.0,Dx=0.0,Dy=0.0,delTheta=0.0,delEx=0.0,delEy=0.0,delGxy=0.0;
  static scalar_t Gx=0.0,Gy=0.0;
  static scalar_t theta=0.0,dudx=0.0,dvdy=0.0,gxy=0.0,cosTheta=0.0,sinTheta=0.0;
  theta = has_rotz_ ? parameters_[rotz_ind_] : 0.0;
  dudx  = has_nsxx_ ? parameters_[nsxx_ind_] : 0.0;
  dvdy  = has_nsyy_ ? parameters_[nsyy_ind_] : 0.0;
  gxy   = has_ssxy_ ? parameters_[ssxy_ind_] : 0.0;
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

  high_order_terms[0] = Gx*(sinTheta*Dy - cosTheta*Dx) + Gy*(-1.0*sinTheta*Dx - cosTheta*Dy);
  high_order_terms[1] = Gy*cosTheta*dx - Gx*sinTheta*dx;
  high_order_terms[2] = -1.0*Gx*cosTheta*dy - Gy*sinTheta*dy;
  high_order_terms[3] = Gx*(-1.0*sinTheta*dy - cosTheta*dx) + Gy*(cosTheta*dy - sinTheta*dx);
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

Quadratic_Shape_Function::Quadratic_Shape_Function(){
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_A_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_B_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_C_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_D_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_E_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_F_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_G_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_H_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_I_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_J_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_K_FS,spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec,size_t>(QUAD_L_FS,spec_map_.size()));
  num_params_ = spec_map_.size();
  assert(num_params_==12);
  parameters_.resize(num_params_);
  deltas_.resize(num_params_);
  // initialize the parameter values
  clear();
  // initialize the deltas:
  deltas_[0]  = 0.001;
  deltas_[1]  = 0.001;
  deltas_[2]  = 1.0E-6;
  deltas_[3]  = 1.0E-6;
  deltas_[4]  = 1.0E-6;
  deltas_[5]  = 1.0;
  deltas_[6]  = 0.001;
  deltas_[7]  = 0.001;
  deltas_[8]  = 1.0E-6;
  deltas_[9]  = 1.0E-6;
  deltas_[10] = 1.0E-6;
  deltas_[11] = 1.0;
}

void
Quadratic_Shape_Function::clear(){
  Local_Shape_Function::clear();
  (*this)(QUAD_A_FS) = 1.0;
  (*this)(QUAD_H_FS) = 1.0;
}

void
Quadratic_Shape_Function::reset_fields(Schema * schema){
  Local_Shape_Function::reset_fields(schema);
  schema->mesh()->get_field(QUAD_A_FS)->put_scalar(1.0);
  schema->mesh()->get_field(QUAD_H_FS)->put_scalar(1.0);
}

void
Quadratic_Shape_Function::map(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  scalar_t & out_x,
  scalar_t & out_y){
  const scalar_t dx = x - cx;
  const scalar_t dy = y - cy;
  out_x = parameter(QUAD_A_FS)*dx + parameter(QUAD_B_FS)*dy + parameter(QUAD_C_FS)*dx*dy + parameter(QUAD_D_FS)*dx*dx + parameter(QUAD_E_FS)*dy*dy + parameter(QUAD_F_FS) + cx;
  out_y = parameter(QUAD_G_FS)*dx + parameter(QUAD_H_FS)*dy + parameter(QUAD_I_FS)*dx*dy + parameter(QUAD_J_FS)*dx*dx + parameter(QUAD_K_FS)*dy*dy + parameter(QUAD_L_FS) + cy;
}

void
Quadratic_Shape_Function::add_translation(const scalar_t & u,
  const scalar_t & v){
  (*this)(QUAD_F_FS) += u;
  (*this)(QUAD_L_FS) += v;
}

void
Quadratic_Shape_Function::insert_motion(const scalar_t & u,
  const scalar_t & v,
  const scalar_t & theta){
  // clear the other parameters:
  clear();
  (*this)(QUAD_F_FS) = u;
  (*this)(QUAD_L_FS) = v;
  (*this)(QUAD_A_FS) = std::cos(theta);
  (*this)(QUAD_B_FS) = -std::sin(theta);
  (*this)(QUAD_G_FS) = std::sin(theta);
  (*this)(QUAD_H_FS) = std::cos(theta);
}

void
Quadratic_Shape_Function::insert_motion(const scalar_t & u,
  const scalar_t & v){
  // clear the other parameters:
  clear();
  (*this)(QUAD_F_FS) = u;
  (*this)(QUAD_L_FS) = v;
}

void
Quadratic_Shape_Function::map_to_u_v_theta(const scalar_t & cx,
  const scalar_t & cy,
  scalar_t & out_u,
  scalar_t & out_v,
  scalar_t & out_theta){
  scalar_t cxp=0.0,cyp=0.0;
  // map the centroid
  map(cx,cy,cx,cy,cxp,cyp);
  out_u = cxp - cx;
  out_v = cyp - cy;
  // map a point 5 pixels to the right of the centroid
  // the 5 is arbitrary
  scalar_t rxp=0.0,ryp=0.0;
  map(cx+5.0,cy,cx,cy,rxp,ryp);
  // get the angle between two rays...
  const scalar_t ax = rxp - cxp;
  const scalar_t ay = ryp - cyp;
  const scalar_t mag_a = std::sqrt(ax*ax+ay*ay);
  const scalar_t bx = 5.0;
  const scalar_t by = 0.0;
  const scalar_t mag_b = 5.0;
  const scalar_t a_dot_b_over_mags = (ax*bx + ay*by)/(mag_a*mag_b);
  // if the change in y is negative, have to add pi to account for accute angle measured
  // from the other direction in terms of clockwise
  out_theta = ay < 0.0 ? DICE_TWOPI - std::acos(a_dot_b_over_mags) : std::acos(a_dot_b_over_mags);
}

void
Quadratic_Shape_Function::residuals(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  const scalar_t & gx,
  const scalar_t & gy,
  std::vector<scalar_t> & residuals,
  const bool use_ref_grads){
  const scalar_t dx = x - cx;
  const scalar_t dy = y - cy;
  assert((int_t)residuals.size()==num_params_);
  scalar_t Gx = gx;
  scalar_t Gy = gy;
  if(use_ref_grads){
    scalar_t u=0.0,v=0.0,theta=0.0;
    map_to_u_v_theta(cx,cy,u,v,theta);
    scalar_t cosTheta = std::cos(theta);
    scalar_t sinTheta = std::sin(theta);
    Gx = cosTheta*gx - sinTheta*gy;
    Gy = sinTheta*gx + cosTheta*gy;
  }
  residuals[spec_map_.find(QUAD_A_FS)->second] = Gx*dx;
  residuals[spec_map_.find(QUAD_B_FS)->second] = Gx*dy;
  residuals[spec_map_.find(QUAD_C_FS)->second] = Gx*dx*dy;
  residuals[spec_map_.find(QUAD_D_FS)->second] = Gx*dx*dx;
  residuals[spec_map_.find(QUAD_E_FS)->second] = Gx*dy*dy;
  residuals[spec_map_.find(QUAD_F_FS)->second] = Gx;
  residuals[spec_map_.find(QUAD_G_FS)->second] = Gy*dx;
  residuals[spec_map_.find(QUAD_H_FS)->second] = Gy*dy;
  residuals[spec_map_.find(QUAD_I_FS)->second] = Gy*dx*dy;
  residuals[spec_map_.find(QUAD_J_FS)->second] = Gy*dx*dx;
  residuals[spec_map_.find(QUAD_K_FS)->second] = Gy*dy*dy;
  residuals[spec_map_.find(QUAD_L_FS)->second] = Gy;
}

void
Quadratic_Shape_Function::save_fields(Schema * schema,
  const int_t subset_gid){
  Local_Shape_Function::save_fields(schema,subset_gid);
  // since u, v, and theta are not explicitly parameters, need to save them
  // manually here
  scalar_t u=0.0,v=0.0,theta=0.0;
  const scalar_t cx = schema->global_field_value(subset_gid,SUBSET_COORDINATES_X_FS);
  const scalar_t cy = schema->global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS);
  map_to_u_v_theta(cx,cy,u,v,theta);
  schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS) = u;
  schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS) = v;
  schema->global_field_value(subset_gid,ROTATION_Z_FS) = theta;
}

void
Quadratic_Shape_Function::update_params_for_centroid_change(const scalar_t & delta_x,
  const scalar_t & delta_y){
  const scalar_t A = parameter(QUAD_A_FS);
  const scalar_t B = parameter(QUAD_B_FS);
  const scalar_t C = parameter(QUAD_C_FS);
  const scalar_t D = parameter(QUAD_D_FS);
  const scalar_t E = parameter(QUAD_E_FS);
  const scalar_t G = parameter(QUAD_G_FS);
  const scalar_t H = parameter(QUAD_H_FS);
  const scalar_t I = parameter(QUAD_I_FS);
  const scalar_t J = parameter(QUAD_J_FS);
  const scalar_t K = parameter(QUAD_K_FS);

  (*this)(QUAD_A_FS) += C*delta_y + 2.0*D*delta_x;
  (*this)(QUAD_B_FS) += C*delta_x + 2.0*E*delta_y;
  (*this)(QUAD_F_FS) += A*delta_x + B*delta_y + C*delta_x*delta_y + D*delta_x*delta_x + E*delta_y*delta_y - delta_x;
  (*this)(QUAD_G_FS) += I*delta_y + 2.0*J*delta_x;
  (*this)(QUAD_H_FS) += I*delta_x + 2.0*K*delta_y;
  (*this)(QUAD_L_FS) += G*delta_x + H*delta_y + I*delta_x*delta_y + J*delta_x*delta_x + K*delta_y*delta_y - delta_y;
}

Projection_Shape_Function::Projection_Shape_Function(const std::string & system_file,
  const size_t source_camera_id,
  const size_t target_camera_id):
  Local_Shape_Function(),
  source_cam_id_(source_camera_id),
  target_cam_id_(target_camera_id),
  camera_system_(Teuchos::rcp(new Camera_System(system_file))){
  spec_map_.insert(std::pair<Field_Spec, size_t>(PROJECTION_Z_FS, spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec, size_t>(PROJECTION_PHI_FS, spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec, size_t>(PROJECTION_THETA_FS, spec_map_.size()));
  num_params_ = spec_map_.size();
  assert(num_params_ == 3);
  parameters_.resize(num_params_);
  deltas_.resize(num_params_);
  // initialize the parameter values
  clear();
  // initialize the deltas:
  deltas_[ZP] = 0.005;
  deltas_[THETA] = 0.001;
  deltas_[PHI] = 0.001;
}

void
Projection_Shape_Function::clear() {
  Local_Shape_Function::clear();
  (*this)(PROJECTION_Z_FS) = 100.0;
}

void
Projection_Shape_Function::reset_fields(Schema * schema) {
  //difference between this and clear?
  Local_Shape_Function::reset_fields(schema);
  schema->mesh()->get_field(PROJECTION_Z_FS)->put_scalar(100.0);
}

void
Projection_Shape_Function::map(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  scalar_t & out_x,
  scalar_t & out_y) {
  const std::vector<scalar_t> source_x(1,x);
  const std::vector<scalar_t> source_y(1,y);
  // cx and cy are not used
  std::vector<scalar_t> target_x(1,0);
  std::vector<scalar_t> target_y(1,0);
  camera_system_->camera_to_camera_projection(source_cam_id_,target_cam_id_,
    source_x,source_y,target_x,target_y,parameters_);
  out_x = target_x[0];
  out_y = target_y[0];
}

void
Projection_Shape_Function::residuals(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  const scalar_t & gx,
  const scalar_t & gy,
  std::vector<scalar_t> & residuals,
  const bool use_ref_grads) {
  const std::vector<scalar_t> source_x(1,x);
  const std::vector<scalar_t> source_y(1,y);
  // cx and cy are not used
  // the target x and y are not used, but computed as part of the mapping
  std::vector<scalar_t> target_x(1,0);
  std::vector<scalar_t> target_y(1,0);
  std::vector<std::vector<scalar_t> > dx(num_params_,std::vector<scalar_t>(1,0));
  std::vector<std::vector<scalar_t> > dy(num_params_,std::vector<scalar_t>(1,0));
  camera_system_->camera_to_camera_projection(source_cam_id_,target_cam_id_,
    source_x,source_y,target_x,target_y,parameters_,dx,dy);

  residuals[spec_map_.find(PROJECTION_Z_FS)->second] = gx * dx[ZP][0] + gy * dy[ZP][0];
  residuals[spec_map_.find(PROJECTION_THETA_FS)->second] = gx * dx[THETA][0] + gy * dy[THETA][0];
  residuals[spec_map_.find(PROJECTION_PHI_FS)->second] = gx *dx[PHI][0] + gy * dy[PHI][0];
}

void
Projection_Shape_Function::save_fields(Schema * schema,
  const int_t subset_gid) {
  Local_Shape_Function::save_fields(schema, subset_gid);
}

Rigid_Body_Shape_Function::Rigid_Body_Shape_Function(const std::string & system_file):
  Local_Shape_Function(),
  camera_system_(Teuchos::rcp(new Camera_System(system_file))){
  TEUCHOS_TEST_FOR_EXCEPTION(camera_system_->num_cameras()!=1,std::runtime_error,"");
  facet_params_ = camera_system_->camera(0)->get_facet_params();
  TEUCHOS_TEST_FOR_EXCEPTION(facet_params_.size()!=3,std::runtime_error,"");
  spec_map_.insert(std::pair<Field_Spec, size_t>(ROT_TRANS_3D_ANG_X_FS, spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec, size_t>(ROT_TRANS_3D_ANG_Y_FS, spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec, size_t>(ROT_TRANS_3D_ANG_Z_FS, spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec, size_t>(ROT_TRANS_3D_TRANS_X_FS, spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec, size_t>(ROT_TRANS_3D_TRANS_Y_FS, spec_map_.size()));
  spec_map_.insert(std::pair<Field_Spec, size_t>(ROT_TRANS_3D_TRANS_Z_FS, spec_map_.size()));
  num_params_ = spec_map_.size();
  assert(num_params_ == 6);
  parameters_.resize(num_params_);
  deltas_.resize(num_params_);
  // initialize the parameter values
  clear();
  // initialize the deltas:
  deltas_[ANGLE_X] = 0.005; // these are all in units of physical space (not pixels)
  deltas_[ANGLE_Y] = 0.005;
  deltas_[ANGLE_Z] = 0.005;
  deltas_[TRANS_X] = 0.1;
  deltas_[TRANS_Y] = 0.1;
  deltas_[TRANS_Z] = 0.1;
}

void
Rigid_Body_Shape_Function::clear() {
  Local_Shape_Function::clear();
}

void
Rigid_Body_Shape_Function::reset_fields(Schema * schema) {
  Local_Shape_Function::reset_fields(schema);
}

void
Rigid_Body_Shape_Function::map(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  scalar_t & out_x,
  scalar_t & out_y) {
  const std::vector<scalar_t> source_x(1,x);
  const std::vector<scalar_t> source_y(1,y);
  // cx and cy are not used (they are included to keep a consistent interfacet to the map method for all shape functions)
  std::vector<scalar_t> mapped_x(1,0.0);
  std::vector<scalar_t> mapped_y(1,0.0);
  // project from the image out to the facet (which gives coordinates in terms of camera 0's coordinate system)
  // then convert the camera 0 coordinates to the world coordinates (which correspond to the surface of the calibration plate or specimen surface)
  // then apply a rigid body transformation to get the deformed coordinates, still in terms of the calibration plate's coordinate system
  // then convert the world coordinates back to camera 0's coordinate system
  // then convert from camera 0 coordinates to sensor coordinates
  // then from sensor coordinates to the updated image coordinates
  // all of this is accomplished with the following call
  camera_system_->camera_to_camera_projection(0,0,source_x,source_y,mapped_x,mapped_y,facet_params_,parameters_);
  out_x = mapped_x[0];
  out_y = mapped_y[0];
}

void
Rigid_Body_Shape_Function::residuals(const scalar_t & x,
  const scalar_t & y,
  const scalar_t & cx,
  const scalar_t & cy,
  const scalar_t & gx,
  const scalar_t & gy,
  std::vector<scalar_t> & residuals,
  const bool use_ref_grads) {
  const std::vector<scalar_t> source_x(1,x);
  const std::vector<scalar_t> source_y(1,y);
  // cx and cy are not used
  // mapped x and y are not used either, but returned by the camera_to_camera projection
  std::vector<scalar_t> mapped_x(1,0);
  std::vector<scalar_t> mapped_y(1,0);
  std::vector<std::vector<scalar_t> > dx(num_params_,std::vector<scalar_t>(1,0));
  std::vector<std::vector<scalar_t> > dy(num_params_,std::vector<scalar_t>(1,0));
  camera_system_->camera_to_camera_projection(0,0,source_x,source_y,mapped_x,mapped_y,facet_params_,dx,dy,parameters_);
  residuals[spec_map_.find(ROT_TRANS_3D_ANG_X_FS)->second] = gx * dx[ANGLE_X][0] + gy * dy[ANGLE_X][0];
  residuals[spec_map_.find(ROT_TRANS_3D_ANG_Y_FS)->second] = gx * dx[ANGLE_Y][0] + gy * dy[ANGLE_Y][0];
  residuals[spec_map_.find(ROT_TRANS_3D_ANG_Z_FS)->second] = gx * dx[ANGLE_Z][0] + gy * dy[ANGLE_Z][0];
  residuals[spec_map_.find(ROT_TRANS_3D_TRANS_X_FS)->second] = gx * dx[TRANS_X][0] + gy * dy[TRANS_X][0];
  residuals[spec_map_.find(ROT_TRANS_3D_TRANS_Y_FS)->second] = gx * dx[TRANS_Y][0] + gy * dy[TRANS_Y][0];
  residuals[spec_map_.find(ROT_TRANS_3D_TRANS_Z_FS)->second] = gx * dx[TRANS_Z][0] + gy * dy[TRANS_Z][0];
}

void
Rigid_Body_Shape_Function::save_fields(Schema * schema,
  const int_t subset_gid){
  Local_Shape_Function::save_fields(schema,subset_gid);
  // since u, v, and theta are not explicitly parameters, need to save them
  // manually here
  scalar_t u=0.0,v=0.0,theta=0.0;
  const scalar_t cx = schema->global_field_value(subset_gid,SUBSET_COORDINATES_X_FS);
  const scalar_t cy = schema->global_field_value(subset_gid,SUBSET_COORDINATES_Y_FS);
  map_to_u_v_theta(cx,cy,u,v,theta);
  schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_X_FS) = u;
  schema->global_field_value(subset_gid,SUBSET_DISPLACEMENT_Y_FS) = v;
  schema->global_field_value(subset_gid,ROTATION_Z_FS) = theta;
}

void
Rigid_Body_Shape_Function::map_to_u_v_theta(const scalar_t & cx,
  const scalar_t & cy,
  scalar_t & out_u,
  scalar_t & out_v,
  scalar_t & out_theta){
  scalar_t cxp=0.0,cyp=0.0;
  // map the centroid
  map(cx,cy,cx,cy,cxp,cyp);
  out_u = cxp - cx;
  out_v = cyp - cy;
  // map a point 5 pixels to the right of the centroid
  // the 5 is arbitrary
  scalar_t rxp=0.0,ryp=0.0;
  map(cx+5.0,cy,cx,cy,rxp,ryp);
  // get the angle between two rays...
  const scalar_t ax = rxp - cxp;
  const scalar_t ay = ryp - cyp;
  const scalar_t mag_a = std::sqrt(ax*ax+ay*ay);
  const scalar_t bx = 5.0;
  const scalar_t by = 0.0;
  const scalar_t mag_b = 5.0;
  const scalar_t a_dot_b_over_mags = (ax*bx + ay*by)/(mag_a*mag_b);
  // if the change in y is negative, have to add pi to account for accute angle measured
  // from the other direction in terms of clockwise
  out_theta = ay < 0.0 ? DICE_TWOPI - std::acos(a_dot_b_over_mags) : std::acos(a_dot_b_over_mags);
}

}// End DICe Namespace
