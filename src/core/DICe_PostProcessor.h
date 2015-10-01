// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

#ifndef DICE_POSTPROCESSOR_H
#define DICE_POSTPROCESSOR_H

#include <DICe.h>
#include <DICe_Schema.h>

#include <Teuchos_ParameterList.hpp>

#include <cassert>

namespace DICe {

/// String parameter name
const char * const post_process_vsg_strain = "post_process_vsg_strain";
/// String parameter name
const char * const post_process_nlvc_strain = "post_process_nlvc_strain";
/// String parameter name
const char * const post_process_keys4_strain = "post_process_keys4_strain";
/// String parameter name
const char * const post_process_global_strain = "post_process_global_strain";
/// String Parameter name
const char * const strain_window_size_in_pixels = "strain_window_size_in_pixels";
/// String Parameter name
const char * const horizon_diameter_in_pixels = "horizon_diameter_in_pixels";
/// String Parameter name
const char * const spline_padding = "spline_padding";
/// String Parameter name
const char * const spline_order = "spline_order";

/// Number of post processor options
const int_t num_valid_post_processor_params = 4;
/// Set of all the valid post processors
const char * const valid_post_processor_params[num_valid_post_processor_params] = {
  post_process_vsg_strain,
  post_process_nlvc_strain,
  post_process_keys4_strain,
  post_process_global_strain
};

/// String field name
const char * const vsg_strain_xx = "VSG_STRAIN_XX";
/// String field name
const char * const vsg_strain_yy = "VSG_STRAIN_YY";
/// String field name
const char * const vsg_strain_xy = "VSG_STRAIN_XY";
/// String field name
const char * const vsg_dudx = "VSG_DUDX";
/// String field name
const char * const vsg_dudy = "VSG_DUDY";
/// String field name
const char * const vsg_dvdx = "VSG_DVDX";
/// String field name
const char * const vsg_dvdy = "VSG_DVDY";
/// String field name
const char * const global_strain_xx = "GLOBAL_STRAIN_XX";
/// String field name
const char * const global_strain_yy = "GLOBAL_STRAIN_YY";
/// String field name
const char * const global_strain_xy = "GLOBAL_STRAIN_XY";
/// String field name
const char * const global_dudx = "GLOBAL_DUDX";
/// String field name
const char * const global_dudy = "GLOBAL_DUDY";
/// String field name
const char * const global_dvdx = "GLOBAL_DVDX";
/// String field name
const char * const global_dvdy = "GLOBAL_DVDY";
/// String field name
const char * const nlvc_strain_xx = "NLVC_STRAIN_XX";
/// String field name
const char * const nlvc_strain_yy = "NLVC_STRAIN_YY";
/// String field name
const char * const nlvc_strain_xy = "NLVC_STRAIN_XY";
/// String field name
const char * const nlvc_dudx = "NLVC_DUDX";
/// String field name
const char * const nlvc_dudy = "NLVC_DUDY";
/// String field name
const char * const nlvc_dvdx = "NLVC_DVDX";
/// String field name
const char * const nlvc_dvdy = "NLVC_DVDY";
/// String field name
const char * const nlvc_integrated_alpha_x = "NLVC_INTEGRATED_ALPHA_X";
/// String field name
const char * const nlvc_integrated_alpha_y = "NLVC_INTEGRATED_ALPHA_Y";
/// String field name
const char * const nlvc_integrated_phi_x = "NLVC_INTEGRATED_PHI_X";
/// String field name
const char * const nlvc_integrated_phi_y = "NLVC_INTEGRATED_PHI_Y";
/// String field name
const char * const keys4_strain_xx = "KEYS4_STRAIN_XX";
/// String field name
const char * const keys4_strain_yy = "KEYS4_STRAIN_YY";
/// String field name
const char * const keys4_strain_xy = "KEYS4_STRAIN_XY";
/// String field name
const char * const keys4_dudx = "KEYS4_DUDX";
/// String field name
const char * const keys4_dudy = "KEYS4_DUDY";
/// String field name
const char * const keys4_dvdx = "KEYS4_DVDX";
/// String field name
const char * const keys4_dvdy = "KEYS4_DVDY";


/// \class DICe::Post_Processor
/// \brief A class for computing variables based on the field values of a schema and associated utilities
///
class DICE_LIB_DLL_EXPORT
Post_Processor{
public:
  /// \brief Default constructor
  /// \param schema The parent schema that is using this post processor
  /// \param name The name of this post processor, so that it can be referenced later or used in debugging
  Post_Processor(Schema * schema,
    const std::string & name);

  /// Pure virtual destructor
  virtual ~Post_Processor(){}

  /// operations that all post processors need to do after construction
  void initialize();

  /// Return a pointer to the vector of field names
  std::vector<std::string> * field_names(){
    return &field_names_;
  }

  /// \brief Field value accessor
  /// \param field_name The string name of the field to access
  /// \param global_id The global index of the requested element of the field
  scalar_t & field_value( const int_t global_id, const std::string & field_name){
    assert(fields_.find(field_name)!=fields_.end());
    assert(global_id < data_num_points_);
    return fields_.find(field_name)->second[global_id];
  }

  /// Tasks that are done once after initialization, but before execution
  virtual void pre_execution_tasks()=0;

  /// Returns the size of the strain window in pixels
  virtual const int_t strain_window_size()=0;

  /// Read in parameters for this post processor
  virtual void set_params(const Teuchos::RCP<Teuchos::ParameterList> & params)=0;

  /// Execute the post processor
  virtual void execute()=0;

protected:
  /// Pointer to the parent schema
  Schema * schema_;
  /// String name of this post processor
  std::string name_;
  /// Number of points in the field arrays
  int_t data_num_points_;
  /// The collection of field names specific to this post processor
  std::vector<std::string> field_names_;
  /// The fields that are computed by this post processor
  // TODO use the same data strcutures as the schema (Epetra/Tpetra...)
  std::map<std::string,std::vector<scalar_t> > fields_;
};

/// \class DICe::VSG_Strain_Post_Processor
/// \brief A specific instance of post processor that computes vurtual strain gauge (VSG) strain
///
/// The VSG strain is computed by doing a least-squares fit of the data and computing the
/// strain using the coefficients of the fitted polynomial. It is well know that for large
/// neighborhood sizes or VSG window size, this method will be very diffusive and smooth
/// features out of the strain profile.
class DICE_LIB_DLL_EXPORT
VSG_Strain_Post_Processor : public Post_Processor {

public:

  /// Default constructor
  /// \param schema Pointer to the parent schema that is using this post processor
  /// \param params Pointer to the set of parameters for this post processor
  VSG_Strain_Post_Processor(Schema * schema,
    const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Virtual destructor
  virtual ~VSG_Strain_Post_Processor(){}

  /// Set the parameters for this post processor
  virtual void set_params(const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Collect the neighborhoods of each of the points
  virtual void pre_execution_tasks();

  /// See base clase docutmentation
  virtual const int_t strain_window_size(){
    return window_size_;
  }

  /// Execute the post processor
  virtual void execute();

  /// See base class documentation
  using Post_Processor::field_names_;

  /// See base class documentation
  using Post_Processor::data_num_points_;

  /// See base class documentation
  using Post_Processor::schema_;

  /// See base class documentation
  using Post_Processor::field_value;

private:
  /// Window size for the virtual strain gauge (in pixels)
  int_t window_size_;
  /// The stride used for the compute vectors
  int_t vec_stride_;
  /// List of all the subsets within the window size
  Teuchos::ArrayRCP<int_t> neighbor_lists_;
  /// Number of neighbors for each subset
  Teuchos::ArrayRCP<int_t> num_neigh_;
  /// X distances to each of this subset's neighbors
  Teuchos::ArrayRCP<scalar_t> neighbor_distances_x_;
  /// Y distances to each of this subset's neighbors
  Teuchos::ArrayRCP<scalar_t> neighbor_distances_y_;
};


/// \class DICe::Global_Strain_Post_Processor
/// \brief A specific instance of post processor that computes strain in an FEM fashion
///
class DICE_LIB_DLL_EXPORT
Global_Strain_Post_Processor : public Post_Processor {

public:

  /// Default constructor
  /// \param schema Pointer to the parent schema that is using this post processor
  /// \param params Pointer to the set of parameters for this post processor
  Global_Strain_Post_Processor(Schema * schema,
    const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Virtual destructor
  virtual ~Global_Strain_Post_Processor(){}

  /// Set the parameters for this post processor
  virtual void set_params(const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Collect the neighborhoods of each of the points
  virtual void pre_execution_tasks();

  /// See base clase docutmentation
  virtual const int_t strain_window_size(){
    return mesh_size_;
  }

  /// Execute the post processor
  virtual void execute();

  /// See base class documentation
  using Post_Processor::field_names_;

  /// See base class documentation
  using Post_Processor::data_num_points_;

  /// See base class documentation
  using Post_Processor::schema_;

  /// See base class documentation
  using Post_Processor::field_value;

private:
  /// Window size for the virtual strain gauge (in pixels)
  int_t mesh_size_;
  /// The stride used for the compute vectors
  int_t vec_stride_;
};


/// \class DICe::NLVC_Strain_Post_Processor
/// \brief A specific instance of post processor that computes Nonlocal Vector Calculus (NLVC) strains
///
/// The NLVC Strain is computed by using the nonlocal correlary of the derivative operator
class DICE_LIB_DLL_EXPORT
NLVC_Strain_Post_Processor : public Post_Processor{

public:

  /// Default constructor
  /// \param schema Pointer to the parent schema that is using this post processor
  /// \param params the parameters to use for this post processor
  NLVC_Strain_Post_Processor(Schema * schema,
    const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Virtual destructor
  virtual ~NLVC_Strain_Post_Processor(){}

  /// Set the parameters for this post processor
  virtual void set_params(const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Collect the neighborhoods of each of the points
  virtual void pre_execution_tasks();

  /// See base clase docutmentation
  virtual const int_t strain_window_size(){return horizon_;}

  /// Execute the post processor
  virtual void execute();

  /// See base class documentation
  using Post_Processor::field_names_;

  /// See base class documentation
  using Post_Processor::data_num_points_;

  /// See base class documentation
  using Post_Processor::schema_;

  /// See base class documentation
  using Post_Processor::field_value;

private:
  /// Neighborhood diameter (circular distance around the point of interest where the interaction is non-negligible)
  int_t horizon_;
  /// The stride used for the compute vectors
  int_t vec_stride_;
  /// List of all the subsets within the window size
  Teuchos::ArrayRCP<int_t> neighbor_lists_;
  /// Number of neighbors for each subset
  Teuchos::ArrayRCP<int_t> num_neigh_;
  /// X distances to each of this subset's neighbors
  Teuchos::ArrayRCP<scalar_t> neighbor_distances_x_;
  /// Y distances to each of this subset's neighbors
  Teuchos::ArrayRCP<scalar_t> neighbor_distances_y_;
};

/// \class DICe::Keys4_Strain_Post_Processor
/// \brief A specific instance of post processor that computes fourth order Keys inspired strains
class DICE_LIB_DLL_EXPORT
Keys4_Strain_Post_Processor : public Post_Processor{

public:

  /// Default constructor
  /// \param schema Pointer to the parent schema that is using this post processor
  /// \param params Pointer to the set of parameters for this post processor
  Keys4_Strain_Post_Processor(Schema * schema,
    const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// Virtual destructor
  virtual ~Keys4_Strain_Post_Processor(){}

  /// Set the parameters for this post processor
  virtual void set_params(const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// See base clase docutmentation
  virtual const int_t strain_window_size(){
    return 6;
  }

  /// Collect the neighborhoods of each of the points
  virtual void pre_execution_tasks();

  /// Execute the post processor
  virtual void execute();

  /// See base class documentation
  using Post_Processor::field_names_;

  /// See base class documentation
  using Post_Processor::data_num_points_;

  /// See base class documentation
  using Post_Processor::schema_;

  /// See base class documentation
  using Post_Processor::field_value;

private:
  /// The stride used for the compute vectors
  int_t vec_stride_;
  /// List of all the subsets within the window size
  Teuchos::ArrayRCP<int_t> neighbor_lists_;
  /// Number of neighbors for each subset
  Teuchos::ArrayRCP<int_t> num_neigh_;
  /// X distances to each of this subset's neighbors
  Teuchos::ArrayRCP<scalar_t> neighbor_distances_x_;
  /// Y distances to each of this subset's neighbors
  Teuchos::ArrayRCP<scalar_t> neighbor_distances_y_;
};

}// End DICe Namespace

#endif
