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

#ifndef DICE_INITIALIZER_H
#define DICE_INITIALIZER_H

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Subset.h>
#include <DICe_PointCloud.h>
#include <DICe_LocalShapeFunction.h>

#include <Teuchos_RCP.hpp>

#include <nanoflann.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <set>
#include <cassert>


/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

class Schema;

/// Deformation triad to store three parameter values in a set
struct def_triad
{
  /// displacement x
  scalar_t u_;
  /// displacement y
  scalar_t v_;
  /// rotation
  scalar_t t_;
  /// constructor
  /// \param u input value for displacement x
  /// \param v input value for displacement y
  /// \param t input value for rotation
  def_triad(const scalar_t & u,
    const scalar_t & v,
    const scalar_t & t):
  u_(u),
  v_(v),
  t_(t){};
};

/// operator < overload for inserting values in a set
/// \param lhs left hand side def_triad
/// \param rhs right hand side def_triad
inline bool operator<(const def_triad& lhs,
  const def_triad& rhs);

/// operator == overload for inserting values in a set
/// \param lhs left hand side def_triad
/// \param rhs right hand side def_triad
inline bool operator==(const def_triad& lhs,
    const def_triad& rhs);

/// \class DICe::Initializer
/// \brief A generic class that provides an initial guess for the optimization routines
/// in a DICe::Objective.
class DICE_LIB_DLL_EXPORT
Initializer {
public:
  /// base class constructor
  Initializer(Schema * schema):
    schema_(schema)
  {};

  /// virtual destructor
  virtual ~Initializer(){};

  /// Tasks that should be performed before the frame is correlated
  virtual void pre_execution_tasks(){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Base class method should never be called.");
  }

  /// Initialize method, called by the objective function to start the optimization with a good first guess.
  /// \param subset_gid the global id of the subset being initialized
  /// \param shape_function [out] the shape function returned with the initial guess
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Base class method should never be called.");
  };

protected:
  /// pointer to the schema that created this initializer, used for field access
  Schema * schema_;

};


/// \class DICe::Path_Initializer
/// \brief A class that takes a text file as input and produces a unique set of
/// points to test when initializing a subset. The text file gets filtered down to
/// a certain resolution and then duplicate points are removed. This class is useful
/// when the path of a particular subset is pre-defined, for example in a mechanism.
/// This class also provides some basic methods for testing solution values and finding
/// the nearest point on the path.

class DICE_LIB_DLL_EXPORT
Path_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema pointer to the parent schema (with all the field info)
  /// \param subset pointer to a specific subset
  /// \param file_name the name of the file to use for input. The input should be in the following
  /// format: ascii text file with values separated by spaces in three columns. The first column
  /// is the x displacement, the second column is y displacement and the third is rotation. If one
  /// of these columns is not needed, a column of zeros should still be provided. Also, there should
  /// not be any blank lines at the end of the file. TODO create a more robust file reader.
  /// No header should be included in the text file.
  /// \param num_neighbors the k value for the k-closest neighbors to store for each point
  Path_Initializer(Schema * schema,
    Teuchos::RCP<Subset> subset,
    const char * file_name,
    const size_t num_neighbors=6);

  /// virtual destructor
  virtual ~Path_Initializer(){};

  /// accessor for triad values
  const std::set<def_triad> * triads() const{
    return &triads_;
  }

  /// returns the number of triads in the set
  size_t num_triads()const{
    return num_triads_;
  }

  /// return the number of neighbors for each triad
  size_t num_neighbors()const{
    return num_neighbors_;
  }

  /// return the id of the neighbor
  /// \param triad_id the id of the triad to gather a neighbor for
  /// \param neighbor_index the index of the neighbor
  size_t neighbor(const size_t triad_id,
    const size_t neighbor_index)const{
    assert(triad_id>=0&&triad_id<num_triads_);
    assert(neighbor_index>=0&&neighbor_index<num_neighbors_);
    return neighbors_[triad_id*num_neighbors_ + neighbor_index];
  }

  /// return the id of the path triad closest to the given point
  /// \param u displacement in x
  /// \param v displacement in y
  /// \param t rotation
  /// \param id [out] the return id of the closest triad
  /// \param distance_sqr [out] the euclidean distance to this point
  void closest_triad(const scalar_t &u,
    const scalar_t &v,
    const scalar_t &t,
    size_t &id,
    scalar_t & distance_sqr)const;

  /// see base class description
  virtual void pre_execution_tasks(){// do nothing for path_initializer
  };

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);

  /// Initialize the subset in the vicinity of the previous guess
  /// \param def_image the deformed image
  /// \param shape_function [out] the shape function to initialize
  /// \param u previous displacemcent x
  /// \param v previous displacement y
  /// \param t previous theta
  /// in this case only the closest k-neighbors will be searched for the best solution
  scalar_t initial_guess(Teuchos::RCP<Image> def_image,
    Teuchos::RCP<Local_Shape_Function> shape_function,
    const scalar_t & u,
    const scalar_t & v,
    const scalar_t & t);

  /// Initializer where the entire set of path triads will be searched for the best solution
  /// \param def_image pointer to the deformed image
  /// \param shape_function [out] the shape function to initialize
  scalar_t initial_guess(Teuchos::RCP<Image> def_image,
    Teuchos::RCP<Local_Shape_Function> shape_function);

  /// write the filtered set of points out to an output file
  void write_to_text_file(const std::string & file_name)const;

private:
  /// pointer to the subset being initialized
  Teuchos::RCP<Subset> subset_;
  /// unique triads of deformation params: u, v, and t
  std::set<def_triad> triads_;
  /// number of unique triads
  size_t num_triads_;
  /// k-closest neighbors for each point in the set of triads
  size_t num_neighbors_;
  /// a strided array of neighbor ids
  std::vector<size_t> neighbors_;
  /// pointer to the kd-tree used for searching
  Teuchos::RCP<kd_tree_3d_t> kd_tree_;
  /// pointer to the point cloud used for the neighbor searching
  Teuchos::RCP<Point_Cloud_3D<scalar_t> > point_cloud_;
};

/// \class DICe::Phase_Correlation_Initializer
/// \brief A class that computes the phase correlation of the
/// whole image to get the initial values of displacement in x and y.
/// This initializer is good for cases where the objects being tracked are
/// on a larger object that is translating through the frame
class DICE_LIB_DLL_EXPORT
Phase_Correlation_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema the parent schema
  Phase_Correlation_Initializer(Schema * schema);

  /// virtual destructor
  virtual ~Phase_Correlation_Initializer(){};

  /// see base class description
  virtual void pre_execution_tasks();

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);

protected:
  /// phase correlation displacement x estimate from previous frame
  scalar_t phase_cor_u_x_;
  /// phase correlation displacement y estimate from previous frame
  scalar_t phase_cor_u_y_;
};

/// \class DICe::Search_Initializer
/// \brief A class that searches a nearby neighborhood for the subset
class DICE_LIB_DLL_EXPORT
Search_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema the parent schema
  /// \param subset pointer to a subset
  /// \param step_size_u the search step size in u (negative 1 means don't search in this dim)
  /// \param search_dim_u the extents of the search in u
  /// \param step_size_v the search step size in v (negative 1 means don't search in this dim)
  /// \param search_dim_v the extents of the search in v
  /// \param step_size_theta the angle step size (negative 1 means don't search in this dim)
  /// \param search_dim_theta the extents of the search in angle
  Search_Initializer(Schema * schema,
    Teuchos::RCP<Subset> subset,
    const scalar_t & step_size_u,
    const scalar_t & search_dim_u,
    const scalar_t & step_size_v,
    const scalar_t & search_dim_v,
    const scalar_t & step_size_theta,
    const scalar_t & search_dim_theta);

  /// virtual destructor
  virtual ~Search_Initializer(){};

  /// see base class description
  virtual void pre_execution_tasks(){};

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);

protected:
  /// pointer to a specific subset
  Teuchos::RCP<Subset> subset_;
  /// search step size in x and y
  scalar_t step_size_u_;
  scalar_t step_size_v_;
  /// extent of search in x and y (added and substracted from input to get extents)
  scalar_t search_dim_u_;
  scalar_t search_dim_v_;
  /// search step size in theta
  scalar_t step_size_theta_;
  /// extent of search in theta
  scalar_t search_dim_theta_;
};


/// \class DICe::Field_Value_Initializer
/// \brief an initializer that grabs values from the field values
/// in the schema as the first guess, for example, the last frame's solution
/// or a neighbor's values
class DICE_LIB_DLL_EXPORT
Field_Value_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema the parent schema
  Field_Value_Initializer(Schema * schema):
  Initializer(schema){};

  /// virtual destructor
  virtual ~Field_Value_Initializer(){};

  /// see base class description
  virtual void pre_execution_tasks(){};

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);
};

/// \class DICe::Feature_Matching_Initializer
/// \brief an initializer that uses nearby feature matching to initialize the solution
class DICE_LIB_DLL_EXPORT
Feature_Matching_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema the parent schema
  Feature_Matching_Initializer(Schema * schema,
    const int_t threshold_block_size=-1);

  /// virtual destructor
  virtual ~Feature_Matching_Initializer(){};

  /// see base class description
  virtual void pre_execution_tasks();

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);

protected:
  /// pointer to the kd-tree used for searching
  Teuchos::RCP<kd_tree_2d_t> kd_tree_;
  /// pointer to the point cloud used for the neighbor searching
  Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud_;
  /// storage for displacements of feautures
  std::vector<scalar_t> u_;
  /// storage for displacements of features
  std::vector<scalar_t> v_;
  /// block size to use for thresholding
  int_t threshold_block_size_;
};

/// \class DICe::Satellite_Geometry_Initializer
/// \brief an initializer that uses camera angles to initialize the solution
class DICE_LIB_DLL_EXPORT
Satellite_Geometry_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema the parent schema
  Satellite_Geometry_Initializer(Schema * schema);

  /// virtual destructor
  virtual ~Satellite_Geometry_Initializer(){};

  /// see base class description
  virtual void pre_execution_tasks(){};

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);

};



/// \class DICe::Image_Registration_Initializer
/// \brief an initializer that uses an ECC transform to initialize the fields
class DICE_LIB_DLL_EXPORT
Image_Registration_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema the parent schema
  Image_Registration_Initializer(Schema * schema);

  /// virtual destructor
  virtual ~Image_Registration_Initializer(){};

  /// see base class description
  virtual void pre_execution_tasks();

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);

protected:
  /// matrix to hold the tranform values
  cv::Mat ecc_transform_;
  /// storage for the rotation angle which should be the same for all points
  scalar_t theta_;
};


/// \class DICe::Zero_Value_Initializer
/// \brief an initializer that places zeros in all values
class DICE_LIB_DLL_EXPORT
Zero_Value_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema the parent schema
  Zero_Value_Initializer(Schema * schema):
  Initializer(schema){};

  /// virtual destructor
  virtual ~Zero_Value_Initializer(){};

  /// see base class description
  virtual void pre_execution_tasks(){};

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);
};

/// \class DICe::Optical_Flow_Initializer
/// \brief an initializer that uses optical flow to predict the next location
/// of the subset (Lucas-Kanade algorithm). This initializer is only intended for
/// the TRACKING_ROUTINE correlation routine
///
/// Initially, two pixels are selected as the locations for doing optical flow based
/// the gradient values for each pixel. If one of the pixels become obstructed the locations
/// need to be reset. The word reference in this class is used to denote the first frame
/// after the positions have been reset. The displacements and rotations are cumulative
/// starting with the reference frame. If the positions need to be reset due to
/// obstructions, the optical flow routine restarts from the new location, which is now
/// the reference frame.
class DICE_LIB_DLL_EXPORT
Optical_Flow_Initializer : public Initializer{
public:

  /// constructor
  /// \param schema the parent schema
  /// \param subset pointer to the subset for this initializer
  Optical_Flow_Initializer(Schema * schema,
    Teuchos::RCP<Subset> subset);

  /// virtual destructor
  virtual ~Optical_Flow_Initializer(){};

  /// see base class description
  virtual void pre_execution_tasks(){};

  /// see base class description
  virtual Status_Flag initial_guess(const int_t subset_gid,
    Teuchos::RCP<Local_Shape_Function> shape_function);

  /// returns the id of the neighbor pixel
  /// \param pixel_id the id of the pixel to gather a neighbor for
  /// \param neighbor_index the index of the neighbor
  size_t neighbor(const int_t pixel_id,
    const int_t neighbor_index)const{
    assert(pixel_id>=0&&pixel_id<subset_->num_pixels());
    assert(neighbor_index>=0&&neighbor_index<(int_t)num_neighbors_);
    return neighbors_[pixel_id*num_neighbors_ + neighbor_index];
  }

  /// determine the best locations in the subset to do optical flow based on the gradients
  /// \param subset_gid the global id of the subset using this initializer
  Status_Flag set_locations(const int_t subset_gid);

  /// returns the pixel id of the best location for optical flow
  /// \param best_grad [out] the gradient metric at the best location for an optical flow point
  /// \param existing_points the set of global ids for points that are already OF points
  /// \param def_x array of the deformed x coordinates for the pixels
  /// \param def_y array of the deformed y coordinates for the pixels
  /// \param gx the array of x gradient values from the deformed location
  /// \param gy the array of y gradient values from the deformed location
  /// \param subset_pixels the current pixels occupied by the subset in the deformed configuration
  /// \param existing_points vector of the existing optical flow points
  /// \param allow_close_points enable the point locations to be near each other (usually leads to more error)
  int_t best_optical_flow_point(scalar_t & best_grad,
    Teuchos::ArrayRCP<int_t> & def_x,
    Teuchos::ArrayRCP<int_t> & def_y,
    Teuchos::ArrayRCP<scalar_t> & gx,
    Teuchos::ArrayRCP<scalar_t> & gy,
    std::set<std::pair<int_t,int_t> > & subset_pixels,
    Teuchos::RCP<std::vector<int_t> > existing_points = Teuchos::null,
    const bool allow_close_points = false);

  /// returns true if the point is deactivated or neighbors a deactivated pixel
  /// \param pixel_id the index of the pixel to test
  bool is_near_deactivated(const int_t pixel_id);

protected:
  /// pointer to the subset being initialized
  Teuchos::RCP<Subset> subset_;
  /// k-closest neighbors for each point in the set of triads
  size_t num_neighbors_;
  /// size of the window
  int_t window_size_;
  /// half the size of the window
  int_t half_window_size_;
  /// a strided array of neighbor ids
  std::vector<size_t> neighbors_;
  /// pointer to the kd-tree used for searching
  Teuchos::RCP<kd_tree_2d_t> kd_tree_;
  /// pointer to the point cloud used for the neighbor searching
  Teuchos::RCP<Point_Cloud_2D<scalar_t> > point_cloud_;
  /// Gauss filter coefficients
  scalar_t window_coeffs_[13][13]; // the optical flow window is 13 pixels wide
  /// x coord of reference position for optical flow 1
  int_t ref_pt1_x_;
  /// y coord of reference position for optical flow 1
  int_t ref_pt1_y_;
  /// x coord of reference position for optical flow 2
  int_t ref_pt2_x_;
  /// y coord of reference position for optical flow 2
  int_t ref_pt2_y_;
  /// x coord of current position for optical flow 1
  scalar_t current_pt1_x_;
  /// y coord of current position for optical flow 1
  scalar_t current_pt1_y_;
  /// x coord of current position for optical flow 2
  scalar_t current_pt2_x_;
  /// y coord of current position for optical flow 2
  scalar_t current_pt2_y_;
  /// flag to reset the optical flow positions
  bool reset_locations_;
  /// x comp of vector from optical flow position 1 to the centroid
  scalar_t delta_1c_x_;
  /// y comp of vector from optical flow position 1 to the centroid
  scalar_t delta_1c_y_;
  /// x comp of vector from optical flow position 1 to position 2
  scalar_t delta_12_x_;
  /// y comp vector from optical flow position 1 to position 2
  scalar_t delta_12_y_;
  /// magnitude of the vector between positions 1 and 2 in the reference frame
  scalar_t mag_ref_;
  /// reference centroid x
  scalar_t ref_cx_;
  /// reference centroid y
  scalar_t ref_cy_;
  /// initial displacement for this sequence
  scalar_t initial_u_;
  /// initial displacement for this sequence
  scalar_t initial_v_;
  /// initial rotation for this sequence
  scalar_t initial_t_;
  /// pixel ids of the optical flow points
  int_t ids_[2];
};

//
//  Initialization utilities
//

/// \class DICe::Motion_Test_Utility
/// \brief tests to see if there has been any motion since the last frame
/// if not, this frame can be skipped.
class DICE_LIB_DLL_EXPORT
Motion_Test_Utility{
public:
  /// constructor
  /// \param schema pointer to the schema that will be calling the motion test utility
  /// \param tol determines the threshold for the image diff to register motion
  Motion_Test_Utility(Schema * schema,
    const scalar_t & tol);

  /// virtual destructor
  ~Motion_Test_Utility(){};

  /// reset the result flag (should be done at the beginning of each frame)
  void reset(){
    motion_state_ = MOTION_NOT_SET;
  };

  /// Returns true if motion is detected
  /// \param sub_image_id the id of the window image around the subset
  bool motion_detected(const int_t sub_image_id);

private:
  /// pointer to the schema that created this initializer, used for field access
  Schema * schema_;
  /// image diff tolerance (above this means motion is occurring)
  scalar_t tol_;
  /// keep a copy of the result incase another call is
  /// made for this initializer by another subset
  Motion_State motion_state_;
};


}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
