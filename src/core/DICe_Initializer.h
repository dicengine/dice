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

#ifndef DICE_INITIALIZER_H
#define DICE_INITIALIZER_H

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Subset.h>

#include <Teuchos_RCP.hpp>

#include <nanoflann.hpp>

#include <set>
#include <cassert>

using namespace nanoflann;

/// point clouds
template <typename T>
struct Point_Cloud
{
  /// point struct
  struct Point
  {
    T  x,y,z;
  };
  /// vector of points
  std::vector<Point>  pts;
  /// Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }
  /// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
  inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t /*size*/) const
  {
    const T d0=p1[0]-pts[idx_p2].x;
    const T d1=p1[1]-pts[idx_p2].y;
    const T d2=p1[2]-pts[idx_p2].z;
    return d0*d0+d1*d1+d2*d2;
  }
  /// Returns the dim'th component of the idx'th point in the class:
  /// Since this is inlined and the "dim" argument is typically an immediate value, the
  ///  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, int dim) const
  {
    if (dim==0) return pts[idx].x;
    else if(dim==1) return pts[idx].y;
    else return pts[idx].z;
  }
  /// Optional bounding-box computation: return false to default to a standard bbox computation loop.
  ///   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
  ///   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};


/// a kd-tree index:
typedef KDTreeSingleIndexAdaptor<
  L2_Simple_Adaptor<DICe::scalar_t, Point_Cloud<DICe::scalar_t> > ,
  Point_Cloud<DICe::scalar_t>,
  3 /* dim */
  > my_kd_tree_t;

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

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
  /// \param def_image pointer to the deformed image being correlated
  /// \param subset pointer to the subset being initialized
  Initializer(Teuchos::RCP<Image> def_image,
    Teuchos::RCP<Subset> subset):
    def_image_(def_image),
    subset_(subset){
  }

  /// virtual destructor
  virtual ~Initializer(){};

  /// Initialize method called by the objective function to init the optimization with a good first guess.
  /// The return value is a measure of how good the initial guess is
  /// \param deformaion [out] the deformation vector returned with the initial guess
  /// \param u a seed for the x-displacement
  /// \param v a seed for the y-displacement
  /// \param t a seed for the rotation
  virtual scalar_t initial_guess(Teuchos::RCP<std::vector<scalar_t> > deformation,
    const scalar_t & u,
    const scalar_t & v,
    const scalar_t & t){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Base class method should never be called.");
  };

  /// Initialize method called by the objective function to init the optimization with a good first guess.
  /// In this case, no seeds are provided for the initial guess
  /// The return value is a measure of how good the initial guess is
  /// \param deformaion [out] the deformation vector returned with the initial guess
  virtual scalar_t initial_guess(Teuchos::RCP<std::vector<scalar_t> > deformation){
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Base class method should never be called.");
  };

protected:
  /// pointer to the deformed image
  Teuchos::RCP<Image> def_image_;
  /// pointer to the subset being initialized
  Teuchos::RCP<Subset> subset_;
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
  /// \param def_image pointer to the deformed image being correlated
  /// \param subset pointer to the subset being initialized
  /// \param file_name the name of the file to use for input. The input should be in the following
  /// format: ascii text file with values separated by spaces in three columns. The first column
  /// is the x displacement, the second column is y displacement and the third is rotation. If one
  /// of these columns is not needed, a column of zeros should still be provided. Also, there should
  /// not be any blank lines at the end of the file. TODO create a more robust file reader.
  /// No header should be included in the text file.
  /// \param num_neighbors the k value for the k-closest neighbors to store for each point
  Path_Initializer(Teuchos::RCP<Image> def_image,
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
  /// \param distance [out] the euclidean distance to this point
  void closest_triad(const scalar_t &u,
    const scalar_t &v,
    const scalar_t &t,
    size_t id,
    scalar_t & distance_sqr)const;

  /// see base class description
  /// in this case only the closest k-neighbors will be searched for the best solution
  virtual scalar_t initial_guess(Teuchos::RCP<std::vector<scalar_t> > deformation,
    const scalar_t & u,
    const scalar_t & v,
    const scalar_t & t);

  /// see base class description
  /// in this case, the entire set of path triads will be searched for the best solution
  virtual scalar_t initial_guess(Teuchos::RCP<std::vector<scalar_t> > deformation);

private:
  /// unique triads of deformation params: u, v, and t
  std::set<def_triad> triads_;
  /// number of unique triads
  size_t num_triads_;
  /// k-closest neighbors for each point in the set of triads
  size_t num_neighbors_;
  /// a strided array of neighbor ids
  std::vector<size_t> neighbors_;
  /// pointer to the kd-tree used for searching
  Teuchos::RCP<my_kd_tree_t> kd_tree_;
  /// pointer to the point cloud used for the neighbor searching
  Teuchos::RCP<Point_Cloud<scalar_t> > point_cloud_;

};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
