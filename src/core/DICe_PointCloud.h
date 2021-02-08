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

#ifndef DICE_POINTCLOUD_H
#define DICE_POINTCLOUD_H

#include <nanoflann.hpp>

#include <vector>

using namespace nanoflann;

/// point clouds
template <typename T>
struct Point_Cloud_2D
{
  /// point struct
  struct Point
  {
    /// data x
    T x;
    /// data y
    T y;
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
    return d0*d0+d1*d1;
  }
  /// Returns the dim'th component of the idx'th point in the class:
  /// Since this is inlined and the "dim" argument is typically an immediate value, the
  ///  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, int dim) const
  {
    if (dim==0) return pts[idx].x;
    else return pts[idx].y;
  }
  /// Optional bounding-box computation: return false to default to a standard bbox computation loop.
  ///   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
  ///   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

/// a kd-tree index:
typedef KDTreeSingleIndexAdaptor<
  L2_Simple_Adaptor<DICe::scalar_t, Point_Cloud_2D<DICe::scalar_t> > ,
  Point_Cloud_2D<DICe::scalar_t>,
  2 /* dim */
  > kd_tree_2d_t;

/// point clouds
template <typename T>
struct Point_Cloud_3D
{
  /// point struct
  struct Point
  {
    /// data x
    T x;
    /// data y
    T y;
    /// data z
    T z;
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
  L2_Simple_Adaptor<DICe::scalar_t, Point_Cloud_3D<DICe::scalar_t> > ,
  Point_Cloud_3D<DICe::scalar_t>,
  3 /* dim */
  > kd_tree_3d_t;

#endif
