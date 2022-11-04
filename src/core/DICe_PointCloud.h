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

#ifndef DICE_POINTCLOUD_H
#define DICE_POINTCLOUD_H

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <nanoflann.hpp>

#include <vector>

using namespace nanoflann;

namespace DICe {

// forward declaration
template <typename T>
class DICE_LIB_DLL_EXPORT
Point_Cloud_2D;

/// a kd-tree index:
typedef KDTreeSingleIndexAdaptor<
  L2_Simple_Adaptor<scalar_t, Point_Cloud_2D<scalar_t> > ,
  Point_Cloud_2D<scalar_t>,
  2 /* dim */
  > kd_tree_2d_t;

template <typename T>
class DICE_LIB_DLL_EXPORT
Point_Cloud_3D;

/// a kd-tree index:
typedef KDTreeSingleIndexAdaptor<
  L2_Simple_Adaptor<scalar_t, Point_Cloud_3D<scalar_t> > ,
  Point_Cloud_3D<scalar_t>,
  3 /* dim */
  > kd_tree_3d_t;

template <typename T>
class DICE_LIB_DLL_EXPORT
Point_Cloud_6D;

/// a kd-tree index:
typedef KDTreeSingleIndexAdaptor<
  L2_Simple_Adaptor<scalar_t, Point_Cloud_6D<scalar_t> > ,
  Point_Cloud_6D<scalar_t>,
  6 /* dim */
  > kd_tree_6d_t;


/// point clouds
template <typename T>
class DICE_LIB_DLL_EXPORT
Point_Cloud_2D{
public:
  /// default constructor with no arguments
  Point_Cloud_2D():N(3),
      tree_requires_update(true){
    IPIV_v.resize(N+1);
    IPIV = &IPIV_v[0];
    LWORK = N*N;
    INFO = 0;
    WORK_v = std::vector<double>(LWORK);
    WORK = &WORK_v[0];
    GWORK_v = std::vector<double>(10*N);
    GWORK = &GWORK_v[0];
    IWORK_v = std::vector<int>(LWORK);
    IWORK = &IWORK_v[0];
    params.sorted = true; // sort by distance in ascending order
  };
  /// constructor with point cloud points passed in
  /// \param x_in x coordinates of the point cloud
  /// \param y_in y coordinates of the point cloud
  Point_Cloud_2D(const std::vector<T> & x_in,
      const std::vector<T> & y_in):Point_Cloud_2D(){
    pts.resize(x_in.size());
    assert(x_in.size()==y_in.size());
    for(size_t i=0;i<x_in.size();++i){
      pts[i].x = x_in[i];
      pts[i].y = y_in[i];
    }
    kd_tree = Teuchos::rcp(new kd_tree_2d_t(2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
    kd_tree->buildIndex();
    tree_requires_update = false;
  }
  /// convenience method for building a point cloud from a map of integer pixel locations
  Point_Cloud_2D(const std::map<std::pair<int_t,int_t>,std::vector<scalar_t> > & points):Point_Cloud_2D(){
    pts.resize(points.size());
    int_t i = 0;
    std::map<std::pair<int_t,int_t>,std::vector<scalar_t> >::const_iterator map_it;
    std::map<std::pair<int_t,int_t>,std::vector<scalar_t> >::const_iterator map_begin = points.begin();
    std::map<std::pair<int_t,int_t>,std::vector<scalar_t> >::const_iterator map_end   = points.end();
    for(map_it=map_begin;map_it!=map_end;++map_it){
      pts[i].x = map_it->first.first;
      pts[i].y = map_it->first.second;
      i++;
    }
    kd_tree = Teuchos::rcp(new kd_tree_2d_t(2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
    kd_tree->buildIndex();
    tree_requires_update = false;
  }
  /// function to add a point to the point cloud
  /// since the points are changed the update flag is set since the kd tree will have to be rebuilt
  void add_point(const T & x, const T & y){
    Point_Cloud_2D<T>::Point p;
    p.x = x;
    p.y = y;
    pts.push_back(p);
    tree_requires_update = true;
  }
  /// method to perform a least squares fit of surrounding points
  /// \param x x coordinate of point of interest
  /// \param y y coordinate of point of interest
  /// \param neigh_size the number of neighbors to use or the neighbor radius. If the neigh_size is an integer value it is assumed to be a knn search
  /// \param data the vector of vectors of values to fit, must be the same dimension as the point cloud points, each column of the vector gets fit
  /// \param result the array holding the fit value for each column
  /// \param force_radius force a radius search even if the input neigh_size is an integer value
  void least_squares_fit(const T & x,
      const T & y,
      const scalar_t & neigh_size,
      const std::vector<std::vector<T>> & data,
      std::vector<T> & result,
      const bool force_radius=false){
    // make sure data is the right size
    TEUCHOS_TEST_FOR_EXCEPTION(data.size()<1,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(data[0].size()!=result.size(),std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(data.size()!=pts.size(),std::runtime_error,"");
    for(size_t i=0;i<result.size();++i)
      result[i] =0.0;

    if(tree_requires_update){
      kd_tree = Teuchos::rcp(new kd_tree_2d_t(2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
      kd_tree->buildIndex();
    }
    T query_pt[2];
    query_pt[0] = x;
    query_pt[1] = y;
    int_t num_neigh = 0;
    std::vector<size_t> ret_index;
    if(std::floor(neigh_size)!=neigh_size||force_radius){ // assume it's a radius search
      std::vector<std::pair<size_t,scalar_t> > ret_matches;
      const scalar_t neigh_rad_2 = neigh_size*neigh_size;
      kd_tree->radiusSearch(&query_pt[0],neigh_rad_2,ret_matches,params);
      num_neigh = ret_matches.size();
      ret_index.resize(num_neigh);
      for(int_t i=0;i<num_neigh;++i)
        ret_index[i] = ret_matches[i].first;
    }else{
      num_neigh = std::floor(neigh_size);
      ret_index.resize(num_neigh);
      std::vector<T> out_dist_sqr(num_neigh);
      kd_tree->knnSearch(&query_pt[0], num_neigh, &ret_index[0], &out_dist_sqr[0]);
    }

    // check if enough neigbors were found
    if(num_neigh<3){
      DEBUG_MSG("least_squares_fit(): not enough neighbors to fit");
      return;
    }

    Teuchos::SerialDenseMatrix<int,double> X_t(N,num_neigh, true);
    Teuchos::SerialDenseMatrix<int,double> X_t_X(N,N,true);
    for(int i=0;i<num_neigh;++i){
      X_t(0,i) = 1.0;
      X_t(1,i) = pts[ret_index[i]].x - query_pt[0];
      X_t(2,i) = pts[ret_index[i]].y - query_pt[1];
    }
    // set up X^T*X
    for(int k=0;k<N;++k){
      for(int m=0;m<N;++m){
        for(int j=0;j<num_neigh;++j){
          X_t_X(k,m) += X_t(k,j)*X_t(m,j);
        }
      }
    }
    lapack.GETRF(X_t_X.numRows(),X_t_X.numCols(),X_t_X.values(),X_t_X.numRows(),IPIV,&INFO);
    lapack.GETRI(X_t_X.numRows(),X_t_X.values(),X_t_X.numRows(),IPIV,WORK,LWORK,&INFO);

    for(size_t f=0;f<data[0].size();++f){
      Teuchos::ArrayRCP<double> neigh_data(num_neigh,0.0);
      for(int j=0;j<num_neigh;++j){
        neigh_data[j] = data[ret_index[j]][f];
      }
      // compute X^T*u
      Teuchos::ArrayRCP<double> X_t_u(N,0.0);
      for(int k=0;k<N;++k)
        for(int j=0;j<num_neigh;++j)
          X_t_u[k] += X_t(k,j)*neigh_data[j];

      // compute the coeffs
      Teuchos::ArrayRCP<double> coeffs(N,0.0);
      for(int k=0;k<N;++k)
        for(int j=0;j<N;++j)
          coeffs[k] += X_t_X(k,j)*X_t_u[j];
      result[f] = coeffs[0];
    }
  }
  /// method to perform a least squares fit of surrounding points
  /// convenience method for a vector of d=1
  /// \param x x coordinate of point of interest
  /// \param y y coordinate of point of interest
  /// \param num_neigh the number of neighbors to use in the fit
  /// \param data the vector of values to fit, must be the same dimension as the point cloud points
  T least_squares_fit(const T & x, const T & y,
      const scalar_t & neigh_size,
      const std::vector<std::vector<T> > & data,
      const bool force_radius = false){
    TEUCHOS_TEST_FOR_EXCEPTION(data.size()<1,std::runtime_error,"");
    TEUCHOS_TEST_FOR_EXCEPTION(data[0].size()!=1,std::runtime_error,"");
    std::vector<T> result(1,0.0);
    least_squares_fit(x,y,neigh_size,data,result,force_radius);
    return result[0];
  }
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

private:
  /// storage variable for the lapack routines
  const int N;
  /// flag to know when the kd tree needs to be rebuilt
  bool tree_requires_update;
  /// storage variable for the lapack routines
  int *IPIV;
  /// storage variable for the lapack routines
  std::vector<int> IPIV_v;
  /// storage variable for the lapack routines
  int LWORK;
  /// storage variable for the lapack routines
  int INFO;
  /// storage variable for the lapack routines
  double *WORK;
  /// storage variable for the lapack routines
  std::vector<double> WORK_v;
  /// storage variable for the lapack routines
  double *GWORK;
  /// storage variable for the lapack routines
  std::vector<double> GWORK_v;
  /// storage variable for the lapack routines
  int *IWORK;
  /// storage variable for the lapack routines
  std::vector<int> IWORK_v;
  // Note, LAPACK does not allow templating on long int or scalar_t...must use int and double
  Teuchos::LAPACK<int,double> lapack;
  /// search params for kd tree
  nanoflann::SearchParams params;
  /// kd tree to use for nearest neighbor search
  Teuchos::RCP<kd_tree_2d_t> kd_tree;
};


/// point clouds
template <typename T>
class DICE_LIB_DLL_EXPORT
Point_Cloud_3D{
public:
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


/// point clouds
template <typename T>
class DICE_LIB_DLL_EXPORT
Point_Cloud_6D{
public:
  /// point struct
  struct Point
  {
    /// data x
    T x;
    /// data y
    T y;
    /// data z
    T z;
    /// data p
    T p;
    /// data q
    T q;
    /// data r
    T r;
  };
  /// constructor with point cloud points passed in
  /// \param in vector of 6 dof vectors to define the coordinates of the point cloud
  Point_Cloud_6D(const std::vector<std::vector<T> > & in){
    params.sorted = true; // sort by distance in ascending order
    pts.resize(in.size());
    assert(in.size()>0);
    assert(in[0].size()==6);
    for(size_t i=0;i<in.size();++i){
      pts[i].x = in[i][0];
      pts[i].y = in[i][1];
      pts[i].z = in[i][2];
      pts[i].p = in[i][3];
      pts[i].q = in[i][4];
      pts[i].r = in[i][5];
    }
    kd_tree = Teuchos::rcp(new kd_tree_6d_t(6 /*dim*/, *this, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
    kd_tree->buildIndex();
  }

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
    const T d3=p1[3]-pts[idx_p2].p;
    const T d4=p1[4]-pts[idx_p2].q;
    const T d5=p1[5]-pts[idx_p2].r;
    return d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5;
  }
  /// Returns the dim'th component of the idx'th point in the class:
  /// Since this is inlined and the "dim" argument is typically an immediate value, the
  ///  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, int dim) const
  {
    if (dim==0) return pts[idx].x;
    else if(dim==1) return pts[idx].y;
    else if(dim==1) return pts[idx].z;
    else if(dim==1) return pts[idx].p;
    else if(dim==1) return pts[idx].q;
    else return pts[idx].r;
  }
  /// Optional bounding-box computation: return false to default to a standard bbox computation loop.
  ///   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
  ///   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

  /// return a pointer to the kd tree
  Teuchos::RCP<kd_tree_6d_t> get_kd_tree()const{return kd_tree;}

  /// search params for kd tree
  nanoflann::SearchParams params;
  /// kd tree to use for nearest neighbor search
  Teuchos::RCP<kd_tree_6d_t> kd_tree;

};

}

#endif
