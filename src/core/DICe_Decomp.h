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

#ifndef DICE_DECOMP_H
#define DICE_DECOMP_H

#include <DICe.h>
#include <DICe_Image.h>
#ifdef DICE_TPETRA
  #include "DICe_MultiFieldTpetra.h"
#else
  #include "DICe_MultiFieldEpetra.h"
#endif


#include <Teuchos_RCP.hpp>
#include <nanoflann.hpp>

using namespace nanoflann;

/// point clouds
template <typename T>
struct Decomp_Point_Cloud
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
  L2_Simple_Adaptor<DICe::scalar_t, Decomp_Point_Cloud<DICe::scalar_t> > ,
  Decomp_Point_Cloud<DICe::scalar_t>,
  2 /* dim */
  > decomp_kd_tree_t;

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

/// \class DICe::Decomp
/// \brief decomposes the images and the mesh points for parallel execution.
class DICE_LIB_DLL_EXPORT
Decomp {
public:
  /// constructor
  /// \param input_params the input parameters
  /// \param correlation_params the correlation parameters
  Decomp(const Teuchos::RCP<Teuchos::ParameterList> & input_params,
    const Teuchos::RCP<Teuchos::ParameterList> & correlation_params);

  /// constructor that splits up an already populated set of correlation points
  /// \param subset_centroids_x x coordinates of all global points
  /// \param subset_centroids_y y coordinates of all global points prior to decomposition
  /// \param neighbor_ids the neighbors in global ids to use for initialization of the solution if neighbor values are used
  /// \param obstructing_subset_ids obstructions listed for each subset if necessary
  /// \param correlation_params pointer to the parameters used for the correlation
  Decomp(const Teuchos::ArrayRCP<scalar_t> subset_centroids_x,
    const Teuchos::ArrayRCP<scalar_t> subset_centroids_y,
    const Teuchos::RCP<std::vector<int_t> > neighbor_ids,
    const Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids,
    const Teuchos::RCP<Teuchos::ParameterList> & correlation_params);

  /// destructor
  ~Decomp(){};

  /// returns the global number of subsets
  int_t num_global_subsets()const{
    return num_global_subsets_;
  }

  /// returns a copy of the execution ordering
  std::vector<int_t> this_proc_gid_order()const{
    return this_proc_gid_order_;
  }

  /// returns the one to one decomp map
  Teuchos::RCP<MultiField_Map> id_decomp_map(){
    return id_decomp_map_;
  }

  /// return the non one-to-one decomposition map (some ids are ghosted across processors)
  Teuchos::RCP<MultiField_Map> id_decomp_overlap_map(){
    return id_decomp_overlap_map_;
  }

  /// returns a pointer to the trimmed set of overlap x coordinates
  Teuchos::ArrayRCP<scalar_t> overlap_coords_x()const{
    return overlap_coords_x_;
  }

  /// returns a pointer to the trimmed set of overlap y coordinates
  Teuchos::ArrayRCP<scalar_t> overlap_coords_y()const{
    return overlap_coords_y_;
  }

  /// returns a pointer to the neighbor ids vector
  Teuchos::RCP<std::vector<int_t> > neighbor_ids()const{
    return neighbor_ids_;
  }

  /// returns the image width
  int_t image_width()const{
    return image_width_;
  }

  /// returns the image width
  int_t image_height()const{
    return image_height_;
  }

private:
  /// initialize all the maps and split up the subsets across processors preserving obstructions and seeds
  /// \param subset_centroids_x x coordinates of all global points
  /// \param subset_centroids_y y coordinates of all global points prior to decomposition
  /// \param neighbor_ids the neighbors in global ids to use for initialization of the solution if neighbor values are used
  /// \param obstructing_subset_ids obstructions listed for each subset if necessary
  /// \param correlation_params pointer to the parameters used for the correlation
  void initialize(const Teuchos::ArrayRCP<scalar_t> subset_centroids_x,
    const Teuchos::ArrayRCP<scalar_t> subset_centroids_y,
    const Teuchos::RCP<std::vector<int_t> > neighbor_ids,
    const Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids,
    const Teuchos::RCP<Teuchos::ParameterList> & correlation_params);

  /// populates the coordinates of all global points
  /// \param subset_centroids_x vector that gets populated with all global subset x coordinates
  /// \param subset_centroids_y vector that gets populated with all global subset y coordinates
  /// \param neighbor_ids vector with the neighbor ids for each subset
  /// \param image_file_name name of the image to read for computing SSSIG if necessary
  /// \param input_params input parameters from xml file
  /// \param correlation_params correlation parameters from xml file
  void populate_global_coordinate_vector(Teuchos::ArrayRCP<scalar_t> & subset_centroids_x,
    Teuchos::ArrayRCP<scalar_t> & subset_centroids_y,
    Teuchos::RCP<std::vector<int_t> > & neighbor_ids,
    Teuchos::RCP<std::map<int_t,std::vector<int_t> > > & obstructing_subset_ids,
    const std::string & image_file_name,
    const Teuchos::RCP<Teuchos::ParameterList> & input_params,
    const Teuchos::RCP<Teuchos::ParameterList> & correlation_params);

  /// redo the ordering and decomposition of points if there are obstructions involved
  /// \param obstructing_subset_ids map giving the obstructions for each subset
  void create_obstruction_dist_map( Teuchos::RCP<std::map<int_t,std::vector<int_t> > > & obstructing_subset_ids);

  /// redo the ordering and decomposition to include seeds
  /// \param neighbor_ids vector of neighbor ids for each global point
  /// \param obstructing_subset_ids used to make sure both obstructions and seeds aren's used together
  /// \param correlation_params correlation parameters from xml file
  void create_seed_dist_map(Teuchos::RCP<std::vector<int_t> > & neighbor_ids,
    Teuchos::RCP<std::map<int_t,std::vector<int_t> > > & obstructing_subset_ids,
    const Teuchos::RCP<Teuchos::ParameterList> & correlation_params);

  /// total number of points in the discretization
  int_t num_global_subsets_;
  /// image width of reference image (used for filtering points out of bounds)
  int_t image_width_;
  /// image height of reference image (used for filtering points out of bounds)
  int_t image_height_;
  /// global mpi communicator
  Teuchos::RCP<MultiField_Comm> comm_;
  /// one-to-one decomposition map (all ids are only onwned by one processor)
  Teuchos::RCP<MultiField_Map> id_decomp_map_;
  /// non one-to-one decomposition map (some ids are ghosted across processors)
  Teuchos::RCP<MultiField_Map> id_decomp_overlap_map_;
  /// vector to keep track of the execution order for this processor
  std::vector<int_t> this_proc_gid_order_;
  /// all the local and ghosted point x coordinates for this processor
  Teuchos::ArrayRCP<scalar_t> overlap_coords_x_;
  /// all the local and ghosted point y coordinates for this processor
  Teuchos::ArrayRCP<scalar_t> overlap_coords_y_;
  /// trimmed list of neighbor ids
  Teuchos::RCP<std::vector<int_t> > neighbor_ids_;
};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
