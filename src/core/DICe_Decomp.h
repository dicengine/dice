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

#ifndef DICE_DECOMP_H
#define DICE_DECOMP_H

#include <DICe.h>
#include <DICe_Image.h>
#include <DICe_Parser.h>
#include <DICe_PointCloud.h>
#ifdef DICE_TPETRA
  #include "DICe_MultiFieldTpetra.h"
#else
  #include "DICe_MultiFieldEpetra.h"
#endif

#include <Teuchos_RCP.hpp>

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
  /// \param neighbor_ids neighbor id to use for initialization if USE_NEIGHBOR_VALUES is selected
  /// \param obstructing_subset_ids obstructions listed for each subset if necessary
  /// \param correlation_params pointer to the parameters used for the correlation
  Decomp(Teuchos::ArrayRCP<scalar_t> subset_centroids_x,
    Teuchos::ArrayRCP<scalar_t> subset_centroids_y,
    Teuchos::RCP<std::vector<int_t> > neighbor_ids,
    Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids,
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

  /// returns a pointer to the subset info
  Teuchos::RCP<DICe::Subset_File_Info> subset_info()const{
    return subset_info_;
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
    Teuchos::RCP<std::vector<int_t> > & neighbor_ids,
    Teuchos::RCP<std::map<int_t,std::vector<int_t> > > obstructing_subset_ids,
    const Teuchos::RCP<Teuchos::ParameterList> & correlation_params);

  /// populate the coordinate vectors
  /// note: all other procs besides proc 0 get an empty vector for coords_x and coords_y
  /// \param image_file_name name of the image to read for computing SSSIG if necessary
  /// \param input_params input parameters from xml file
  /// \param correlation_params correlation parameters from xml file
  /// \param subset_centroids_x vector that gets populated with all global subset x coordinates
  /// \param subset_centroids_y vector that gets populated with all global subset y coordinates
  /// \param neighbor_ids list of neighbors to use if USE_NEIGHBOR_VALUES is the initialization type
  /// \param obstructing_subset_ids list of obstructions
  void populate_coordinate_vectors(const std::string & image_file_name,
    const Teuchos::RCP<Teuchos::ParameterList> & input_params,
    const Teuchos::RCP<Teuchos::ParameterList> & correlation_params,
    Teuchos::ArrayRCP<scalar_t> & subset_centroids_x,
    Teuchos::ArrayRCP<scalar_t> & subset_centroids_y,
    Teuchos::RCP<std::vector<int_t> > & neighbor_ids,
    Teuchos::RCP<std::map<int_t,std::vector<int_t> > > & obstructing_subset_ids);

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
  /// info about ROI's, etc
  Teuchos::RCP<DICe::Subset_File_Info> subset_info_;
};

// free functions to help with creating grids:
/// \brief Creates a regular square grid of correlation points
/// \param correlation_points Vector of global point coordinates
/// \param neighbor_ids list the neighbor to use for initialization if done with USE_NEIGHBOR_VALUES
/// \param params Used to determine the step size (spacing of points)
/// \param img_w width of the image
/// \param img_h height of the image
/// \param subset_file_info Optional information from the subset file (ROIs, etc.)
/// \param pointer to an image to use for checking SSSIG
/// \param grad_threshold subsets with a gradiend SSSIG lower than this will be removed
DICE_LIB_DLL_EXPORT
void create_regular_grid_of_correlation_points(std::vector<scalar_t> & correlation_points,
  std::vector<int_t> & neighbor_ids,
  Teuchos::RCP<Teuchos::ParameterList> params,
  const int_t img_w,
  const int_t img_h,
  Teuchos::RCP<DICe::Subset_File_Info> subset_file_info=Teuchos::null,
  Teuchos::RCP<DICe::Image> image=Teuchos::null,
  const scalar_t & grad_threshold=0.0);

/// \brief Test to see that the point falls with the boundary of a conformal def and not in the excluded area
/// \param x_coord X coordinate of the point in question
/// \param y_coord Y coordinate of the point in question
/// \param subset_size Size of the subset
/// \param img_w width of the image
/// \param img_h height of the image
/// \param coords Set of valid coordinates in the area
/// \param excluded_coords Set of coordinates that should be excluded
/// \param pointer to an image to use for checking SSSIG
/// \param grad_threshold the SSSIG threshold to eliminate a subset without enough gradients to correlate
DICE_LIB_DLL_EXPORT
bool valid_correlation_point(const int_t x_coord,
  const int_t y_coord,
  const int_t subset_size,
  const int_t img_w,
  const int_t img_h,
  std::set<std::pair<int_t,int_t> > & coords,
  std::set<std::pair<int_t,int_t> > & excluded_coords,
  Teuchos::RCP<DICe::Image> image=Teuchos::null,
  const scalar_t & grad_threshold=0.0);


}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
