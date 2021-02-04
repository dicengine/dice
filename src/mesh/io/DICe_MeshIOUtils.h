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

#ifndef DICE_MESHIOUTILS_H_
#define DICE_MESHIOUTILS_H_

#include <DICe.h>
#include <DICe_Mesh.h>

#include <vector>

namespace DICe{
namespace mesh{

/// \class Importer_Projector
/// \brief holds all the methods and data for reading in points from an exodus or text file
/// and projecting the data to another set of points if requested
class
DICE_LIB_DLL_EXPORT
Importer_Projector
{
public:
  /// constructor
  /// \param source_file_name same format as the target file, but this is where the data will come from
  /// \param target_file_name either an exodus file, a DICe text output file, or a txt file with two columns of data
  /// the target locations at which the data will be projected to are read from this file.
  /// Once the points are read in, they are fixed for the life of the Importer_Projector
  /// TODO write a update_points() method for this class to relax this constraint
  Importer_Projector(const std::string & source_file_name,
    const std::string & target_file_name);

  /// constructor
  /// \param source_file_name same format as the target file, but this is where the data will come from
  /// \param target_mesh an exodus mesh to project the source data to
  Importer_Projector(const std::string & source_file_name,
    Teuchos::RCP<DICe::mesh::Mesh> target_mesh);

  /// returns the number of target points
  int_t num_target_pts(){
    return (int_t)target_pts_x_.size();
  }

  /// returns the number of source points
  int_t num_source_pts(){
    return (int_t)source_pts_x_.size();
  }

  /// read a vector field from file
  /// \param file_name the file to read
  /// \param field_name the field to gather
  /// \param field_x [out] vector returned with field values x
  /// \param field_y [out] vector returned with field values y
  /// \param step time step requested (must be zero for text input)
  void read_vector_field(const std::string & file_name,
    const std::string & field_name,
    std::vector<work_t> & field_x,
    std::vector<work_t> & field_y,
    const int_t step=0);

  /// read the coordinates from the given file
  /// \param file_name the string name of the file
  /// \param coords_x [out] the output coords x component
  /// \param coords_y [out] the output coords y component
  void read_coordinates(const std::string & file_name,
    std::vector<work_t> & coords_x,
    std::vector<work_t> & coords_y);

  /// import and project (if necessary) a vector field from the source
  /// file
  /// \param file_name the name of the file to import from (must have same number of points as source file)
  /// \param field_name the name of the field to import
  /// \param field_x [out] vector of field x values
  /// \param field_y [out] vector of field y values
  void import_vector_field(const std::string & file_name,
    const std::string & field_name,
    std::vector<work_t> & field_x,
    std::vector<work_t> & field_y,
    const int_t step=0);

  /// return a pointer to the target pt locations in x
  std::vector<work_t> * target_pts_x(){
    return & target_pts_x_;
  }

  /// return a pointer to the target pt locations in y
  std::vector<work_t> * target_pts_y(){
    return & target_pts_y_;
  }

  /// returns true if the field is a valid source field
  /// \param file_name the name of the file to check
  /// \param field_name the name of the requested field
  bool is_valid_vector_source_field(const std::string & file_name,
    const std::string & field_name);

private:
  /// protect the default constructor
  Importer_Projector(const Importer_Projector&);
  /// comparison operator
  Importer_Projector& operator=(const Importer_Projector&);

  /// set up the locations where the data will be projected from
  /// \param source_file_name the string name of the file to use for source locations
  void initialize_source_points(const std::string & source_file_name);

  /// locations to import the data from
  std::vector<work_t> source_pts_x_;
  /// locations to import the data from
  std::vector<work_t> source_pts_y_;
  /// locations to project the imported file to
  std::vector<work_t> target_pts_x_;
  /// locations to project the imported file to
  std::vector<work_t> target_pts_y_;
  /// determines if the target and source points are colinear or not, true if not colinear
  bool projection_required_;
  /// holds the neighbor ids in the source set for each target point
  std::vector<std::vector<int_t> > neighbors_;
  /// holds the neighbor x dist of the source neighbor for each target point
  std::vector<std::vector<work_t> > neighbor_dist_x_;
  /// holds the neighbor y dist of the source neighbor for each target point
  std::vector<std::vector<work_t> > neighbor_dist_y_;
  /// number of neighbors to use
  int_t num_neigh_;

};


} //mesh
} //DICe

#endif /* DICE_MESHIO_H_ */
