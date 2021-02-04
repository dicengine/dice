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

#ifndef DICE_SUBSET_H
#define DICE_SUBSET_H

#include <DICe.h>
#include <DICe_Shape.h>
#include <DICe_Image.h>
#include <DICe_LocalShapeFunction.h>

#include <Teuchos_ArrayRCP.hpp>

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

/// \class DICe::Subset
/// \brief Subsets are used to store temporary collections of pixels for comparison between the
/// reference and deformed images. The data that is stored by a subset is a list of x and y
/// corrdinates of each pixel (this allows for arbitrary shape) and containers for pixel
/// intensity values.

class DICE_LIB_DLL_EXPORT
Subset {
public:

  /// constructor that takes arrays of the pixel locations as input
  /// (note this type of subset has not been enabled for obstruction detection,
  /// only conformal subsets with defined shapes have this enabled. See the
  /// constructor below with a subset_def argument)
  /// \param cx centroid x pixel location
  /// \param cy centroid y pixel location
  /// \param x array of x coordinates
  /// \param y array of y coordinates
  Subset(int_t cx,
    int_t cy,
    Teuchos::ArrayRCP<int_t> x,
    Teuchos::ArrayRCP<int_t> y);

  /// constructor that takes a centroid and dims as arguments
  /// (note this type of subset has not been enabled for obstruction detection,
  /// only conformal subsets with defined shapes have this enabled. See the
  /// constructor below with a subset_def argument)
  /// \param cx centroid x pixel location
  /// \param cy centroid y pixel location
  /// \param width width of the centroid (should be an odd number, otherwise the next larger odd number is used)
  /// \param height height of the centroid (should be an odd number, otherwise the next larger odd numer is used)
  Subset(const int_t cx,
    const int_t cy,
    const int_t width,
    const int_t height);

  /// constructor that takes a conformal subset def as the input
  /// \param cx centroid x pixel location
  /// \param cy centroid y pixel location
  /// \param subset_def the definition of the subset areas
  Subset(const int_t cx,
    const int_t cy,
    const Conformal_Area_Def & subset_def);

  /// virtual destructor
  virtual ~Subset(){};

  /// returns the number of pixels in the subset
  int_t num_pixels()const{
    return num_pixels_;
  }

  /// returns the centroid x pixel location
  int_t centroid_x()const{
    return cx_;
  }

  /// returns the centroid y pixel location
  int_t centroid_y()const{
    return cy_;
  }

  /// update the centroid of a subset
  void update_centroid(const int_t cx, const int_t cy);

  /// x coordinate accessor
  /// \param pixel_index the pixel id
  /// note there is no bounds checking on the index
  const int_t& x(const int_t pixel_index)const;

  /// y coordinate accessor
  /// \param pixel_index the pixel id
  /// note there is no bounds checking on the index
  const int_t& y(const int_t pixel_index)const;

  /// gradient x accessor
  /// \param pixel_index the pixel id
  /// note there is no bounds checking
  const work_t& grad_x(const int_t pixel_index)const;

  /// gradient y accessor
  /// \param pixel_index the pixel id
  /// note there is no bounds checking
  const work_t& grad_y(const int_t pixel_index)const;

  /// returns true if the gradients have been computed
  bool has_gradients()const{
    return has_gradients_;
  }

  /// returns a copy of the gradient x values as an array
  Teuchos::ArrayRCP<work_t> grad_x_array()const;

  /// returns a copy of the gradient x values as an array
  Teuchos::ArrayRCP<work_t> grad_y_array()const;

  /// ref intensities accessor
  /// \param pixel_index the pixel id
  work_t & ref_intensities(const int_t pixel_index);

  /// ref intensities accessor
  /// \param pixel_index the pixel id
  work_t & def_intensities(const int_t pixel_index);

  /// initialization method:
  /// \param image the image to get the intensity values from
  /// \param target the initialization mode (put the values in the ref or def intensities)
  /// \param shape_function contains the deformation map (optional)
  /// \param interp interpolation method (optional)
  void initialize(Teuchos::RCP<Image> image,
    const Subset_View_Target target=REF_INTENSITIES,
    Teuchos::RCP<Local_Shape_Function> shape_function=Teuchos::null,
    const Interpolation_Method interp=KEYS_FOURTH);

  /// write the subset intensity values to a tif file
  /// \param file_name the name of the tif file to write
  /// \param use_def_intensities use the deformed intensities rather than the reference
  void write_image(const std::string & file_name,
    const bool use_def_intensities=false);

  /// draw the subset over an image
  /// \param file_name the name of the tif file to write
  /// \param image pointer to the image to use as the background
  /// \param shape_function contains the deformation map (optional)
  void write_subset_on_image(const std::string & file_name,
    Teuchos::RCP<Image> image,
    Teuchos::RCP<Local_Shape_Function> shape_function=Teuchos::null);

  /// returns the max intensity value
  /// \param target either the reference or deformed intensity values
  work_t max(const Subset_View_Target target=REF_INTENSITIES);

  /// returns the min intensity value
  /// \param target either the reference or deformed intensity values
  work_t min(const Subset_View_Target target=REF_INTENSITIES);

  /// returns the mean intensity value
  /// \param target either the reference or deformed intensity values
  work_t mean(const Subset_View_Target target);

  /// returns the mean intensity value
  /// \param target either the reference or deformed intensity values
  /// \param sum [output] returns the reduction value of the intensities minus the mean
  work_t mean(const Subset_View_Target target,
    work_t & sum);

  /// round the values in the subset to the nearest integer number
  void round(const Subset_View_Target target);

  /// returns the ZNSSD gamma correlation value between the reference and deformed subsets
  work_t gamma();

  /// returns the un-normalized difference between the the reference and deformed intensity values
  work_t diff_ref_def() const;

  /// returns the SSSIG value for the reference intensities
  work_t sssig();

  /// reset the is_active bool for each pixel to true;
  void reset_is_active();

  /// returns true if this pixel is active
  bool & is_active(const int_t pixel_index);

  /// returns the number of active pixels in the subset
  int_t num_active_pixels();

  /// returns true if this pixel is deactivated for this particular frame
  bool & is_deactivated_this_step(const int_t pixel_index);

  /// reset the is_deactivated_this_step bool for each pixel to false
  void reset_is_deactivated_this_step();

  /// True if the subset is not square and has geometry information in a Conformal_Area_Def
  bool is_conformal()const{
    return is_conformal_;
  }

  /// Returns a pointer to the subset's Conformal_Area_Def
  const Conformal_Area_Def * conformal_subset_def()const{
    return &conformal_subset_def_;
  }

  /// Return the id of the sub-region of a global image to use
  int_t sub_image_id()const{
    return sub_image_id_;
  }

  /// set the sub_image_id
  void set_sub_image_id(const int_t id){
    sub_image_id_ = id;
  }

  /// \brief Returns an estimate of the noise standard deviation for this subset based on the method
  /// of J. Immerkaer, Fast Noise Variance Estimation, Computer Vision and
  /// Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996
  /// The estimate is computed for a rectangular window that encompases the entire subset if the subset is conformal
  /// \param image the image for which to estimate the noise for this subset
  /// \param shape_function contains the deformation map (optional)
  work_t noise_std_dev(Teuchos::RCP<Image> image,
    Teuchos::RCP<Local_Shape_Function> shape_function);

  /// \brief Returns the std deviation of the image intensity values
  work_t contrast_std_dev();

  /// \brief EXPERIMENTAL Check the deformed position of the pixel to see if it falls inside an obstruction, if so, turn it off
  /// \param shape_function contains the deformation map (optional)
  ///
  /// This method uses the specified deformation map to deform the subset to the current position. It then
  /// checks to see if any of the deformed pixels fall behind an obstruction. These obstructed pixels
  /// are turned off by setting the is_deactivated_this_step_[i] flag to true. These flags are reset on every frame.
  /// When methods like gamma() are called on an objective, these pixels get skipped so they do not contribute to
  /// the correlation or the optimization routine.
  void turn_off_obstructed_pixels(Teuchos::RCP<Local_Shape_Function> shape_function);

  /// \brief EXPERIMENTAL See if any pixels that were obstructed to begin with are now in view, if so use the
  /// newly visible intensity values to reconstruct the subset
  ///
  /// Some pixels in the subset may be hidden in the first frame by an obstruction (or another subset that
  /// sits on top of this one). As the frames progress, these pixels eventually become visible. This method
  /// can be used to evolve the reference subset intensities by taking the intensity value of the pixel
  /// when it becomes visible. The location of the pixel in the current frame is not needed because it
  /// is assumed that the reference and deformed subset intensity arrays are ordered the same. So all that
  /// needs to be checked is if it was deactivated at frame zero, and is now not deactivated this step,
  /// that particular pixel can be used.
  void turn_on_previously_obstructed_pixels();

  /// \brief  EXPERIMENTAL Returns true if the given coordinates fall within an obstructed region for this subset
  /// \param coord_x global x-coordinate
  /// \param coord_y global y-coordinate
  bool is_obstructed_pixel(const work_t & coord_x,
    const work_t & coord_y)const;

  /// \brief EXPERIMENTAL Returns a pointer to the set of pixels currently obstructed by another subset
  std::set<std::pair<int_t,int_t> > * pixels_blocked_by_other_subsets(){
    return & pixels_blocked_by_other_subsets_;
  }

  /// \brief EXPERIMENTAL Return the deformed geometry information for the subset boundary
  std::set<std::pair<int_t,int_t> > deformed_shapes(Teuchos::RCP<Local_Shape_Function> shape_function=Teuchos::null,
    const int_t cx=0,
    const int_t cy=0,
    const work_t & skin_factor=1.0);

private:
  /// number of pixels in the subset
  int_t num_pixels_;
  /// pixel container
  Teuchos::ArrayRCP<work_t> ref_intensities_;
  /// pixel container
  Teuchos::ArrayRCP<work_t> def_intensities_;
  /// container for grad_x
  Teuchos::ArrayRCP<work_t> grad_x_;
  /// container for grad_y
  Teuchos::ArrayRCP<work_t> grad_y_;
  /// pixels can be deactivated by obstructions (persistent)
  Teuchos::ArrayRCP<bool> is_active_;
  /// pixels can be deactivated for this frame only
  Teuchos::ArrayRCP<bool> is_deactivated_this_step_;
  /// initial x position of the pixels in the reference image
  Teuchos::ArrayRCP<int_t> x_;
  /// initial x position of the pixels in the reference image
  Teuchos::ArrayRCP<int_t> y_;
  /// \brief EXPERIMENTAL Holds the obstruction coordinates if they exist.
  /// NOTE: The coordinates are switched for this (i.e. (Y,X)) so that
  /// the loops over y then x will be more efficient
  std::set<std::pair<int_t,int_t> > obstructed_coords_;
  /// \brief EXPERIMENTAL Holds the pixels blocked by other subsets if they exist.
  /// NOTE: The coordinates are switched for this (i.e. (Y,X)) so that
  /// the loops over y then x will be more efficient
  std::set<std::pair<int_t,int_t> > pixels_blocked_by_other_subsets_;
  /// centroid location x
  int_t cx_; // assumed to be the middle of the pixel
  /// centroid location y
  int_t cy_; // assumed to be the middle of the pixel
  /// true if the gradient values are populated
  bool has_gradients_;
  /// Conformal_Area_Def that defines the subset geometry
  Conformal_Area_Def conformal_subset_def_;
  /// The subset is not square
  bool is_conformal_;
  /// if sub regions of the frame are used instead of reading in the whole
  /// sub image, this sub_image_id defines which region to draw the pixel information from
  int_t sub_image_id_;
};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
