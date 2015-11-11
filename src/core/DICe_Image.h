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
#ifndef DICE_IMAGE_H
#define DICE_IMAGE_H

#include <DICe.h>
#if DICE_KOKKOS
  #include <DICe_Kokkos.h>
#endif
#include <Teuchos_ParameterList.hpp>

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

/// forward declaration of the conformal_area_def
class Conformal_Area_Def;


/// \class DICe::Image
/// A container class to hold the pixel intensity information and provide some basic methods
/// Note: the coordinates are from the top left corner (positive right for x and positive down for y)
/// intensity access is always in local coordinates, for example if only a portion of an image is read
/// into the intensity values, accessing the first value in the array is via the indicies (0,0) even if
/// the first pixel is not in the upper left corner of the global image from which the poriton was taken

class DICE_LIB_DLL_EXPORT
Image {
public:
  //
  // tiff image constructors
  //

  /// constructor that reads in a whole tiff file
  /// \param file_name the name of the tiff file
  /// \param params image parameters
  Image(const char * file_name,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// constructor that stores only a portion of a tiff file given by the offset and dims
  /// \param file_name the name of the tiff file
  /// \param offset_x upper left corner x-coordinate
  /// \param offset_y upper left corner y-coorindate
  /// \param width x-dim of the image (offset_x + width must be < the global image width)
  /// \param height y-dim of the image (offset_y + height must be < the global image height)
  /// \param params image parameters
  Image(const char * file_name,
    const int_t offset_x,
    const int_t offset_y,
    const int_t width,
    const int_t height,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  //
  // pre allocated array image
  //

  /// constrtuctor that takes an array as input
  /// note: assumes the input array is always stored LayoutRight or "row major"
  /// \param intensities pre-allocated array of intensity values
  /// \param width the width of the image
  /// \param height the height of the image
  /// \param params image parameters
  Image(intensity_t * intensities,
    const int_t width,
    const int_t height,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  //
  // Teuchos::ArrayRCP image
  //

  /// constructor that takes a Teuchos::ArrayRCP as input
  /// note: assumes the input array is always stored LayoutRight or "row major"
  /// \param width image width
  /// \param height image height
  /// \param intensities image intensity values
  /// \param params optional image parameters
  Image(const int_t width,
    const int_t height,
    Teuchos::ArrayRCP<intensity_t> intensities,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  //
  // Empty (zero) image
  //

  /// constructor that creates a zero image
  /// \param width the width of the image
  /// \param height the height of the image
  /// no params allowed since the intensity values are all zeros so gradients
  /// or filters would not make sense
  Image(const int_t width,
    const int_t height);

  //
  // Sub portion of another image constructor (deep copy constructor for default args)
  //

  /// constructor that takes another image and dims of a sub portion
  /// note: no params arg because the parent image's are copied
  /// \param img the image to copy
  /// \param offset_x the upper left corner x-coord in image coordinates
  /// \param offset_y the upper left corner y-coord in image coordinates
  /// \param width the width of the sub image
  /// \param height the height of the sub image
  Image(Teuchos::RCP<Image> img,
    const int_t offset_x = 0,
    const int_t offset_y = 0,
    const int_t width = -1,
    const int_t height = -1);

  /// perform initialization of an image from an array
  /// \param intensities the array of intensity values
  void initialize_array_image(intensity_t * intensities);

  /// default constructor tasks
  void default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// virtual destructor
  virtual ~Image(){};

  /// write the image to tiff file
  /// \param file_name the name of the file to write to
  void write_tiff(const std::string & file_name);

  /// write the image to .rawi format (Raw Intensity)
  /// rather than tiff which will truncate the intensity values to an 8-bit integer value
  /// the rawi format saves the full intesity_t precision value to file
  void write_rawi(const std::string & file_name);

  /// returns the width of the image
  int_t width()const{
    return width_;
  }

  /// return the height of the image
  int_t height()const{
    return height_;
  }

  /// returns the number of pixels in the image
  int_t num_pixels()const{
    return width_*height_;
  }

  /// returns the offset x coordinate
  int_t offset_x()const{
    return offset_x_;
  }

  /// returns the offset y coordinate
  int_t offset_y()const{
    return offset_y_;
  }

  /// intensity accessors:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param x image coordinate x
  /// \param y image coordinate y
  const intensity_t& operator()(const int_t x, const int_t y) const {
#if DICE_KOKKOS
    return intensities_.h_view(y,x);
#else
    return intensities_[y*width_+x];
#endif
  }

  /// intensity accessors:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param i pixel index
  const intensity_t& operator()(const int_t i) const {
#if DICE_KOKKOS
    const int_t y = i / width_;
    const int_t x = i - y*width_;
    return intensities_.h_view(y,x);
#else
    return intensities_[i];
#endif
  }

#if DICE_KOKKOS
  /// returns the view of the intensity values
  intensity_dual_view_2d intensities()const{
    return intensities_;
  }

  /// gradient x dual view accessor
  scalar_dual_view_2d grad_x()const{
    return grad_x_;
  }

  /// gradient y dual view accessor
  scalar_dual_view_2d grad_y()const{
    return grad_y_;
  }

  /// mask dual view accessor
  scalar_dual_view_2d mask() const{
    return mask_;
  }
#endif

  /// returns a copy of the intenisity values as an array
  Teuchos::ArrayRCP<intensity_t> intensity_array()const;

  /// replaces the intensity values of the image
  /// \param intensities the new intensity value array
  void replace_intensities(Teuchos::ArrayRCP<intensity_t> intensities);

  /// gradient accessors:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& grad_x(const int_t x,
    const int_t y) const {
#if DICE_KOKKOS
    return grad_x_.h_view(y,x);
#else
    return grad_x_[y*width_+x];
#endif
  }

  /// gradient accessor for y
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& grad_y(const int_t x,
    const int_t y) const {
#if DICE_KOKKOS
    return grad_y_.h_view(y,x);
#else
    return grad_y_[y*width_+x];
#endif
  }

  /// mask value accessor
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& mask(const int_t x,
    const int_t y) const {
#if DICE_KOKKOS
    return mask_.h_view(y,x);
#else
    return mask_[y*width_+x];
#endif
  }

  /// create the image mask field, but don't apply it to the image
  /// For the area_def, the boundary defines the outer edge of the region for which the
  /// mask will be set to 1.0. For the excluded region within the boundary, the mask
  /// will be set to 0.0
  /// mask values will be 1.0 for excluded regions.
  /// note: mask coordinates are always global, if the image is a portion of a larger
  /// image, the offsets will be applied to the coordinates to align the mask
  /// \param area_def defines the shape of the mask and what is included/excluded
  /// \param smooth_edges smooths the edges of the mask to avoid high freq. content
  void create_mask(const Conformal_Area_Def & area_def,
    const bool smooth_edges=true);

  /// creates the image mask and then applies it to the intensity values
  /// For the area_def, the boundary defines the outer edge of the region for which the
  /// mask will be set to 1.0. For the excluded region within the boundary, the mask
  /// will be set to 0.0
  /// mask values will be 1.0 for excluded regions.
  /// note: mask coordinates are always global, if the image is a portion of a larger
  /// image, the offsets will be applied to the coordinates to align the mask
  /// \param area_def defines the shape of the mask and what is included/excluded
  /// \param smooth_edges smooths the edges of the mask to avoid high freq. content
  void apply_mask(const Conformal_Area_Def & area_def,
    const bool smooth_edges=true);

  /// apply the mask field to the image and sync up the arrays between device and host
  /// assumes that the mask field is already populated
  /// \param smooth_edges use Gaussian smoothing along the edges
  void apply_mask(const bool smooth_edges);

  /// apply a transformation to this image to create another image
  /// \param deformation the deformation mapping parameters u,v,theta,...
  /// \param apply_in_place true if the mapped intensity values should replace the existing values in the image
  /// (in this case return is null pointer)
  /// \param cx centroid of mapping in the current image (used when applying rotation)
  /// \param cy centroid of mapping in the current image (used when applying rotation)
  Teuchos::RCP<Image> apply_transformation(Teuchos::RCP<const std::vector<scalar_t> > deformation,
    const bool apply_in_place=false,
    int_t cx=-1,
    int_t cy=-1);

  /// compute the image gradients
  void compute_gradients(const bool use_hierarchical_parallelism=false,
    const int_t team_size=256);

  /// returns true if the gradients have been computed
  bool has_gradients()const{
    return has_gradients_;
  }

  /// filter the image using a 7 point gauss filter
  void gauss_filter(const bool use_hierarchical_parallelism=false,
    const int_t team_size=256);

  /// returns the name of the file if available
  std::string file_name()const{
    return file_name_;
  }

  /// set the filename for the image
  /// \param file_name the string name to use
  void set_file_name(std::string & file_name){
    file_name_ = file_name;
  }

  /// returns the difference of two images:
  scalar_t diff(Teuchos::RCP<Image> rhs)const;

  /// returns the size of the gauss filter mask
  int_t gauss_filter_mask_size()const{
    return gauss_filter_mask_size_;
  }

#if DICE_KOKKOS
  //
  // Kokkos functors:
  //
  /// tag
  struct Init_Mask_Tag {};
  /// initialize a scalar dual view
  KOKKOS_INLINE_FUNCTION
  void operator()(const Init_Mask_Tag &, const int_t pixel_index) const;

  /// tag
  struct Grad_Flat_Tag {};
  /// compute the image gradient using a flat algorithm (no hierarchical parallelism)
  KOKKOS_INLINE_FUNCTION
  void operator()(const Grad_Flat_Tag &, const int_t pixel_index)const;
  /// tag
  struct Grad_Tag {};
  /// compute the image gradient using a heirarchical algorithm
  KOKKOS_INLINE_FUNCTION
  void operator()(const Grad_Tag &, const member_type team_member)const;

  /// tag
  struct Gauss_Flat_Tag{};
  /// Gauss filter the image
  KOKKOS_INLINE_FUNCTION
  void operator()(const Gauss_Flat_Tag &, const int_t pixel_index)const;
  /// tag
  struct Gauss_Tag{};
  /// Gauss filter the image
  KOKKOS_INLINE_FUNCTION
  void operator()(const Gauss_Tag &, const member_type team_member)const;
#endif

private:
  /// pixel container width_
  int_t width_;
  /// pixel container height_
  int_t height_;
  /// offsets are used to convert to global image coordinates
  /// (the pixel container may be a subset of a larger image)
  int_t offset_x_;
  /// offsets are used to convert to global image coordinates
  /// (the pixel container may be a subset of a larger image)
  int_t offset_y_;
  /// rcp to the intensity array (used to ensure it doesn't get deallocated)
  Teuchos::ArrayRCP<intensity_t> intensity_rcp_;
#if DICE_KOKKOS
  /// pixel container
  intensity_dual_view_2d intensities_;
  /// device intensity work array
  intensity_device_view_2d intensities_temp_;
  /// mask coefficients
  scalar_dual_view_2d mask_;
  /// image gradient x container
  scalar_dual_view_2d grad_x_;
  /// image gradient y container
  scalar_dual_view_2d grad_y_;
#else
  /// pixel container
  Teuchos::ArrayRCP<intensity_t> intensities_;
  /// device intensity work array
  Teuchos::ArrayRCP<intensity_t> intensities_temp_;
  /// mask coefficients
  Teuchos::ArrayRCP<scalar_t> mask_;
  /// image gradient x container
  Teuchos::ArrayRCP<scalar_t> grad_x_;
  /// image gradient y container
  Teuchos::ArrayRCP<scalar_t> grad_y_;
#endif
  /// flag that the gradients have been computed
  bool has_gradients_;
  /// coeff used in computing gradients
  scalar_t grad_c1_;
  /// coeff used in computing gradients
  scalar_t grad_c2_;
  /// Gauss filter coefficients
  scalar_t gauss_filter_coeffs_[13][13]; // 13 is the maximum size for the filter window
  /// Gauss filter mask size
  int_t gauss_filter_mask_size_;
  /// half the gauss filter mask size
  int_t gauss_filter_half_mask_;
  /// name of the file that was the source of this image
  std::string file_name_;
};

/// free function to apply a transformation to an image intensity array:
/// \param intensities_from the array where the intensities are taken
/// \param intensities_to the output array
/// \param cx the centroid x coordiante
/// \param cy the centroid y coordinate
/// \param width the width of the array
/// \param height the height of the array
void apply_transform(Teuchos::ArrayRCP<intensity_t> & intensities_from,
  Teuchos::ArrayRCP<intensity_t> & intensities_to,
  const int_t cx,
  const int_t cy,
  const int_t width,
  const int_t height,
  Teuchos::RCP<const std::vector<scalar_t> > deformation);

#if DICE_KOKKOS
/// image mask initialization functor
/// note, the number of pixels is the size of the x and y arrays, not the image
struct Mask_Init_Functor{
  /// pointer to the mask dual view
  scalar_device_view_2d mask_;
  /// pointer to the array of x values to enable (mask = 1.0)
  pixel_coord_device_view_1d x_;
  /// pointer to the array of y values to enable (mask = 1.0)
  pixel_coord_device_view_1d y_;
  /// all other pixels will set to mask = 0.0
  /// constructor
  /// \param mask pointer to the mask array on the device
  /// \param x pointer to the array of x coordinates on the device
  /// \param y pointer to the array of y coordinates on the device
  Mask_Init_Functor(scalar_device_view_2d mask,
    pixel_coord_device_view_1d x,
    pixel_coord_device_view_1d y):
    mask_(mask),
    x_(x),
    y_(y){};
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index)const{
    mask_(y_(pixel_index),x_(pixel_index)) = 1.0;
  }
};

/// image mask initialization functor
/// note, the number of pixels is the size the image
struct Mask_Smoothing_Functor{
  /// pointer to the mask dual view
  scalar_dual_view_2d mask_;
  /// pointer to a temporary copy of the mask field prior to smoothing
  scalar_device_view_2d mask_tmp_;
  /// width of the image
  int_t width_;
  /// height of the image
  int_t height_;
  /// gauss filter coefficients
  scalar_t gauss_filter_coeffs_[5][5];
  /// constructor
  /// \param mask pointer to the mask array on the device
  /// \param width the width of the image
  /// \param height the height of the image
  Mask_Smoothing_Functor(scalar_dual_view_2d mask,
    const int_t width,
    const int_t height);
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index)const;
};

/// image mask apply functor
struct Mask_Apply_Functor{
  /// pointer to the intensity values
  intensity_device_view_2d intensities_;
  /// pointer to the mask dual view
  scalar_device_view_2d mask_;
  /// image width
  int_t width_;
  /// constructor
  /// \param intensities pointer to the intensity array on the device
  /// \param mask pointer to the mask array on the device
  /// \param width the image width
  Mask_Apply_Functor(intensity_device_view_2d intensities,
    scalar_device_view_2d mask,
    int_t width):
    intensities_(intensities),
    mask_(mask),
    width_(width){};
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index)const{
    const int_t y = pixel_index / width_;
    const int_t x = pixel_index - y*width_;
    intensities_(y,x) = mask_(y,x)*intensities_(y,x);
  }
};

/// image transformation functor
/// given parameters theta, u, and v, transform the given image
/// uses the keys interpolant (TODO add other interpolants)
struct Transform_Functor{
  /// pointer to the intensity dual view
  intensity_device_view_2d intensities_from_;
  /// pointer to the intensity dual view
  intensity_device_view_2d intensities_to_;
  /// centroid in x (note this is the transformed centroid)
  scalar_t cx_;
  /// centroid in y (node this is the transformed centroid)
  scalar_t cy_;
  /// width of the image
  int_t width_;
  /// height of the image
  int_t height_;
  /// displacement x;
  scalar_t u_;
  /// displacement y;
  scalar_t v_;
  /// rotation angle
  scalar_t t_;
  /// normal strain x
  scalar_t ex_;
  /// normal strain y
  scalar_t ey_;
  /// shear strain xy
  scalar_t g_;
  /// cosine of theta
  scalar_t cost_;
  /// sin of theta
  scalar_t sint_;
  /// tolerance
  scalar_t tol_;
  /// constructor
  /// \param intensities_from pointer to the intensity array to be transformed
  /// \param intensities_to pointer to the result intensity array
  /// \param width the width of the image
  /// \param height the height of the image
  /// \param cx centroid in x
  /// \param cy centroid in y
  /// \param def the deformation map parameters
  Transform_Functor(intensity_device_view_2d intensities_from,
    intensity_device_view_2d intensities_to,
    const int_t width,
    const int_t height,
    const int_t cx,
    const int_t cy,
    Teuchos::RCP<const std::vector<scalar_t> > def):
    intensities_from_(intensities_from),
    intensities_to_(intensities_to),
    cx_(cx + (*def)[DISPLACEMENT_X]),
    cy_(cy + (*def)[DISPLACEMENT_Y]),
    width_(width),
    height_(height),
    u_((*def)[DISPLACEMENT_X]),
    v_((*def)[DISPLACEMENT_Y]),
    t_((*def)[ROTATION_Z]),
    ex_((*def)[NORMAL_STRAIN_X]),
    ey_((*def)[NORMAL_STRAIN_Y]),
    g_((*def)[SHEAR_STRAIN_XY]),
    tol_(0.00001){
    cost_ = std::cos(t_);
    sint_ = std::sin(t_);
  };
  /// operator
  KOKKOS_INLINE_FUNCTION
  void operator()(const int_t pixel_index)const;
  /// Tag
  struct Rot_180_Tag {};
  /// operator
  /// if the requested transformation is rotation by 180 degrees,
  /// its faster to swap the x and y indices
  KOKKOS_INLINE_FUNCTION
  void operator()(const Rot_180_Tag&, const int_t pixel_index)const;
};
#endif
}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
