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
#ifndef DICE_IMAGE_H
#define DICE_IMAGE_H

#include <DICe.h>
#if DICE_KOKKOS
  #include <DICe_Kokkos.h>
#endif
#include <Teuchos_ParameterList.hpp>
namespace DICe {

/// forward declaration of the conformal_area_def
class Conformal_Area_Def;
class Local_Shape_Function;

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
  /// \param offset_x the x offset for a sub image
  /// \param offset_y the y offset for a sub image
  Image(const int_t width,
    const int_t height,
    Teuchos::ArrayRCP<intensity_t> intensities,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null,
    const int_t offset_x = 0,
    const int_t offset_y = 0);

  //
  // Empty (zero) image
  //

  /// constructor that creates a zero image
  /// \param width the width of the image
  /// \param height the height of the image
  /// \param intensity value to fill the array with
  /// \param offset_x offset to upper left corner x coord if this is a subregion of a larger image
  /// \param offset_y offset to upper left corner y coord if this is a subregion of a larger image
  /// no params allowed since the intensity values are all zeros so gradients
  /// or filters would not make sense
  Image(const int_t width,
    const int_t height,
    const intensity_t intensity=0.0,
    const int_t offset_x = 0,
    const int_t offset_y = 0);

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
  /// \param params image parameters (for example compute_gradients, etc.)
  Image(Teuchos::RCP<Image> img,
    const int_t offset_x = 0,
    const int_t offset_y = 0,
    const int_t width = -1,
    const int_t height = -1,
    const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// perform initialization of an image from an array
  /// \param intensities the array of intensity values
  void initialize_array_image(intensity_t * intensities);

  /// default constructor tasks
  void default_constructor_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// update the fields in an existing image
  void update_image_fields(const char * file_name,
    const Teuchos::RCP<Teuchos::ParameterList> & params);

  /// post allocation tasks
  void post_allocation_tasks(const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// virtual destructor
  virtual ~Image(){};

  /// write the image to a file
  /// (tiff, jpeg, or png, depending on which file extension is used in the name)
  /// Tiff, jpeg, and png will truncate the intensity values to an 8-bit integer value
  /// and scale the image so that the histogram is spread over the entire 0-255 range.
  /// The rawi format saves the full intesity_t precision value to file
  /// \param file_name the name of the file to write to
  /// \param scale_to_8_bit scale image to eight bit range if true
  void write(const std::string & file_name,
    const bool scale_to_8_bit=true);

  /// write an image to file that combines this image and another of the same size
  /// with both overlayed using transparency
  /// \param file_name the name of the file to output
  /// \param top_img pointer to the image to be overlayed on top of this one
  void write_overlap_image(const std::string & file_name,
    Teuchos::RCP<Image> top_img);

  /// write the image x gradients to a file
  /// (tiff, jpeg, or png, depending on which file extension is used in the name)
  /// Tiff, jpeg, and png will truncate the intensity values to an 8-bit integer value
  /// and scale the image so that the histogram is spread over the entire 0-255 range.
  /// The rawi format saves the full intesity_t precision value to file
  /// \param file_name the name of the file to write to
  void write_grad_x(const std::string & file_name);

  /// write the image y gradients to a file
  /// (tiff, jpeg, or png, depending on which file extension is used in the name)
  /// Tiff, jpeg, and png will truncate the intensity values to an 8-bit integer value
  /// and scale the image so that the histogram is spread over the entire 0-255 range.
  /// The rawi format saves the full intesity_t precision value to file
  /// \param file_name the name of the file to write to
  void write_grad_y(const std::string & file_name);

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

  /// returns the mean value of the image
  scalar_t mean()const;

  /// intensity accessors:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param x image coordinate x
  /// \param y image coordinate y
  const intensity_t& operator()(const int_t x, const int_t y) const;

  /// intensity accessors:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param i pixel index
  const intensity_t& operator()(const int_t i) const;

  /// returns a copy of the intenisity values as an array
  Teuchos::ArrayRCP<intensity_t> intensities()const;

  /// returns a copy of the grad_x values as an array
  Teuchos::ArrayRCP<scalar_t> grad_x_array()const;

  /// returns a copy of the grad_y values as an array
  Teuchos::ArrayRCP<scalar_t> grad_y_array()const;

  /// replaces the intensity values of the image
  /// \param intensities the new intensity value array
  void replace_intensities(Teuchos::ArrayRCP<intensity_t> intensities);

  /// interpolate intensity and gradients
  void interpolate_keys_fourth_all(intensity_t& intensity_val,
       scalar_t& grad_x_val, scalar_t& grad_y_val, const bool compute_gradient,
       const scalar_t& local_x, const scalar_t& local_y);


  /// interpolant
  /// \param global_x global image coordinate x
  /// \param global_y global image coordinate y
  intensity_t interpolate_keys_fourth_global(const scalar_t & global_x,
    const scalar_t & global_y){
    return interpolate_keys_fourth(global_x-offset_x_,global_y-offset_y_);
  }

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  intensity_t interpolate_keys_fourth(const scalar_t & local_x,
    const scalar_t & local_y);

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  scalar_t interpolate_grad_x_keys_fourth(const scalar_t & local_x,
    const scalar_t & local_y);

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  scalar_t interpolate_grad_y_keys_fourth(const scalar_t & local_x,
    const scalar_t & local_y);

  /// interpolant
  /// \param global_x global image coordinate x
  /// \param global_y global image coordinate y
  intensity_t interpolate_bilinear_global(const scalar_t & global_x,
    const scalar_t & global_y){
    return interpolate_bilinear(global_x-offset_x_,global_y-offset_y_);
  }

  /// interpolate intensity and gradients
  void interpolate_bilinear_all(intensity_t& intensity_val,
       scalar_t& grad_x_val, scalar_t& grad_y_val, const bool compute_gradient,
       const scalar_t& local_x, const scalar_t& local_y);

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  intensity_t interpolate_bilinear(const scalar_t & local_x,
    const scalar_t & local_y);

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  scalar_t interpolate_grad_x_bilinear(const scalar_t & local_x,
    const scalar_t & local_y);

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  scalar_t interpolate_grad_y_bilinear(const scalar_t & local_x,
    const scalar_t & local_y);

  /// interpolant
  /// \param global_x global image coordinate x
  /// \param global_y global image coordinate y
  intensity_t interpolate_bicubic_global(const scalar_t & global_x,
    const scalar_t & global_y){
    return interpolate_bicubic(global_x-offset_x_,global_y-offset_y_);
  }

  /// interpolate intensity and gradients
  void interpolate_bicubic_all(intensity_t& intensity_val,
       scalar_t& grad_x_val, scalar_t& grad_y_val, const bool compute_gradient,
       const scalar_t& local_x, const scalar_t& local_y);

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  intensity_t interpolate_bicubic(const scalar_t & local_x,
    const scalar_t & local_y);

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  scalar_t interpolate_grad_x_bicubic(const scalar_t & local_x,
    const scalar_t & local_y);

  /// interpolant
  /// \param local_x local image coordinate x
  /// \param local_y local image coordinate y
  scalar_t interpolate_grad_y_bicubic(const scalar_t & local_x,
    const scalar_t & local_y);

  /// gradient accessors:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& grad_x(const int_t x,
    const int_t y) const;

  /// gradient accessor for y
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& grad_y(const int_t x,
    const int_t y) const ;

  /// laplacian accessor:
  /// note the internal arrays are stored as (row,column) so the indices have to be switched from coordinates x,y to y,x
  /// y is row, x is column
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& laplacian(const int_t x,
    const int_t y) const;

  /// mask value accessor
  /// \param x image coordinate x
  /// \param y image coordinate y
  const scalar_t& mask(const int_t x,
    const int_t y) const ;

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
  /// \param shape_function the deformation mapping parameters u,v,theta,...
  /// (in this case return is null pointer)
  /// \param cx centroid of mapping in the current image (used when applying rotation)
  /// \param cy centroid of mapping in the current image (used when applying rotation)
  /// \param apply_in_place true if the mapped intensity values should replace the existing values in the image
  Teuchos::RCP<Image> apply_transformation(Teuchos::RCP<Local_Shape_Function> shape_function,
    const int_t cx,
    const int_t cy,
    const bool apply_in_place=false);

  /// normalize the image intensity values
  /// \param params the image parameters to use
  Teuchos::RCP<Image> normalize(const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// apply a rotation to this image to create another image
  /// in this case, there are only three options 90, 180, and 270 degree rotations
  /// \param rotation enum that defines the rotation
  /// \param params parameters to apply to the new image
  Teuchos::RCP<Image> apply_rotation(const Rotation_Value rotation,
      const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

  /// compute the image gradients
  void compute_gradients(const bool use_hierarchical_parallelism=false,
    const int_t team_size=256);

  /// compute the image gradients
  void smooth_gradients_convolution_5_point();

  /// compute the image gradients
  void compute_gradients_finite_difference();

  /// returns true if the gradients have been computed
  bool has_gradients()const{
    return has_gradients_;
  }

  /// returns true if the image is a frame from a video sequence cine or netcdf file
  bool is_video_frame()const;

  /// returns true if the image was created from a file not array
  bool has_file_name()const{
    return has_file_name_;
  }

  /// returns true if the image has been  gauss filtered
  bool has_gauss_filter()const{
    return has_gauss_filter_;
  }

  /// filter the image using a 7 point gauss filter
  void gauss_filter(const int_t mask_size=-1,const bool use_hierarchical_parallelism=false,
    const int_t team_size=256);

  /// sets the file name of the deformed image
  void set_file_name(const std::string & file_name) {
    file_name_ = file_name;
  }

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
  /// returns the view of the intensity values
  intensity_dual_view_2d intensity_dual_view()const{
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
  /// image laplacian container
  scalar_dual_view_2d laplacian_;
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
  /// image gradient y container
  Teuchos::ArrayRCP<scalar_t> laplacian_;
#endif
  /// flag that the gradients have been computed
  bool has_gradients_;
  /// flag that the image has been filtered
  bool has_gauss_filter_;
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
  /// true if the image was read from a file not created from an array
  bool has_file_name_;
  /// gradient method
  Gradient_Method gradient_method_;
};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
