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
#ifndef DICE_IMAGEUTILS_H
#define DICE_IMAGEUTILS_H

#include <DICe.h>
#include <DICe_Image.h>
#ifdef DICE_TPETRA
  #include "DICe_MultiFieldTpetra.h"
#else
  #include "DICe_MultiFieldEpetra.h"
#endif

#include <Teuchos_ParameterList.hpp>

/*!
 *  \namespace DICe
 *  @{
 */
/// generic DICe classes and functions
namespace DICe {

// forward declaration of DICe::Image
class Image;

/// free function to apply a transformation to an image:
/// \param image_in the image where the intensities are taken
/// \param image_out the output image
/// \param cx the centroid x coordiante
/// \param cy the centroid y coordinate
/// \param deformation a vector that defines the deformation map parameters
DICE_LIB_DLL_EXPORT
void apply_transform(Teuchos::RCP<Image> image_in,
  Teuchos::RCP<Image> image_out,
  const int_t cx,
  const int_t cy,
  Teuchos::RCP<const std::vector<scalar_t> > deformation);

/// struct used to sort solutions by peak values
struct computed_point
{
    scalar_t x_, y_, bx_, by_, sol_bx_, sol_by_,error_bx_,error_by_;
};

/// free function to compute the roll off statistics for the peaks in the exact solution
/// \param period, the period of the exact displacement
/// \param img_w the width of the image
/// \param img_h the height of the image
/// \param coords the coordinates field
/// \param disp the computed displacement field
/// \param exact_disp the exact displacement field
/// \param disp_error the error field
/// \param peaks_avg_error_x [out] computed stat
/// \param peaks_std_dev_error_x [out] computed stat
/// \param peaks_avg_error_y [out] computed stat
/// \param peaks_std_dev_error_y [out] computed stat
DICE_LIB_DLL_EXPORT
void compute_roll_off_stats(const scalar_t & period,
  const scalar_t & img_w,
  const scalar_t & img_h,
  Teuchos::RCP<MultiField> & coords,
  Teuchos::RCP<MultiField> & disp,
  Teuchos::RCP<MultiField> & exact_disp,
  Teuchos::RCP<MultiField> & disp_error,
  scalar_t & peaks_avg_error_x,
  scalar_t & peaks_std_dev_error_x,
  scalar_t & peaks_avg_error_y,
  scalar_t & peaks_std_dev_error_y);

/// free function to create a synthetically speckled image with perfectly smooth and regular speckles of a certain size
/// \param w width of the image
/// \param h height of the image
/// \param offset_x the x offset for a sub image
/// \param offset_y the y offset for a sub image
/// \param speckle_size size of the speckles to create
/// \param params set of image parameters (compute gradients, etc)
DICE_LIB_DLL_EXPORT
Teuchos::RCP<Image> create_synthetic_speckle_image(const int_t w,
  const int_t h,
  const int_t offset_x,
  const int_t offset_y,
  const scalar_t & speckle_size,
  const Teuchos::RCP<Teuchos::ParameterList> & params=Teuchos::null);

/// free function to add noise counts to an image
/// \param image the image to modify
/// \param noise_percent the amount of noise to add in percentage of the maximum intensity
DICE_LIB_DLL_EXPORT
void add_noise_to_image(Teuchos::RCP<Image> & image,
  const scalar_t & noise_percent);

/// free function to determine the distribution of speckle sizes
/// returns the next largest odd integer size (so if the pattern predominant size is 6, the function returns 7)
/// \param output_dir the directory to save the statistics file in
/// \param image the image containing the speckle pattern
/// should only be run on processor zero (has no parallel smarts)
DICE_LIB_DLL_EXPORT
int_t compute_speckle_stats(const std::string & output_dir,
  Teuchos::RCP<Image> & image);

/// \class Image_Deformer
/// \brief base class that deformes an input image according to an analytical function
class
DICE_LIB_DLL_EXPORT
Image_Deformer{
public:

  /// constructor
  Image_Deformer(const scalar_t & rel_factor_disp,
    const scalar_t & rel_factor_strain):
  rel_factor_disp_(rel_factor_disp),
  rel_factor_strain_(rel_factor_strain){};
  virtual ~Image_Deformer(){};

  /// compute the analytical displacement at the given coordinates
  /// \param coord_x the x-coordinate for the evaluation location
  /// \param coord_y the y-coordinate
  /// \param bx [out] the x displacement
  /// \param by [out] the y displacement
  virtual void compute_deformation(const scalar_t & coord_x,
    const scalar_t & coord_y,
    scalar_t & bx,
    scalar_t & by){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Cannot call this base class method")};

  /// compute the analytical derivatives at the given coordinates
  /// \param coord_x the x-coordinate for the evaluation location
  /// \param coord_y the y-coordinate
  /// \param bxx [out] the xx deriv
  /// \param bxy [out] the xy deriv
  /// \param byx [out] the yx deriv
  /// \param byy [out] the yy deriv
  virtual void compute_deriv_deformation(const scalar_t & coord_x,
    const scalar_t & coord_y,
    scalar_t & bxx,
    scalar_t & bxy,
    scalar_t & byx,
    scalar_t & byy){TEUCHOS_TEST_FOR_EXCEPTION(true,std::runtime_error,"Cannot call this base class method")};

  /// perform deformation on the image
  /// returns a pointer to the deformed image
  /// \param ref_image the reference image
  Teuchos::RCP<Image> deform_image(Teuchos::RCP<Image> ref_image);

  /// compute the error of a given solution at the given coords
  /// \param coord_x the x coordinate
  /// \param coord_y the y coordinate
  /// \param sol_x the solution value in x
  /// \param sol_y the solution value in y
  /// \param error_x [out] the x error at the given coords
  /// \param error_y [out] the y error at the given coords
  /// \param use_mag square the difference if true
  /// \param relative divide by the magnitude of the solution at that point (%)
  void compute_displacement_error(const scalar_t & coord_x,
    const scalar_t & coord_y,
    const scalar_t & sol_x,
    const scalar_t & sol_y,
    scalar_t & error_x,
    scalar_t & error_y,
    const bool use_mag = false,
    const bool relative = true);

  /// compute the error of the derivative of a given solution at the given coords
  /// \param coord_x the x coordinate
  /// \param coord_y the y coordinate
  /// \param sol_xx the solution value in xx
  /// \param sol_xy the solution value in xy
  /// \param sol_yy the solution value in yy
  /// \param error_xx [out] the xx error at the given coords
  /// \param error_xy [out] the xy error at the given coords
  /// \param error_yy [out] the yy error at the given coords
  /// \param use_mag square the difference if true
  /// \param relative divide by the magnitude of the solution at that point (%)
  void compute_lagrange_strain_error(const scalar_t & coord_x,
    const scalar_t & coord_y,
    const scalar_t & sol_xx,
    const scalar_t & sol_xy,
    const scalar_t & sol_yy,
    scalar_t & error_xx,
    scalar_t & error_xy,
    scalar_t & error_yy,
    const bool use_mag = false,
    const bool relative = true);

  /// compute the lagrange strain given solution at the given coords
  /// \param coord_x the x coordinate
  /// \param coord_y the y coordinate
  /// \param strain_xx [out] the xx strain at the given coords
  /// \param strain_xy [out] the xy strain at the given coords
  /// \param strain_yy [out] the yy strain at the given coords
  void compute_lagrange_strain(const scalar_t & coord_x,
    const scalar_t & coord_y,
    scalar_t & strain_xx,
    scalar_t & strain_xy,
    scalar_t & strain_yy);
private:
  /// factor to use when computing the relative values of disp error
  scalar_t rel_factor_disp_;
  /// factor to use when computing the relative values of strain error
  scalar_t rel_factor_strain_;
};


/// \class SinCos_Image_Deformer
/// \brief a class that deformed an input image according to a sin()*cos() function
class
DICE_LIB_DLL_EXPORT
SinCos_Image_Deformer : public Image_Deformer
{
public:

  /// constructor
  /// \param period the period of the motion in pixels
  /// \param amplitude the amplitude of the motion in pixels
  SinCos_Image_Deformer(const scalar_t & period,
    const scalar_t & amplitude):
      Image_Deformer(amplitude/200.0,(1.0/period)*DICE_TWOPI*amplitude/200.0),
      period_(period),
      amplitude_(amplitude){};

  /// parameterless constructor
  SinCos_Image_Deformer():
    Image_Deformer(1.0/200.0,(1.0/100.0)*DICE_TWOPI*1.0/200.0),
    period_(100),
    amplitude_(1){};

  /// returns the current amplitude
  scalar_t amplitude(){
    return amplitude_;
  }
  /// returns the current period
  scalar_t period(){
    return period_;
  }

  /// compute the analytical displacement at the given coordinates
  /// \param coord_x the x-coordinate for the evaluation location
  /// \param coord_y the y-coordinate
  /// \param bx [out] the x displacement
  /// \param by [out] the y displacement
  virtual void compute_deformation(const scalar_t & coord_x,
    const scalar_t & coord_y,
    scalar_t & bx,
    scalar_t & by);

  /// compute the analytical derivatives at the given coordinates
  /// \param coord_x the x-coordinate for the evaluation location
  /// \param coord_y the y-coordinate
  /// \param bxx [out] the xx deriv
  /// \param bxy [out] the xy deriv
  /// \param byx [out] the yx deriv
  /// \param byy [out] the yy deriv
  virtual void compute_deriv_deformation(const scalar_t & coord_x,
    const scalar_t & coord_y,
    scalar_t & bxx,
    scalar_t & bxy,
    scalar_t & byx,
    scalar_t & byy);

  /// destructor
  virtual ~SinCos_Image_Deformer(){};

private:
  /// period of the motion
  scalar_t period_;
  /// amplitude of the motion
  scalar_t amplitude_;

};

/// \class ConstantValue_Image_Deformer
/// \brief a class that provides an exact solution that is constant valued
/// The image deforming
class
DICE_LIB_DLL_EXPORT
ConstantValue_Image_Deformer: public Image_Deformer
{
public:

  /// constructor
  /// \param value value of the motion in pixels
  ConstantValue_Image_Deformer(const scalar_t & value_x,
    const scalar_t & value_y):
    Image_Deformer(1.0,1.0),
    value_x_(value_x),
    value_y_(value_y){};

  /// parameterless constructor
  ConstantValue_Image_Deformer():
    Image_Deformer(1.0,1.0),
    value_x_(0.5),
    value_y_(0.5){};

  /// compute the analytical displacement at the given coordinates
  /// \param coord_x the x-coordinate for the evaluation location
  /// \param coord_y the y-coordinate
  /// \param bx [out] the x displacement
  /// \param by [out] the y displacement
  virtual void compute_deformation(const scalar_t & coord_x,
    const scalar_t & coord_y,
    scalar_t & bx,
    scalar_t & by){
    bx = value_x_;
    by = value_y_;
  }
  /// compute the analytical derivatives at the given coordinates
  /// \param coord_x the x-coordinate for the evaluation location
  /// \param coord_y the y-coordinate
  /// \param bxx [out] the xx deriv
  /// \param bxy [out] the xy deriv
  /// \param byx [out] the yx deriv
  /// \param byy [out] the yy deriv
  virtual void compute_deriv_deformation(const scalar_t & coord_x,
    const scalar_t & coord_y,
    scalar_t & bxx,
    scalar_t & bxy,
    scalar_t & byx,
    scalar_t & byy){
    bxx = 0.0;
    bxy = 0.0;
    byx = 0.0;
    byy = 0.0;
  }

  /// destructor
  virtual ~ConstantValue_Image_Deformer(){};

private:
  /// magnitude of the motion
  scalar_t value_x_;
  scalar_t value_y_;
};




}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
