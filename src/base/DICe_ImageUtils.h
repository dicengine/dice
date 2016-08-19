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
void apply_transform(Teuchos::RCP<Image> image_in,
  Teuchos::RCP<Image> image_out,
  const int_t cx,
  const int_t cy,
  Teuchos::RCP<const std::vector<scalar_t> > deformation);

/// \class SinCos_Image_Deformer
/// \brief a class that deformed an input image according to a sin()*cos() function
class
DICE_LIB_DLL_EXPORT
SinCos_Image_Deformer{
public:

  /// constructor
  /// \param num_steps the number of frequency steps to superimpose
  /// \param use_superposition true if the sin curves should be added together
  SinCos_Image_Deformer(const int_t num_steps,
    bool use_superposition = false):
    num_steps_(num_steps),
    use_superposition_(use_superposition){};

  /// parameterless constructor
  SinCos_Image_Deformer():
    num_steps_(1),
    use_superposition_(false){};

  /// perform deformation on the image
  /// returns a pointer to the deformed image
  /// \param ref_image the reference image
  Teuchos::RCP<Image> deform_image(Teuchos::RCP<Image> ref_image);

  /// compute the analytical displacement at the given coordinates
  /// \param coord_x the x-coordinate for the evaluation location
  /// \param coord_y the y-coordinate
  /// \param bx [out] the x displacement
  /// \param by [out] the y displacement
  void compute_deformation(const scalar_t & coord_x,
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
  void compute_deriv_deformation(const scalar_t & coord_x,
    const scalar_t & coord_y,
    scalar_t & bxx,
    scalar_t & bxy,
    scalar_t & byx,
    scalar_t & byy);

  /// compute the error of a given solution at the given coords
  /// \param coord_x the x coordinate
  /// \param coord_y the y coordinate
  /// \param sol_x the solution value in x
  /// \param sol_y the solution value in y
  /// \param error_x [out] the x error at the given coords
  /// \param error_y [out] the y error at the given coords
  void compute_displacement_error(const scalar_t & coord_x,
    const scalar_t & coord_y,
    const scalar_t & sol_x,
    const scalar_t & sol_y,
    scalar_t & error_x,
    scalar_t & error_y);

  /// compute the error of the derivative of a given solution at the given coords
  /// \param coord_x the x coordinate
  /// \param coord_y the y coordinate
  /// \param sol_x the solution value in x
  /// \param sol_y the solution value in y
  /// \param error_x [out] the x error at the given coords
  /// \param error_y [out] the y error at the given coords
  // TODO add off-diagonal terms
  void compute_deriv_error(const scalar_t & coord_x,
    const scalar_t & coord_y,
    const scalar_t & sol_x,
    const scalar_t & sol_y,
    scalar_t & error_x,
    scalar_t & error_y);

  /// destructor
  ~SinCos_Image_Deformer(){};

private:
  /// number of frequency steps to impose
  int_t num_steps_;
  /// use superposition for each step
  bool use_superposition_;

};

}// End DICe Namespace

/*! @} End of Doxygen namespace*/

#endif
