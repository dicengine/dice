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

/*! \file  DICe_api.h
    \brief routines for calculating the rigid body motion of a
    set of subsets
*/

#ifndef DICE_API_H
#define DICE_API_H

#ifdef __cplusplus
extern "C" {
#endif

/// For windows need to determine if to export or import
#if (defined(WIN32) || defined(WIN64))
#  if defined(DICE_LIB_EXPORTS_MODE)
#    define DICE_LIB_DLL_EXPORT __declspec(dllexport)
#  else
#    define DICE_LIB_DLL_EXPORT __declspec(dllimport)
#  endif
#else
#  define DICE_LIB_DLL_EXPORT
#endif

/// The stride of the solution vector in the external application
#define DICE_API_STRIDE 9

/// Typedef integer as size type
typedef int int_t;
#if DICE_USE_DOUBLE
  /// image intensity type
  typedef double intensity_t;
  /// generic scalar type
  typedef double scalar_t;
#else
  /// Typedef double as real type
  typedef float scalar_t;
  /// Tyepdef double as pixel intensity type
  typedef float intensity_t;
#endif

// Forward declaration of ParameterList so the Teuchos headers do not
// have to be included
namespace Teuchos{
  class ParameterList;
}

/// \brief Function to execute a correlation on the data provided in points and the image arrays
///
/// \param points:      An array of (n_points * DICE_API_STRIDE) type Real values where
///                     the nine values mean:
///                     [x, y, u, v, theta, sigma, diagnostic_flag]
///                     If sigma < 0 is passed, an initial guess will be determined
///                     automatically, otherwise [u, v, theta] will be used.
/// \param n_points:    The number of points.
/// \param subset_size: The subset size to use for correlation.
/// \param ref_img:     The reference image as (ref_w) * (ref_h) type Real values.
/// \param ref_w:       The width of the reference image.
/// \param ref_h:       The height of the reference image.
/// \param def_img:     The deformed image as (def_w) * (def_h) type Real values.
/// \param def_w:       The width of the deformed image.
/// \param def_h:       The height of the deformed image.
/// \param input_params Optional ParameterList to change the default correlation parameters
/// \param update_params Optional flag to force schema to update its parameters (useful if the user changes
///                      the parameters mid-analysis.)
/// this function
///
/// Return values:
/// The function returns 0 on success, otherwise an error code is
/// returned.
///
/// The points array is modified as follows:
///  x:     unmodified
///  y:     unmodified
///  u:     the u-displacement of the point
///  v:     the v-displacement of the point
///  theta: the rotation of the point
///  sigma: On success: metric used to determine the variation in the displacement (uncertainty) given image noise
///         and interpolation bias, -1.0 is returned if the correlation fails.
///  gamma: Correlation quality measure (0.0 is perfect correlation, larger numbers are poor correlation), -1.0 is failed correlation
///  beta:  Sensitivity of the cost function (lower is better). If beta is high, the correlation criteria cannot judge between nearby solutions,
///         they all have the same cost function. -1.0 is recorded for failed correlation
///  diagnostic_flags: (see DICe_Types.h for definitions)
DICE_LIB_DLL_EXPORT const int_t dice_correlate(scalar_t points[], int_t n_points,
                        int_t subset_size,
                        intensity_t ref_img[], int_t ref_w, int_t ref_h,
                        intensity_t def_img[], int_t def_w, int_t def_h,
                        Teuchos::ParameterList * input_params=0, const bool update_params=false);

/// \brief Function to execute a correlation on the data provided in points and the image arrays
///
/// This signature enables the user to specify an optional conformal subset file as well as an
/// optional file with correlation parameters
///
/// \param points:      An array of (n_points * DICE_API_STRIDE) type Real values where
///                     the nine values mean:
///                     [x, y, u, v, theta, sigma, diagnostic_flag]
///                     If sigma < 0 is passed, an initial guess will be determined
///                     automatically, otherwise [u, v, theta] will be used.
/// \param ref_img:     The reference image as (ref_w) * (ref_h) type Real values.
/// \param ref_w:       The width of the reference image.
/// \param ref_h:       The height of the reference image.
/// \param def_img:     The deformed image as (def_w) * (def_h) type Real values.
/// \param def_w:       The width of the deformed image.
/// \param def_h:       The height of the deformed image.
/// \param subset_file  Required file name containing conformal subset definitions and subset centroids (see Doxygen help for format)
///                     For this library call, all subsets must have a conformal subset definition (even if they are squares, they
///                     have to be defined in the subset_file as a square (rectangular) conformal subset
/// \param param_file   Optional file name containing the user specified xml correlation parameters
/// \param write_output Optional flag to request DICe write output files, the files will be placed in the current execution directory
///
/// Return values:
/// The function returns 0 on success, otherwise an error code is
/// returned.
///
/// The points array is modified as follows:
///  x:     the subset centroid x coordinates (conformal subsets included)
///  y:     the subset centroid y coordinates (conformal subsets included)
///  u:     the u-displacement of the point
///  v:     the v-displacement of the point
///  theta: the rotation of the point
///  sigma: On success: metric used to determine the variation in the displacement (uncertainty) given image noise
///         and interpolation bias, -1.0 is returned if the correlation fails.
///  gamma: Correlation quality measure (0.0 is perfect correlation, larger numbers are poor correlation), -1.0 is failed correlation
///  beta:  Sensitivity of the cost function (lower is better). If beta is high, the correlation criteria cannot judge between nearby solutions,
///         they all have the same cost function. -1.0 is recorded for failed correlation
///  diagnostic_flags: (see DICe_Types.h for definitions)
///  Note: if the params are changed once this function has been called it will have no effect. The
///  correlation parameters are set at the first invokation and will remain the same.
///  Note: no checking is done to ensure that the points array is the right size. This is up to the user.
DICE_LIB_DLL_EXPORT const int_t dice_correlate_conformal(scalar_t points[],
                        intensity_t ref_img[], int_t ref_w, int_t ref_h,
                        intensity_t def_img[], int_t def_w, int_t def_h,
                        const char* subset_file, const char* param_file=0,
                        const bool write_output=false);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
