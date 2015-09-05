// @HEADER
// ************************************************************************
//
//               Digital Image Correlation Engine (DICe)
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact:
//              Dan Turner   (danielzturner@gmail.com)
//
// ************************************************************************
// @HEADER

#ifndef DICE_H
#define DICE_H

#define DICE_PI 3.14159265358979323846
#define DICE_TWOPI 6.28318530717958647692

#if (defined(WIN32) || defined(WIN64))
#  if defined(DICECORE_LIB_EXPORTS_MODE)
#    define DICECORE_LIB_DLL_EXPORT __declspec(dllexport)
#  else
#    define DICECORE_LIB_DLL_EXPORT __declspec(dllimport)
#  endif
#else
#  define DICECORE_LIB_DLL_EXPORT
#endif

// debugging macros:
#ifdef DICE_DEBUG_MSG
#  define DEBUG_MSG(x) do { std::cout << "[DICe_DEBUG]: " << x << std::endl; } while (0)
#else
#  define DEBUG_MSG(x) do {} while (0)
#endif

#include <Kokkos_Core.hpp>

namespace DICe{

/// basic types

/// image intensity type
typedef float intensity_t;
/// generic scalar type
typedef double scalar_t;

/// kokkos view types

/// 2 dimensional array of intensity values for the device
typedef Kokkos::View<intensity_t **> intensity_device_view_t;
/// host mirrors of the intensity value arrays
typedef intensity_device_view_t::HostMirror intensity_host_view_t;

} // end DICe namespace

#endif
