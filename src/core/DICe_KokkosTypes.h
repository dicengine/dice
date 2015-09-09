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

#ifndef DICE_KOKKOSTYPES_H
#define DICE_KOKKOSTYPES_H

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include <DICe.h>

namespace DICe{

/// kokkos view types

typedef Kokkos::DefaultExecutionSpace device_space;
typedef Kokkos::HostSpace host_space;

/// 2 dimensional array of intensity values for the device
typedef Kokkos::View<intensity_t **, Kokkos::MemoryTraits<Kokkos::RandomAccess> > intensity_device_view_t;
/// host mirrors of the intensity value arrays
typedef intensity_device_view_t::HostMirror intensity_host_view_t;

/// 2 dimensional dual view of intensity type values
typedef Kokkos::DualView<intensity_t **,Kokkos::MemoryTraits<Kokkos::RandomAccess> >  intensity_2d_t;

/// 2 dimensional dual view of scalar type values
typedef Kokkos::DualView<scalar_t **>  scalar_2d_t;

} // end DICe namespace

#endif
