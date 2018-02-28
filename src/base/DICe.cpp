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

#include <DICe.h>

#include <iostream>

#if DICE_KOKKOS
  #include <DICe_Kokkos.h>
  #include <Kokkos_Core.hpp>
#endif
#if DICE_MPI
  #include <mpi.h>
#endif


namespace DICe {

DICE_LIB_DLL_EXPORT
void print_banner(){
  std::string mpi_message = "** MPI: disabled";
#if DICE_MPI
  mpi_message = "** MPI: enabled";
#endif
  std::string kokkos_message = "** Manycore: disabled";
#if DICE_KOKKOS
  kokkos_message = "** Manycore: enabled";
#endif
  std::string type_message = "** Data type: float";
#if DICE_USE_DOUBLE
  type_message = "** Data type: double";
#endif
  std::cout << std::endl << "** Digital Image Correlation Engine (DICe)" << std::endl;
  std::cout << "** " << VERSION << std::endl;
  std::cout << "** git: " << GITSHA1 << std::endl;
  std::cout << mpi_message << std::endl;
  std::cout << kokkos_message << std::endl;
  std::cout << type_message << std::endl;
  std::cout << "** Copyright 2015 National Technology & Engineering Solutions of Sandia, LLC (NTESS)" << std::endl;
  std::cout << "** Report bugs and feature requests as issues at https://github.com/dicengine/dice" << std::endl << std::endl;
}

/// Initialization function (mpi and kokkos if enabled):
/// \param argc argument count
/// \param argv array of argument chars
DICE_LIB_DLL_EXPORT
void initialize(int argc,
  char *argv[]){
  // initialize mpi
  int proc_rank=0;
#if DICE_MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&proc_rank);
#endif
  // initialize kokkos
#if DICE_KOKKOS
  Kokkos::initialize(argc, argv);
  DEBUG_MSG("Kokkos has been initialized");
#endif
  bool verbose=false;
  for (int_t i = 1; i < argc; ++i) {
    if (strcmp(argv[i],"-v")==0 || strcmp(argv[i],"--verbose")==0)
      verbose=true;
  }
  if(verbose && proc_rank==0)
    print_banner();
}

/// Finalize function (mpi and kokkos if enabled):
DICE_LIB_DLL_EXPORT
void finalize(){
  // finalize mpi
#if DICE_MPI
  (void) MPI_Finalize ();
#endif
  // finalize kokkos
#if DICE_KOKKOS
  Kokkos::finalize();
#endif
}

/// returns true if the data layout is LayoutRight
DICE_LIB_DLL_EXPORT
bool default_is_layout_right(){
#if DICE_KOKKOS
  return Kokkos::Impl::is_same< intensity_dual_view_2d::array_layout ,
      Kokkos::LayoutRight >::value;
#else
  return true; // in all other cases beside CUDA (like serial , it's row-major or layout right)
#endif
}

}// End DICe Namespace
