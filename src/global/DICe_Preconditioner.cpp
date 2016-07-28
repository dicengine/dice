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

#include <DICe_Preconditioner.h>

namespace DICe {

#ifdef DICE_TPETRA
#else

Teuchos::RCP<Teuchos::ParameterList>
Preconditioner_Factory::parameter_list_for_ifpack() const{
  // The name of the type of preconditioner to use.
  const std::string precondType("ILU");
  // Ifpack expects double-precision arguments here.
  const double fillLevel = 2.0;
  const double dropTol = 0.0;
  const double absThreshold = 0.1;
//  const bool verbose = false;
//  const bool debug = false;
  Teuchos::RCP<Teuchos::ParameterList> pl = Teuchos::parameterList ("Preconditioner");
  pl->set ("Ifpack::Preconditioner", precondType);
  Teuchos::ParameterList precParams ("Ifpack");
  precParams.set ("fact: ilut level-of-fill", fillLevel);
  precParams.set ("fact: drop tolerance", dropTol);
  precParams.set ("fact: absolute threshold", absThreshold);
  pl->set ("Ifpack", precParams);
  return pl;
}

// Compute and return an Ifpack preconditioner.
Teuchos::RCP<Ifpack_Preconditioner>
Preconditioner_Factory::create (Teuchos::RCP<matrix_type> A,
          const Teuchos::RCP<Teuchos::ParameterList> plist) const
{
  DEBUG_MSG("Preconditioner_Factory(): creating ILUT preconditioner");
  DEBUG_MSG("Preconditioner_Factory(): configuring");
  Ifpack factory;

  // Get the preconditioner type.
  const std::string precName =
      plist->get<std::string> ("Ifpack::Preconditioner");

  // Set up the preconditioner of that type.
  int_t overlap_level = 1;
  Teuchos::RCP<Ifpack_Preconditioner> prec = Teuchos::rcp( factory.Create(precName, A.get(), overlap_level) );
  //prec = factory.Create(precName, A.get(),overlap_level);
  Teuchos::ParameterList ifpackParams;
  if (plist->isSublist ("Ifpack"))
    ifpackParams = plist->sublist ("Ifpack");
  else
    ifpackParams.setName ("Ifpack");
  prec->SetParameters (ifpackParams);

  DEBUG_MSG("Preconditioner_Factory(): initializing");
  prec->Initialize();
  DEBUG_MSG("Preconditioner_Factory(): computing");
  prec->Compute();

  return prec;
}
#endif

}// End DICe Namespace
