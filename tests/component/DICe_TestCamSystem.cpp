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

#include <DICe_CamSystem.h>

using namespace DICe;

  int main(int argc, char *argv[]) {

    bool all_passed = true;

    DICe::initialize(argc, argv);
    
    try {
      
      bool allFields = true;
      //temp for testing
      std::string cal_file_name;
      std::string cal_file_dir;
      std::string cal_file;
      std::string save_file;
      std::vector<std::string> file_names;
      file_names.push_back("cal_a.xml");
      file_names.push_back("DICe_cal_a.xml");
      file_names.push_back("cal_a.txt");
      file_names.push_back("cal_a_with_R.txt");
      file_names.push_back("cal_a_with_transform.txt");
      for (int_t i = 0; i < 5; i++)
      {
        cal_file_dir = ".\\cal\\";
        cal_file = cal_file_dir + file_names[i] + "";
        Teuchos::RCP<CamSystem> cam_sys = Teuchos::rcp(new CamSystem());
        cam_sys->read_calibration_parameters(cal_file);

        file_names[i].pop_back();
        file_names[i].pop_back();
        file_names[i].pop_back();
        file_names[i].push_back('x');
        file_names[i].push_back('m');
        file_names[i].push_back('l');

        if (allFields)
          save_file = cal_file_dir + "DICe_allFields_" + file_names[i];
        else
          save_file = cal_file_dir + "DICe_" + file_names[i];

        //cam_sys->write_calibration_file(save_file, allFields);
      }

    }
    catch (std::exception & e) {
      std::cout << e.what() << std::endl;
      all_passed = false;
    }
    
    DICe::finalize();
 
    if (!all_passed)
      std::cout << "End Result: TEST FAILED\n";
    else
      std::cout << "End Result: TEST PASSED\n";

    return 0;

  }

