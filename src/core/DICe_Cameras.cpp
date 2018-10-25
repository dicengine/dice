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

#include <DICe_Cameras.h>
#include <DICe_Parser.h>
#include <fstream>
#include <Teuchos_XMLParameterListHelpers.hpp>



namespace DICe {

	DICE_LIB_DLL_EXPORT

		void
		Cameras::load_calibration_parameters(const std::string & paramFileName) {
		DEBUG_MSG("Cameras::load_calibration_parameters(): begin");
		bool validXML = true;

		//clear the intrinsic and extrinsic vectors

		// intrinsic parameters from both vic3d and the generic text reader are the same and in this order
		// **Q what order from openCV?
		// cx cy fx fy fs k0 k1 k2
		intrinsics_.clear();
		for (int_t i = 0; i < MAX_CAM_INTRINSIC_PARMS; ++i)
			intrinsics_.push_back(std::vector<scalar_t>(18, 0.0));

		extrinsics_.clear();
		for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARMS; ++i)
			extrinsics_.push_back(std::vector<scalar_t>(6, 0.0));

		user_trans_4x4_params_.clear();
		for (int_t i = 0; i < 4; ++i) {
			user_trans_4x4_params_.push_back(std::vector<scalar_t>(4, 0.0));
			user_trans_4x4_params_[i][i] = 1.0;
		}

		user_trans_6_params_.clear();
		for (int_t i = 0; i < 7; ++i) {
			user_trans_6_params_.push_back(0.0);
		}

		int_t camera_index = 0;
		const std::string xml("xml");
		const std::string txt("txt");

		num_cams_ = 0;
		bool has_6_transform = false;
		bool has_4x4_transform = false;
		bool has_extrinsic = false;

		int proc_rank = 0;
		if (proc_rank == 0)
			DEBUG_MSG("Cameras::load_calibration_parameters():Trying to read file with Teuchos XML parser: " << paramFileName);
		Teuchos::RCP<Teuchos::ParameterList> stringParams = Teuchos::rcp(new Teuchos::ParameterList());
		Teuchos::Ptr<Teuchos::ParameterList> stringParamsPtr(stringParams.get());
		try {
			Teuchos::updateParametersFromXmlFile(paramFileName, stringParamsPtr);
		}
		catch (std::exception & e) {
			//std::cout << e.what() << std::endl;
			DEBUG_MSG("Cameras::load_calibration_parameters():Invalid XML file: " << paramFileName);
			validXML = false;
		}

		if (validXML)  //file read into parameters
			DEBUG_MSG("Cameras::load_calibration_parameters():Valid XML file for Teuchos parser");

		else {

			DEBUG_MSG("Cameras::load_calibration_parameters(): Parsing calibration parameters from non Teuchos file: " << paramFileName);
			std::fstream dataFile(paramFileName.c_str(), std::ios_base::in);
			TEUCHOS_TEST_FOR_EXCEPTION(!dataFile.good(), std::runtime_error,
				"Error, the calibration file does not exist or is corrupt: " << paramFileName);

			//check the file extension for xml or txt
			if (paramFileName.find(xml) != std::string::npos) {
				DEBUG_MSG("Cameras::load_calibration_parameters(): assuming calibration file is vic3D xml format");
				// cal.xml file can't be ready by Teuchos parser because it has a !DOCTYPE
				// have to manually read the file here, lots of assumptions in how the file is formatted
				// camera orientation for each camera in vic3d is in terms of the world to camera
				// orientation and the order of variables is alpha beta gamma tx ty tz (the Cardan Bryant angles + translations)
				// read each line of the file
				sys_type_ = VIC3D;
				const int numIntParms = 8;
				const int numExtParms = 6;
				const int parmOrderInt[numIntParms] = { CX, CY, FX, FY, FS, K1, K2, K3 };
				const int parmOrderExt[numExtParms] = { ALPHA, BETA, GAMMA, TX, TY, TZ };

				while (!dataFile.eof()) {
					std::vector<std::string> tokens = tokenize_line(dataFile, " \t<>\"");
					if (tokens.size() == 0) continue;
					if (tokens[0] != "CAMERA") continue;
					num_cams_++;
					camera_index = (int_t)strtod(tokens[2].c_str(), NULL);
					DEBUG_MSG("Cameras::load_calibration_parameters(): reading camera " << camera_index << " parameters");
					assert(camera_index < 10);
					assert(tokens.size() > 18);
					//Store the intrinsic parameters
					for (int i = 0; i < numIntParms; ++i)
						intrinsics_[camera_index][parmOrderInt[i]] = strtod(tokens[i + 3].c_str(), NULL);
					intrinsics_[camera_index][LD_MODEL] = K1R1_K2R2_K3R3;

					//Store the extrinsic parameters
					assert(tokens[11].c_str() == "orientation");
					for (int i = 0; i < numExtParms; ++i)
						extrinsics_[camera_index][parmOrderExt[i]] = strtod(tokens[i + 12].c_str(), NULL);
				} // end file read
			}

			else if (paramFileName.find(txt) != std::string::npos) {
				DEBUG_MSG("Cameras::load_calibration_parameters(): calibration file is generic txt format");
				//may want to modify this file format to allow more than 2 cameras in the future
				sys_type_ = GENERIC_SYSTEM;
				num_cams_ = 2;
				const int numIntParms = 8;
				const int numExtParms = 6;
				const int parmOrderInt[numIntParms] = { CX, CY, FX, FY, FS, K1, K2, K3 };
				const int parmOrderExt[numExtParms] = { ALPHA, BETA, GAMMA, TX, TY, TZ };
				const int_t num_values_base = 22;
				const int_t num_values_expected_with_R = 28;
				//const int_t num_values_with_custom_transform = 28;
				int_t total_num_values = 0;

				//Run through the file to determine the format
				while (!dataFile.eof())
				{
					std::vector<std::string>tokens = tokenize_line(dataFile, " \t<>");
					if (tokens.size() == 0) continue;
					if (tokens[0] == "#") continue;
					if (tokens[0] == "TRANSFORM") {
						has_6_transform = true; //file has a 6 parm user transformation
						has_extrinsic = true; //file has extrinsic values
						continue;
					}
					total_num_values++;
				}

				if (total_num_values == num_values_base) {
					has_extrinsic = true; //file has extrinsic values
				}
				has_4x4_transform = !has_extrinsic;

				TEUCHOS_TEST_FOR_EXCEPTION(total_num_values != num_values_base && total_num_values != num_values_expected_with_R, std::runtime_error,
					"Error, wrong number of parameters in calibration file: " << paramFileName);

				// return to start of file:
				dataFile.clear();
				dataFile.seekg(0, std::ios::beg);

				//text file has cam0 aligned with the world coordinates so all extrinsics are 0 for cam 0
				for (int_t i = 0; i < MAX_CAM_EXTRINSIC_PARMS; ++i)
					extrinsics_[0][i] = 0.0;
				intrinsics_[0][LD_MODEL] = K1R1_K2R2_K3R3;
				intrinsics_[1][LD_MODEL] = K1R1_K2R2_K3R3;

				int_t camera_index = 0;
				int_t num_values = 0;
				int_t i = 0;
				int_t j = 0;
				while (!dataFile.eof())
				{
					std::vector<std::string> tokens = tokenize_line(dataFile, " \t<>");
					if (tokens.size() == 0) continue;
					if (tokens[0] == "#") continue;
					if (tokens[0] == "TRANSFORM") continue;
					if (tokens.size() > 1) {
						assert(tokens[1] == "#"); // only one entry per line plus comments
					}
					if (num_values < 8)
						intrinsics_[0][parmOrderInt[num_values]] = strtod(tokens[0].c_str(), NULL);
					else if (num_values < 16)
						intrinsics_[1][parmOrderInt[num_values - 8]] = strtod(tokens[0].c_str(), NULL);
					else if (num_values < 22 && has_extrinsic)
						extrinsics_[1][parmOrderInt[num_values - 16]] = strtod(tokens[0].c_str(), NULL);
					else if (num_values < 25 && !has_extrinsic) {
						i = int((num_values - 16) / 3);
						j = (num_values - 16) % 3;
						user_trans_4x4_params_[i][j] = strtod(tokens[0].c_str(), NULL);
					}
					else if (num_values < 28 && !has_extrinsic)
						user_trans_4x4_params_[num_values - 25][3] = strtod(tokens[0].c_str(), NULL);
					else if (num_values < 28 && has_6_transform)
						user_trans_6_params_[num_values - 22] = strtod(tokens[0].c_str(), NULL);
					else {
						TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "");
					}
					num_values++;
				}

			}
			else {
				TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
					"Error, unrecognized calibration parameters file format: " << paramFileName);
			}


		}
		if (num_cams_ != 0) {
			DEBUG_MSG("************************************************************************");
			DEBUG_MSG("System type: " << systemType3DStrings[sys_type_]);
			DEBUG_MSG("Number of Cams: " << num_cams_);
			DEBUG_MSG(" ");

			for (int i = 0; i < 10; i++) {
				if (intrinsics_[i][FX] > 1) {
					DEBUG_MSG("******************* CAMERA: " << i << " ******************************");
					for (int j = 0; j < LD_MODEL; j++)
						DEBUG_MSG("Cameras::load_calibration_parameters(): " << camIntrinsicParmsStrings[j] << ": " << intrinsics_[i][j]);
					int j = LD_MODEL;
					DEBUG_MSG("Cameras::load_calibration_parameters(): " << camIntrinsicParmsStrings[j] << ": " << lensDistortionModelStrings[(int)intrinsics_[i][j]]);
					for (int j = 0; j < MAX_CAM_EXTRINSIC_PARMS; j++)
						DEBUG_MSG("Cameras::load_calibration_parameters(): " << camExtrinsicParmsStrings[j] << ": " << extrinsics_[i][j]);
					DEBUG_MSG(" ");
				}
			}
			if (has_4x4_transform) {
				DEBUG_MSG("Cameras::load_calibration_parameters(): 4x4 user transformation");
				for (int_t i = 0; i < 4; ++i)
					DEBUG_MSG("Cameras::load_calibration_parameters(): " << user_trans_4x4_params_[i][0] <<
						" " << user_trans_4x4_params_[i][1] << " " << user_trans_4x4_params_[i][2] << " " << user_trans_4x4_params_[i][3]);
			}

			if (has_6_transform) {
				DEBUG_MSG("Cameras::load_calibration_parameters(): 6 parameter user transformation");
				DEBUG_MSG("Cameras::load_calibration_parameters(): " << user_trans_6_params_[0] <<
					" " << user_trans_6_params_[1] << " " << user_trans_6_params_[2] << " " << user_trans_6_params_[3] <<
					" " << user_trans_6_params_[4] << " " << user_trans_6_params_[5]);
			}

		}
		DEBUG_MSG("Cameras::load_calibration_parameters(): end");
	}
}// End DICe Namespace
