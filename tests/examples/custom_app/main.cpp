//
// An example used to illustrate calling DICe routines from an external code
//

#include <iostream>

// DICe includes for classes used below:
#include <DICe.h>
#include <DICe_Schema.h>

// See the custom_app example in the tutorial for more information about the code below

using namespace DICe::field_enums;

int main(int argc, char *argv[]) {

  std::cout << "Begin custom_app example\n";
  //
  // STEP 0: initialize threading, etc if it is enabled
  //
  DICe::initialize(argc,argv);
  //
  // STEP 1:
  //
  // Create a DICe::Schema that holds all the correlation parameters, image names, solution results, and
  // provides a number of helpful methods (See class DIC::Schema in the documentation for the full list of methods)
  //
  // This step is only needed once, at the beginning of the analysis (not for multiple images or frames in a video).
  // If later, the parameters should change for the analysis, use the set_params(file_name)
  // or set_params(parameterlist) methods to change the parameters
  //
  // Using the parameters file method to create a schema (See params.xml for the correlation parameters and input.xml for the subset locations, etc.)
  DICe::Schema schema("input.xml","params.xml");
  //
  // There are two different ways to set the parameters in the constructor, using an xml file (used here, see above)
  // or by creating a Teuchos::ParameterList manually and setting the parameters.
  // To use a Teuchos::ParameterList, #include<Teuchos_ParameterList.hpp> and link to library teuchosparameterlist (from Trilinos)
  // If the second method is used, set parameters as follows:
  //
  //     Teuchos::RCP<Teuchos::ParameterList> params = rcp(new Teuchos::ParameterList());
  //     params->set("enable_rotation",false);
  //     params->set("enable_normal_strain",true);
  //     params->set("interpolation_method", DICe::BILINEAR);
  //     ... and so on for the rest of the desired parameters
  //
  // The schema constructor would then be
  //
  // DICe::Schema("ref.tif","def.tif",params);
  //
  // STEP 2:
  //
  // set the reference and deformed images
  //
  schema.set_ref_image("ref.tif");
  schema.set_def_image("def.tif");
  //
  // There are also set methods that take DICe::Images as input arguments or pointers to arrays of intensity values.
  // See DICe::Schema.h for these methods
  schema.print_fields();
  //
  // STEP 3:
  //
  // Run the analysis
  //
  schema.execute_correlation();
  // post process the strain, etc.
  schema.execute_post_processors();
  //
  // STEP 4:
  //
  // Write the output
  schema.write_output("","custom_app_output");
  // Update the prev image and do any other tasks meant for after a frame has been processed
  // If your app is iterating over images, this call should be inside the loop, but at the very end of the loop body
  schema.post_execution_tasks();

  std::cout << "End custom_app example\n";

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // The code below this point either provides extra illustrations of schema methods or is used
  // to test that the custom_app example is building and executing properly in the DICe regression test system

  //
  // Examples of other operations
  //
  // Change the deformed image
  //schema.set_def_image("def2.tif");
  // after this, execute_correlation can be called again on this image without having to re-init the schema
  //
  // Direct access to field values in the schema
  // schema.field_value( global_subset_id, field_name)
  std::cout << "The DISPLACEMENT_X field value for subset 0 is " << schema.local_field_value(0,SUBSET_DISPLACEMENT_X_FS) << std::endl;
  // The field_value() method can be used to set the value as well,
  // for example if you wanted to move subset 0 to a new x-location, the syntax would be
  schema.local_field_value(0,SUBSET_COORDINATES_X_FS) = 150;

  //
  // Test the computed values to make sure this example is working properly
  //
  int errorFlag = 0;
  double errorTol = 0.1;
  // check that 4 subsets were created
  if(schema.local_num_subsets()!=4){
    std::cout << "Error, the number of points is not correct" << std::endl;
    errorFlag++;
  }
  // check that the solution displacements in x are in the vicinity of 0.4 pixels
  for(int i=0;i<schema.local_num_subsets();++i){
    if(std::abs(schema.local_field_value(i,SUBSET_DISPLACEMENT_X_FS)-0.4) > errorTol){
      std::cout << "Error, the displacement solution is not correct" << std::endl;
      errorFlag++;
    }
  }

  DICe::finalize();

  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";
  
  return 0;
}

