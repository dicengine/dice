//
// An example used to illustrate calling DICe routines from an external code
//

#include <iostream>

// DICe includes for classes used below:
#include <DICe.h>
#include <DIce_Schema.h>

// See the custom_app example in the tutorial for more information about the code below

int main(int argc, char *argv[]) {

  std::cout << "Begin custom_app example\n";
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
  // Using the parameters file method to create a schema (See params.xml for the parameters)
  DICe::Schema schema("ref.tif","def.tif","params.xml");
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
  // There are also constructors that take DICe::Images as input arguments or pointers to arrays of intensity values.
  // See DICe::Schema.h for these constructors

  //
  // STEP 2:
  //
  // Initialize the Schema in terms of number of subsets and their locations.
  //
  schema.initialize("input.xml");
  //
  // A simple alternative way to initialize the schema is based on the step_size and subset_size.
  //
  // const int step_size_x = 20; // pixels
  // const int step_size_y = 20; // pixels
  // schema.initialize(step_size_x,step_size_y,subset_size);

  schema.print_fields();

  //
  // STEP 3:
  //
  // Run the analysis
  //
  schema.execute_correlation();

  //
  // STEP 4:
  //
  // Write the output
  schema.write_output("custom_app_output.txt");

  std::cout << "End custom_app example\n";

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // The code below this point either provides extra illustrations of schema methods or is used
  // to test that the custom_app example is building and executing properly in the DICe regression test system

  //
  // Examples of other operations
  //
  // Change the deformed image
  schema.set_def_image("def2.tif");
  // after this, execute_correlation can be called again on this image without having to re-init the schema
  //
  // Direct access to field values in the schema
  // schema.field_value( global_subset_id, field_name)
  std::cout << "The DISPLACEMENT_X field value for subset 0 is " << schema.field_value(0,DICe::DISPLACEMENT_X) << std::endl;
  // The field_value() method can be used to set the value as well,
  // for example if you wanted to move subset 0 to a new x-location, the syntax would be
  schema.field_value(0,DICe::COORDINATE_X) = 150;

  //
  // Test the computed values to make sure this example is working properly
  //
  int errorFlag = 0;
  double errorTol = 0.1;
  // check that 4 subsets were created
  if(schema.data_num_points()!=4){
    std::cout << "Error, the number of points is not correct" << std::endl;
    errorFlag++;
  }
  // check that the solution displacements in x are in the vicinity of 0.4 pixels
  for(int i=0;i<schema.data_num_points();++i){
    if(std::abs(schema.field_value(i,DICe::DISPLACEMENT_X)-0.4) > errorTol){
      std::cout << "Error, the displacement solution is not correct" << std::endl;
      errorFlag++;
    }
  }
  if (errorFlag != 0)
    std::cout << "End Result: TEST FAILED\n";
  else
    std::cout << "End Result: TEST PASSED\n";
  
  return 0;
}

