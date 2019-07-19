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

#include <cassert>
#include <iostream>

#include <DICe_ImageIO.h>
#include <DICe_Rawi.h>

#if DICE_ENABLE_NETCDF
  #include <DICe_NetCDF.h>
#endif

namespace DICe{
namespace utils{

DICE_LIB_DLL_EXPORT
std::string netcdf_file_name(const char * decorated_netcdf_file){
  std::string netcdf_string(decorated_netcdf_file);
  // trim off the last underscore and the rest
  size_t found = netcdf_string.find("_frame_");
  if(found==std::string::npos){
    return netcdf_string;
  }
  std::string file_name = netcdf_string.substr(0,found);
  // add the .cine extension back
  file_name+=".nc";
  return file_name;
}

DICE_LIB_DLL_EXPORT
std::string cine_file_name(const char * decorated_cine_file){
  std::string cine_string(decorated_cine_file);
  // trim off the last underscore and the rest
  size_t found = cine_string.find_last_of("_");
  std::string file_name = cine_string.substr(0,found);
  // add the .cine extension back
  file_name+=".cine";
  return file_name;
}

DICE_LIB_DLL_EXPORT
int_t netcdf_index(const char * decorated_netcdf_file){
  std::string netcdf_string(decorated_netcdf_file);
  // trim off the last underscore and the rest
  size_t found = netcdf_string.find("_frame_");
  if(found==std::string::npos) return 0;
  size_t found_underscore = netcdf_string.find_last_of("_");
  size_t found_ext = netcdf_string.find(".nc");
  std::string first_num_str = netcdf_string.substr(found_underscore+1,found_ext - found_underscore - 1);
  DEBUG_MSG("netcdf_index(): " << first_num_str);
  return std::strtol(first_num_str.c_str(),NULL,0);
}

DICE_LIB_DLL_EXPORT
void cine_index(const char * decorated_cine_file,
  int_t & start_index,
  int_t & end_index,
  bool & is_avg){
  end_index = -1;
  is_avg = false;
  std::string cine_string(decorated_cine_file);
  // trim off the last underscore and the rest
  size_t found = cine_string.find_last_of("_");
  std::string tail = cine_string.substr(found);
  //DEBUG_MSG("cine_index(): tail: " << tail);
  // determine if this is an average image
  size_t found_avg = tail.find("avg");
  size_t found_ext = tail.find(".cine");
  //DEBUG_MSG("cine_index(): found average at position: " << found_avg);
  if(found_avg==std::string::npos){
    assert(found_ext!=std::string::npos);
    std::string index_str = tail.substr(1,found_ext-1);
    //DEBUG_MSG("cine_index(): index_str: " << index_str);
    start_index = std::strtol(index_str.c_str(),NULL,0);
  }
  else{
    size_t found_dash = tail.find("to");
    std::string first_num_str = tail.substr(found_avg+3,found_dash-found_avg-3);
    std::string second_num_str = tail.substr(found_dash+2,found_ext-found_dash-2);
    //DEBUG_MSG("cine_index(): first num " << first_num_str << " second num " << second_num_str);
    start_index = std::strtol(first_num_str.c_str(),NULL,0);
    end_index = std::strtol(second_num_str.c_str(),NULL,0);
    is_avg = true;
  }
}


DICE_LIB_DLL_EXPORT
Image_File_Type image_file_type(const char * file_name){
  Image_File_Type file_type = NO_SUCH_IMAGE_FILE_TYPE;
  // determine the file type based on the file_name
  const std::string file_str(file_name);
  const std::string rawi(".rawi");
  if(file_str.find(rawi)!=std::string::npos)
    return RAWI;
  const std::string cine(".cine");
  if(file_str.find(cine)!=std::string::npos)
    return CINE;
  const std::string tif("tif");
  const std::string tiff("tiff");
  if(file_str.find(tif)!=std::string::npos||file_str.find(tiff)!=std::string::npos)
    return TIFF;
  const std::string jpg("jpg");
  const std::string jpeg("jpeg");
  if(file_str.find(jpg)!=std::string::npos||file_str.find(jpeg)!=std::string::npos)
    return JPEG;
  const std::string bmp("bmp");
  if(file_str.find(bmp)!=std::string::npos)
    return BMP;
  const std::string png("png");
  if(file_str.find(png)!=std::string::npos)
    return PNG;
  const std::string nc(".nc");
  if(file_str.find(nc)!=std::string::npos)
    return NETCDF;
  return file_type;
}

DICE_LIB_DLL_EXPORT
void read_image_dimensions(const char * file_name,
  int_t & width,
  int_t & height){

  // determine the file type based on the file_name
  Image_File_Type file_type = image_file_type(file_name);

  if(file_type==NO_SUCH_IMAGE_FILE_TYPE){
    std::cerr << "Error, unrecognized image file type for file: " << file_name << "\n";
    throw std::exception();
  }
  if(file_type==RAWI){
    read_rawi_image_dimensions(file_name,width,height);
  }
  else if(file_type==CINE){
    const std::string cine_file = cine_file_name(file_name);
    DEBUG_MSG("read_image_dimensions(): cine file name: " << cine_file);
    width = Image_Reader_Cache::instance().cine_reader(cine_file)->width();
    height = Image_Reader_Cache::instance().cine_reader(cine_file)->height();
    assert(width>0);
    assert(height>0);
  }
#if DICE_ENABLE_NETCDF
  else if(file_type==NETCDF){
    netcdf::NetCDF_Reader netcdf_reader;
    int_t num_time_steps = 0;
    const std::string netcdf_file = netcdf_file_name(file_name);
    DEBUG_MSG("read_image_dimensions(): netcdf file name: " << netcdf_file);
    netcdf_reader.get_image_dimensions(netcdf_file,width,height,num_time_steps);
  }
#endif
  else{
    DEBUG_MSG("read_image_dimensions(): (opencv) file name: " << file_name);
    cv::Mat image = cv::imread(file_name, cv::ImreadModes::IMREAD_GRAYSCALE);
    height = image.rows;
    width = image.cols;
  }
}

DICE_LIB_DLL_EXPORT
cv::Mat read_image(const char * file_name){
  if(image_file_type(file_name)==CINE){
    // get the image dimensions
    int_t width = 0;
    int_t height = 0;
    read_image_dimensions(file_name,width,height);
    // read the cine
    Teuchos::ArrayRCP<intensity_t> intensities(width*height,0.0);
    read_image(file_name,intensities.getRawPtr());
    // pipe the intensities into an opencv Mat
    cv::Mat img(height,width,CV_8UC1,cv::Scalar(0));
    for(int_t y=0;y<height;++y){
      for(int_t x=0;x<width;++x){
        img.at<uchar>(y,x) = intensities[y*width+x];
      }
    }
    return img;
  }else{
    return cv::imread(file_name,cv::IMREAD_GRAYSCALE);
  }
}


DICE_LIB_DLL_EXPORT
void read_image(const char * file_name,
  intensity_t * intensities,
  const bool is_layout_right,
    const bool convert_to_8_bit,
    const bool filter_failed_pixels){
  // determine the file type based on the file_name
  Image_File_Type file_type = image_file_type(file_name);
  if(file_type==NO_SUCH_IMAGE_FILE_TYPE){
    std::cerr << "Error, unrecognized image file type for file: " << file_name << "\n";
    throw std::exception();
  }
  if(file_type==RAWI){
    read_rawi_image(file_name,intensities,is_layout_right);
  }
  else if(file_type==CINE){
    const std::string cine_file = cine_file_name(file_name);
    DEBUG_MSG("read_image(): cine file name: " << cine_file);
    int_t end_index = -1;
    int_t start_index =-1;
    bool is_avg = false;
    cine_index(file_name,start_index,end_index,is_avg);
    DEBUG_MSG("read_image(): start index : " << start_index << " end index " << end_index << " is avg: " << is_avg);
    // get the image dimensions
    Teuchos::RCP<DICe::cine::Cine_Reader> reader = Image_Reader_Cache::instance().cine_reader(cine_file);
    const int_t width = reader->width();
    const int_t height = reader->height();
    Image_Reader_Cache::instance().set_filter_failed_pixels(filter_failed_pixels);
    if(start_index < reader->first_image_number() || start_index > reader->first_image_number() + reader->num_frames()){
      std::cerr << "Error, invalid start index " << start_index <<", less than first frame in cine file " << reader->first_image_number() <<
          " or greater than last frame " << reader->first_image_number() + reader->num_frames() << std::endl;
      throw std::exception();
    }
    if(is_avg && end_index > reader->first_image_number() + reader->num_frames()){
      std::cerr << "Error, invalid end index " << end_index << ", greater than last frame in cine file " <<
          reader->first_image_number() + reader->num_frames() << std::endl;
      throw std::exception();
    }
    if(is_avg){
      reader->get_average_frame(start_index-reader->first_image_number(),end_index-reader->first_image_number(),
        0,0,width,height,intensities,is_layout_right,filter_failed_pixels,convert_to_8_bit);
    }else{
      reader->get_frame(0,0,width,height,intensities,is_layout_right,start_index-reader->first_image_number(),filter_failed_pixels,convert_to_8_bit);
    }
  }
#ifdef DICE_ENABLE_NETCDF
  /// check if the file is a netcdf file
  else if(file_type==NETCDF){
    netcdf::NetCDF_Reader netcdf_reader;
    const std::string netcdf_file = netcdf_file_name(file_name);
    const int_t index = netcdf_index(file_name);
    netcdf_reader.read_netcdf_image(netcdf_file.c_str(),intensities,index,is_layout_right);
  }
#endif
  else{
    // read the image using opencv:
    cv::Mat image;
    image = cv::imread(file_name, cv::ImreadModes::IMREAD_GRAYSCALE);
    const int_t height = image.rows;
    const int_t width = image.cols;
    for(int_t y=0;y<height; ++y) {
      uchar* p = image.ptr(y);
      if(is_layout_right)
        for (int_t x=0; x<width;++x){
          intensities[y*width+x] = *p++;
        }
      else // otherwise assume LayoutLeft
        for (int_t x=0; x<width;++x){
          intensities[x*height+y] = *p++;
        }
    } //for
  } // else
}

DICE_LIB_DLL_EXPORT
void read_image(const char * file_name,
  int_t offset_x,
  int_t offset_y,
  int_t width,
  int_t height,
  intensity_t * intensities,
  const bool is_layout_right,
  const bool convert_to_8_bit,
  const bool filter_failed_pixels){
  // determine the file type based on the file_name
  Image_File_Type file_type = image_file_type(file_name);
  if(file_type==NO_SUCH_IMAGE_FILE_TYPE){
    std::cerr << "Error, unrecognized image file type for file: " << file_name << "\n";
    throw std::exception();
  }
  if(file_type==RAWI){
    std::cerr << "Error, reading only a portion of an image is not supported for rawi, file name: " << file_name << "\n";
    throw std::exception();
  }
  else if(file_type==CINE){
    const std::string cine_file = cine_file_name(file_name);
    DEBUG_MSG("read_image(): cine file name: " << cine_file);
    int_t end_index = -1;
    int_t start_index =-1;
    bool is_avg = false;
    cine_index(file_name,start_index,end_index,is_avg);
    // get the image dimensions
    Teuchos::RCP<DICe::cine::Cine_Reader> reader = Image_Reader_Cache::instance().cine_reader(cine_file);
    Image_Reader_Cache::instance().set_filter_failed_pixels(filter_failed_pixels);
    if(is_avg){
      reader->get_average_frame(start_index-reader->first_image_number(),end_index-reader->first_image_number(),
        offset_x,offset_y,width,height,intensities,is_layout_right,filter_failed_pixels,convert_to_8_bit);
    }else{
      reader->get_frame(offset_x,offset_y,width,height,intensities,is_layout_right,start_index-reader->first_image_number(),filter_failed_pixels,convert_to_8_bit);
    }
  }
#ifdef DICE_ENABLE_NETCDF
  /// check if the file is a netcdf file
    else if(file_type==NETCDF){
      netcdf::NetCDF_Reader netcdf_reader;
      const std::string netcdf_file = netcdf_file_name(file_name);
      const int_t index = netcdf_index(file_name);
      netcdf_reader.read_netcdf_image(netcdf_file.c_str(),offset_x,offset_y,width,height,index,intensities,is_layout_right);
    }
#endif
  else{
    // read the image using opencv:
    cv::Mat image;
    image = cv::imread(file_name, cv::ImreadModes::IMREAD_GRAYSCALE);
    //const int_t height = image.rows;
    //const int_t width = image.cols;
    assert(width+offset_x <= image.cols);
    assert(height+offset_y <= image.rows);
    for (int_t y=offset_y; y<offset_y+height; ++y) {
      uchar* p = image.ptr(y);
      p+=offset_x;
      if(is_layout_right)
        for (int_t x=offset_x; x<offset_x+width;++x){
          intensities[(y-offset_y)*width + x-offset_x] = *p++;
        }
      else // otherwise assume layout left
        for (int_t x=offset_x; x<offset_x+width;++x){
          intensities[(x-offset_x)*height+y-offset_y] = *p++;
        }
    }
  }
}

DICE_LIB_DLL_EXPORT
void write_color_overlap_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * bottom_intensities,
  intensity_t * top_intensities){

  intensity_t bot_max_intensity = -1.0E10;
  intensity_t bot_min_intensity = 1.0E10;
  intensity_t top_max_intensity = -1.0E10;
  intensity_t top_min_intensity = 1.0E10;
  for(int_t i=0; i<width*height; ++i){
    if(bottom_intensities[i] > bot_max_intensity) bot_max_intensity = bottom_intensities[i];
    if(bottom_intensities[i] < bot_min_intensity) bot_min_intensity = bottom_intensities[i];
    if(top_intensities[i] > top_max_intensity) top_max_intensity = top_intensities[i];
    if(top_intensities[i] < top_min_intensity) top_min_intensity = top_intensities[i];
  }
  intensity_t bot_fac = 1.0;
  intensity_t top_fac = 1.0;
  if((bot_max_intensity - bot_min_intensity) != 0.0)
    bot_fac = 0.5*255.0 / (bot_max_intensity - bot_min_intensity);
  if((top_max_intensity - top_min_intensity) != 0.0)
    top_fac = 0.5*255.0 / (top_max_intensity - top_min_intensity);

  cv::Mat out_img(height,width,CV_8UC3,cv::Scalar(0,0,0));
  for (int_t y=0; y<height; ++y) {
      for (int_t x=0; x<width;++x){
        out_img.at<cv::Vec3b>(y,x)[1] = std::floor((bottom_intensities[y*width + x]-bot_min_intensity)*bot_fac);
        out_img.at<cv::Vec3b>(y,x)[2] = std::floor((top_intensities[y*width + x]-top_min_intensity)*top_fac);
      }
  }
  cv::imwrite(file_name,out_img);
}


DICE_LIB_DLL_EXPORT
void write_image(const char * file_name,
  const int_t width,
  const int_t height,
  intensity_t * intensities,
  const bool is_layout_right,
  const bool scale_to_8_bit){
  // determine the file type based on the file_name
  Image_File_Type file_type = image_file_type(file_name);
  if(file_type==NO_SUCH_IMAGE_FILE_TYPE){
    std::cerr << "Error, unrecognized image file type for file: " << file_name << "\n";
    throw std::exception();
  }
  // rawi files are not scaled to the 8 bit range, because the file type holds double precision values
  if(file_type==RAWI){
    write_rawi_image(file_name,width,height,intensities,is_layout_right);
  }
  else{
    // rip through the intensity values and determine if they need to be scaled to 8-bit range (0-255):
    // negative values are shifted to start at zero so all values will be positive
    intensity_t max_intensity = -1.0E10;
    intensity_t min_intensity = 1.0E10;
    for(int_t i=0; i<width*height; ++i){
      if(intensities[i] > max_intensity) max_intensity = intensities[i];
      if(intensities[i] < min_intensity) min_intensity = intensities[i];
    }
    intensity_t fac = 1.0;
    if((max_intensity - min_intensity) != 0.0)
      fac = 255.0 / (max_intensity - min_intensity);
    cv::Mat out_img(height,width,CV_8UC1,cv::Scalar(0));
    for (int_t y=0; y<height; ++y) {
      if(is_layout_right)
        for (int_t x=0; x<width;++x){
          if(scale_to_8_bit)
            out_img.at<uchar>(y,x) = std::floor((intensities[y*width + x]-min_intensity)*fac);
          else
            out_img.at<uchar>(y,x) = std::floor(intensities[y*width + x]);
        }
      else
        for (int_t x=0; x<width;++x){
          if(scale_to_8_bit)
            out_img.at<uchar>(y,x) = std::floor((intensities[x*height+y]-min_intensity)*fac);
          else
            out_img.at<uchar>(y,x) = std::floor(intensities[x*height+y]);
        }
    }
    cv::imwrite(file_name,out_img);
  }
}

Teuchos::RCP<DICe::cine::Cine_Reader>
Image_Reader_Cache::cine_reader(const std::string & id){
  if(cine_reader_map_.find(id)==cine_reader_map_.end()){
    Teuchos::RCP<DICe::cine::Cine_Reader> cine_reader = Teuchos::rcp(new DICe::cine::Cine_Reader(id,NULL,filter_failed_pixels_));
    cine_reader_map_.insert(std::pair<std::string,Teuchos::RCP<DICe::cine::Cine_Reader> >(id,cine_reader));
    return cine_reader;
  }
  else
    return cine_reader_map_.find(id)->second;
}

} // end namespace utils
} // end namespace DICe
