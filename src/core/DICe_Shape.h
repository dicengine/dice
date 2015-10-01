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

#ifndef DICE_SHAPE_H
#define DICE_SHAPE_H

#include <DICe.h>
#include <DICe_Image.h>

#include <Teuchos_ArrayRCP.hpp>

#include <set>
#include <cassert>

namespace DICe {

/// \class DICe::Shape
/// \brief Generic class for defining regions in an image
///
/// For conformal subsets, the boundary is defined by an arbitrarily complex shape
/// built up of DICe::Shapes. Depending on the derived class implementation, a shape holds
/// nothing more than geometry information such as the coordinates of the vertices (for a
/// polygon). There are only a couple basic methods for a shape used to determine
/// if a particular pixel is inside or outside a shape. These methods are used in the construction
/// of a conformal subset.

class DICE_LIB_DLL_EXPORT
Shape {
public:
  Shape(){};
  virtual ~Shape(){};

  /// \brief Returns a set of the coordinates of all pixels interior to this shape.
  /// NOTE: The pair is (y,x) not (x,y) so that the ordering in the set will match with loops over y then x
  /// \param deformation Optional mapping to the deformed shape, otherwise reference map is used
  /// \param cx Optional x centroid of the map
  /// \param cy Optional y centroid of the map
  /// \param skin_factor Optional padding added to the outside of the shape to make it larger or smaller
  virtual std::set<std::pair<int_t,int_t> > get_owned_pixels(Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null,
    const int_t cx=0,
    const int_t cy=0,
    const scalar_t skin_factor=1.0)const{
    assert(false && "  DICe ERROR: Base class implementation of this method should not be called.");
    std::set<std::pair<int_t,int_t> > nullSet;
    return nullSet;
  }

  /// \brief Method used to turn pixels off that fall inside the shape.
  /// Mostly called in the construction of a conformal subset to turn off interior regions to the subset.
  /// \param pixel_flags [out] An array of bools true means the pixel is still active false means that
  /// the pixel falls inside the shape and deactivated (should be of same size as x_coords and y_coords)
  /// \param x_coords Array of local x-coordinates for each pixel in reference to the origin_x and origin_y param (should be of same size as pixel_flags and y_coords)
  /// \param y_coords Array of local y-coordinates for each pixel in reference to the origin_x and origin_y param (should be of same size as pixel_flags and x_coords)
  /// \param origin_x global image x-coordinate of origin for the x_coords and y_coords arguments above
  /// \param origin_y global image y-coordinate of origin for the x_coords and y_coords arguments above
  // TODO This is a little awkward and could be done more elegantly
  virtual void deactivate_pixels(Teuchos::ArrayRCP<bool> & pixel_flags,
    Teuchos::ArrayRCP<int_t> x_coords,
    Teuchos::ArrayRCP<int_t> y_coords,
    const int_t origin_x,
    const int_t origin_y) const{
    assert(false && "  DICe ERROR: Base class implementation of this method should not be called.");
  }

  /// \brief For the given image argument, output a tif image with the shape superposed on the image.
  /// Only enabled if boost is enabled, otherwise this is a no-op.
  /// \param layer_0_image The underlying image to draw the shape on
  /// \param deformation An optional vector of displacement, rotation and stretch values to draw the shape in the deformed state
  /// \param cx x-coordiante of the origin of the deformation mapping
  /// \param cy y-coordinate of the origin of the deformation mapping
  virtual void draw(Teuchos::RCP<Image> & layer_0_image,
    Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null,
    const int_t cx=0,
    const int_t cy=0)const{
    assert(false && "  DICe ERROR: Base class implementation of this method should not be called.");
  }
};

///
/// \class DICe::Polygon
/// \brief A straight sided DICe::Shape made of an arbitrary number of sides.
///
/// NOTE: Vertices must be listed in order as the boundary of the
/// polygon is traversed (clockwise or couterclockwise), otherwise the
/// polygon will have folds.

// TODO: add a check for folds
// TODO: add check for repeated vertices

/// NOTE: Coordinates are always in image coordinates, i.e. (0,0) is the upper left corner,
/// x+ points to the right, y+ points down

class DICE_LIB_DLL_EXPORT
Polygon: public Shape {
public:
  /// \brief Constructor that takes the vertices as arguements
  /// \param coords_x A vector of integer valued global x-coordinates for the polygon vertices
  /// \param coords_y A vector of integer valued global y-coordinates for the polygon vertices
  Polygon(std::vector<int_t> & coords_x,
    std::vector<int_t> & coords_y);

  virtual ~Polygon(){};

  /// See base class documentation
  // NOTE: The pair is (y,x) not (x,y) so that the ordering in the set will match loops over y then x
  virtual std::set<std::pair<int_t,int_t> > get_owned_pixels(Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null,
    const int_t cx=0,
    const int_t cy=0,
    const scalar_t skin_factor=1.0)const;

  /// See base class documentation
  virtual void deactivate_pixels(Teuchos::ArrayRCP<bool> & pixel_flags,
    Teuchos::ArrayRCP<int_t> x_coords,
    Teuchos::ArrayRCP<int_t> y_coords,
    const int_t origin_x,
    const int_t origin_y) const;

  /// See base class documentation
  virtual void draw(Teuchos::RCP<Image> & layer_0_image,
    Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null,
    const int_t cx=0,
    const int_t cy=0)const;

private:
  /// vector storing the integer vertex global x-coordinates
  std::vector<int_t> vertex_coordinates_x_;
  /// vector storing the integer vertex global y-coordinates
  std::vector<int_t> vertex_coordinates_y_;
  /// Number of vertices
  int_t num_vertices_;
  /// Minimum x global coordinate of all the vertices
  int_t min_x_;
  /// Maximum x global coordinate of all the vertices
  int_t max_x_;
  /// Minimum y global coordinate of all the vertices
  int_t min_y_;
  /// Maximum y global coordinate of all the vertices
  int_t max_y_;
};

/// \brief Return the angle between two vectors on a plane
///  The angle is from vector 1 to vector 2, positive anticlockwise
///  The result is between -pi -> pi
/// \param x1 run of vector 1
/// \param y1 rise of vector 1
/// \param x2 run of vector 2
/// \param y2 rise of vector 2
DICE_LIB_DLL_EXPORT
const scalar_t angle_2d(const scalar_t & x1,
  const scalar_t & y1,
  const scalar_t & x2,
  const scalar_t & y2);


///
/// \class DICe::Circle
/// \brief A circle is a DICe::Shape defined by a centroid and radius.

class DICE_LIB_DLL_EXPORT
Circle: public Shape {
public:
  /// \brief Constructor
  /// \param centroid_x integer valued global x-coordinate for the center of the circle
  /// \param centroid_y integer valued global y-coordinate for the center of the circle
  /// \param radius Radius of the circle
  Circle(const int_t centroid_x,
    const int_t centroid_y,
    const scalar_t & radius);

  virtual ~Circle(){};

  /// See base class documentation
  virtual std::set<std::pair<int_t,int_t> > get_owned_pixels(Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null,
    const int_t cx=0,
    const int_t cy=0,
    const scalar_t skin_factor=1.0)const;

  /// See base class documentation
  virtual void deactivate_pixels(Teuchos::ArrayRCP<bool> & pixel_flags,
    Teuchos::ArrayRCP<int_t> x_coords,
    Teuchos::ArrayRCP<int_t> y_coords,
    const int_t origin_x,
    const int_t origin_y) const;

  /// See base class documentation
  virtual void draw(Teuchos::RCP<Image> & layer_0_image,
    Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null)const;

private:
  /// Center of the circle global x-coordinate
  int_t centroid_x_;
  /// Center of the circle global y-coordinate
  int_t centroid_y_;
  /// Radius of the circle
  scalar_t radius_;
  /// Radius of the circle squared
  scalar_t radius2_;
  /// Minimum x global coordinate of the circle
  int_t min_x_;
  /// Maximum x global coordinate of the circle
  int_t max_x_;
  /// Minimum y global coordinate of the circle
  int_t min_y_;
  /// Maximum y global coordinate of the circle
  int_t max_y_;
};


///
/// \class DICe::Rectangle
/// \brief A rectangle is a DICe::Shape defined by a centroid and an x and y size.
class DICE_LIB_DLL_EXPORT
Rectangle: public Shape {
public:
  /// \brief Constructor
  /// \param centroid_x integer valued global x-coordinate for the center
  /// \param centroid_y integer valued global y-coordinate for the center
  /// \param width integer valued width of the rectangle
  /// \param height integer valued height of the rectangle
  Rectangle(const int_t centroid_x,
    const int_t centroid_y,
    const int_t width,
    const int_t height);

  virtual ~Rectangle(){};

  /// See base class documentation
  virtual std::set<std::pair<int_t,int_t> > get_owned_pixels(Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null,
    const int_t cx=0,
    const int_t cy=0,
    const scalar_t skin_factor=1.0)const;

  /// See base class documentation
  virtual void deactivate_pixels(Teuchos::ArrayRCP<bool> & pixel_flags,
    Teuchos::ArrayRCP<int_t> x_coords,
    Teuchos::ArrayRCP<int_t> y_coords,
    const int_t origin_x,
    const int_t origin_y) const;

  /// See base class documentation
  virtual void draw(Teuchos::RCP<Image> & layer_0_image,
    Teuchos::RCP<const std::vector<scalar_t> > deformation=Teuchos::null)const;

private:
  /// Center global x-coordinate
  int_t centroid_x_;
  /// Center global y-coordinate
  int_t centroid_y_;
  /// Width of the rectangle
  int_t width_;
  /// Height of the rectangle
  int_t height_;
  /// Origin is the upper left corner (needed if the rectangle is of an even dimension)
  int_t origin_x_;
  /// Origin is the upper left corner
  int_t origin_y_;
};

/// A vector that stores a collection of pointers to shapes, used as a way to associate shapes into a larger object.
typedef std::vector<Teuchos::RCP<Shape> > multi_shape;

/// \class DICe::Conformal_Area_Def
/// \brief A simple container for geometry information defining the boundary of a DICe::Subset.
///
/// There are three parts to a Conformal_Area_Def. The boundary defines the outer boundary of
/// the region of interest for a subset, the excluded area defines a region interior to the boundary
/// that is deactivated initially (possibly becuase it begins obstructed in the image sequence) and
/// the obstructed area which defines fixed or moving regions in the image in which the
/// pixels should be deactivated. Deactivated pixels do not contribute to the correlation criteria or
/// the optimization process.
class DICE_LIB_DLL_EXPORT
Conformal_Area_Def {
public:
  Conformal_Area_Def(){has_excluded_area_ = false;has_boundary_ = false;has_obstructed_area_=false;}

  ~Conformal_Area_Def(){};

  /// \brief Constructor that takes only one argument, the boundary of the subset
  /// \param boundary A multi_shape the defines the outer perimeter of the subset
  Conformal_Area_Def(const multi_shape & boundary){
    assert(boundary.size()>0);
    boundary_ = boundary;
    has_excluded_area_ = false;
    has_boundary_ = true;
    has_obstructed_area_ = false;
  }

  /// \brief Constructor to define both the outer boundary and an interior excluded region
  /// \param boundary A multi_shape that defines the outer perimeter ofd the susbet
  /// \param excluded_area An additional multi_shape that defines an internal region that should be deactivated
  Conformal_Area_Def(const multi_shape & boundary,
    const multi_shape & excluded_area){
    assert(boundary.size()>0);
    boundary_ = boundary;
    excluded_area_ = excluded_area;
    has_excluded_area_ = excluded_area.size()>0;
    has_boundary_ = true;
    has_obstructed_area_ = false;
  }

  /// \brief Constructor that defines the outer boundary, an excluded interior region and obstucted areas
  /// \param boundary A multi_shape that defines the outer perimeter ofd the susbet
  /// \param excluded_area An additional multi_shape that defines an internal region that should be deactivated
  /// \param obstructed_area A third multi_shape that defines regions that are obstructed
  Conformal_Area_Def(const multi_shape & boundary,
    const multi_shape & excluded_area,
    const multi_shape & obstructed_area){
    assert(boundary.size()>0);
    boundary_ = boundary;
    excluded_area_ = excluded_area;
    obstructed_area_ = obstructed_area;
    has_excluded_area_ = excluded_area.size()>0;
    has_boundary_ = true;
    has_obstructed_area_ = obstructed_area.size()>0;
  }

  /// Returns true if the boundary has been defined (should be true for a valid Conformal_Area_Def)
  const bool has_boundary()const{
    return has_boundary_;
  }

  /// Returns true if the Conformal_Area_Def has an internal area initially inactive
  const bool has_excluded_area()const{
    return has_excluded_area_;
  }

  /// Returns true if the Conformal_Area_Def has obstructed areas defined
  const bool has_obstructed_area()const{
    return has_obstructed_area_;
  }

  /// Returns a pointer to the multi_shape that defines the boundary
  const multi_shape * boundary()const{
    return &boundary_;
  }

  /// Returns a pointer to the multi_shape that defines the excluded area
  const multi_shape * excluded_area()const{
    assert(has_excluded_area_);
    return &excluded_area_;
  }

  /// Returns a pointer to the multi_shape that defines the obstructed area
  const multi_shape * obstructed_area()const{
    assert(has_obstructed_area_);
    return &obstructed_area_;
  }

private:
  /// An excluded are has been defined
  bool has_excluded_area_;
  /// The boundary has been defined
  bool has_boundary_;
  /// An obstructed area has been defined
  bool has_obstructed_area_;
  /// Defines the boundary
  multi_shape boundary_;
  /// Defines the excluded area
  multi_shape excluded_area_;
  /// Defines the obstructed area
  multi_shape obstructed_area_;
};

}// End DICe Namespace

#endif
