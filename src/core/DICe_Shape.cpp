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

#include <DICe.h>
#include <DICe_Shape.h>

#include <cassert>

namespace DICe {

Polygon::Polygon(std::vector<int_t> & coords_x,
  std::vector<int_t> & coords_y):
  vertex_coordinates_x_(coords_x),
  vertex_coordinates_y_(coords_y)
{
  assert(vertex_coordinates_x_.size()==vertex_coordinates_y_.size()
    && "  DICe ERROR: The vertex coordinate vectors for x and y must be the same size.");
  assert(vertex_coordinates_x_.size() >= 3
    && "  DICe ERROR: vertex coordinates vectors must have at least three points.");

  // note num_vertices is of size one less than the length of
  // the vertex coordinates vectors, this is because the zeroeth entry
  // gets tacked on the end of the coordinates vectors to close the
  // polygon (see methods like draw() to see why this is done)
  num_vertices_ = vertex_coordinates_x_.size();

  // add the first vertex to the end of the list to compete the enclosure
  vertex_coordinates_x_.push_back(coords_x[0]);
  vertex_coordinates_y_.push_back(coords_y[0]);

  // determine the extents of the polygon
  int_t min_x=vertex_coordinates_x_[0];
  int_t min_y=vertex_coordinates_y_[0];
  int_t max_x=vertex_coordinates_x_[0];
  int_t max_y=vertex_coordinates_y_[0];
  for(int_t i=0;i<num_vertices_;++i){
    if(vertex_coordinates_x_[i] < min_x) min_x = vertex_coordinates_x_[i];
    if(vertex_coordinates_y_[i] < min_y) min_y = vertex_coordinates_y_[i];
    if(vertex_coordinates_x_[i] > max_x) max_x = vertex_coordinates_x_[i];
    if(vertex_coordinates_y_[i] > max_y) max_y = vertex_coordinates_y_[i];
  }
  min_x_ = min_x;
  min_y_ = min_y;
  max_x_ = max_x;
  max_y_ = max_y;
}

const scalar_t
angle_2d(const scalar_t & x1,
  const scalar_t & y1,
  const scalar_t & x2,
  const scalar_t & y2){

  scalar_t dtheta=0,theta1=0,theta2=0;
  theta1 = std::atan2(y1,x1);
  theta2 = std::atan2(y2,x2);
  dtheta = theta2 - theta1;
  while (dtheta > DICE_PI)
    dtheta -= DICE_TWOPI;
  while (dtheta < -DICE_PI)
    dtheta += DICE_TWOPI;
  return(dtheta);
}

void
Polygon::deactivate_pixels(Teuchos::ArrayRCP<bool> & pixel_flags,
  Teuchos::ArrayRCP<int_t> x_coords,
  Teuchos::ArrayRCP<int_t> y_coords,
  const int_t origin_x,
  const int_t origin_y) const{

  assert(x_coords.size()==y_coords.size());
  assert(pixel_flags.size()==x_coords.size());

  int x=0,y=0;
  int dx1=0,dx2=0,dy1=0,dy2=0;
  scalar_t angle=0.0;
  for(int_t j=0;j<x_coords.size();++j){
    x = x_coords[j] + origin_x;
    y = y_coords[j] + origin_y;
    angle=0.0;
    for (int_t i=0;i<num_vertices_;i++) {
      // get the two end points of the polygon side and construct
      // a vector from the point to each one:
      dx1 = vertex_coordinates_x_[i] - x;
      dy1 = vertex_coordinates_y_[i] - y;
      dx2 = vertex_coordinates_x_[i+1] - x;
      dy2 = vertex_coordinates_y_[i+1] - y;
      angle += angle_2d(dx1,dy1,dx2,dy2);
    }
    // if the angle is greater than PI, the point is in the polygon
    if(std::abs(angle) >= DICE_PI){
      pixel_flags[j] = false;
    }
  }
}


// NOTE: The pair is (y,x) not (x,y) so that the ordering in the set will match loops over y then x
std::set<std::pair<int_t,int_t> >
Polygon::get_owned_pixels(Teuchos::RCP<Def_Map> map,
  const int_t cx,
  const int_t cy,
  const scalar_t skin_factor)const{

  std::vector<int_t> verts_x = vertex_coordinates_x_;
  std::vector<int_t> verts_y = vertex_coordinates_y_;
  int_t min_x = min_x_;
  int_t min_y = min_y_;
  int_t max_x = max_x_;
  int_t max_y = max_y_;

  if(map!=Teuchos::null){
    scalar_t u     = map->u_;
    scalar_t v     = map->v_;
    scalar_t theta = map->t_;
    scalar_t dudx  = map->ex_;
    scalar_t dvdy  = map->ey_;
    scalar_t gxy   = map->g_;
    scalar_t dx=0.0,dy=0.0;
    scalar_t X=0.0,Y=0.0;
    int_t new_x=0,new_y=0;
    for(int_t i=0;i<vertex_coordinates_x_.size();++i){
      int_t x = vertex_coordinates_x_[i];
      int_t y = vertex_coordinates_y_[i];
      // iterate over all the pixels in the reference blocking subset and compute the current position
      dx = (1.0+dudx)*(x-cx) + gxy*(y-cy);
      dy = (1.0+dvdy)*(y-cy) + gxy*(x-cx);
      X = std::cos(theta)*dx - std::sin(theta)*dy + u + cx;
      Y = std::sin(theta)*dx + std::cos(theta)*dy + v + cy;
      new_x = (int)X;
      if(X - (int)X >= 0.5) new_x++;
      new_y = (int)Y;
      if(Y - (int)Y >= 0.5) new_y++;
      verts_x[i] = new_x;
      verts_y[i] = new_y;
    }
    // compute the geometric centroid of the new vertices:
    int_t centroid_x = 0;
    int_t centroid_y = 0;
    for(int_t i=0;i<verts_x.size()-1;++i){
      centroid_x+=verts_x[i];
      centroid_y+=verts_y[i];
    }
    centroid_x /= (verts_x.size()-1);
    centroid_y /= (verts_y.size()-1);
    // apply the skin factor
    for(int_t i=0;i<verts_x.size();++i){
      // add the skin to the new vertex (applied as a stretch in x and y):
      new_x = skin_factor*(verts_x[i] - centroid_x) + centroid_x;
      new_y = skin_factor*(verts_y[i] - centroid_y) + centroid_y;
      verts_x[i] = new_x;
      verts_y[i] = new_y;
      if(i==0){
        min_x = new_x;
        max_x = new_x;
        min_y = new_y;
        max_y = new_y;
      }
      else{
        if(new_x < min_x){min_x = new_x;}
        if(new_x > max_x){max_x = new_x;}
        if(new_y < min_y){min_y = new_y;}
        if(new_y > max_y){max_y = new_y;}
      }
    } // vertex_loop
  }

  std::set<std::pair<int_t,int_t> > coordSet;

  int dx1=0,dx2=0,dy1=0,dy2=0;
  scalar_t angle=0;
  // rip over the points in the extents of the polygon to determine which onese are inside
  for(int_t y=min_y;y<=max_y;++y){
    for(int_t x=min_x;x<=max_x;++x){
      // x and y are the global coordinates of the point to test
      angle=0.0;
      for (int_t i=0;i<num_vertices_;i++) {
        // get the two end points of the polygon side and construct
        // a vector from the point to each one:
        dx1 = verts_x[i] - x;
        dy1 = verts_y[i] - y;
        dx2 = verts_x[i+1] - x;
        dy2 = verts_y[i+1] - y;
        angle += angle_2d(dx1,dy1,dx2,dy2);
      }
      // if the angle is greater than PI, the point is in the polygon
      if(std::abs(angle) >= DICE_PI){
        coordSet.insert(std::pair<int_t,int_t>(y,x));
      }
    }
  }
  return coordSet;
}

void
Polygon::draw(Teuchos::RCP<Image> & layer_0_image,
  Teuchos::RCP<Def_Map> map,
  const int_t cx,
  const int_t cy)const{

  const int_t img_width = layer_0_image->width();
  const int_t img_height = layer_0_image->height();

  // get the intensities of the layer_0_image
  Teuchos::ArrayRCP<const scalar_t> img_intensities = layer_0_image->intensity_array();
  Teuchos::ArrayRCP<scalar_t> intensity_values(img_width*img_height,0.0);
  for(int_t i=0;i<img_width*img_height;++i){
    intensity_values[i] = img_intensities[i];
  }

  scalar_t u=0.0,v=0.0,theta=0.0,dudx=0.0,dvdy=0.0,gxy=0.0;
  if(map!=Teuchos::null){
    u     = map->u_;
    v     = map->v_;
    theta = map->t_;
    dudx  = map->ex_;
    dvdy  = map->ey_;
    gxy   = map->g_;
  }

  std::vector<int_t> new_vertices_x(vertex_coordinates_x_.size());
  std::vector<int_t> new_vertices_y(vertex_coordinates_y_.size());

  int x=0,y=0,token_size=3;
  int dx=0,dy=0;
  int lx=0,ly=0;
  scalar_t X=0.0, Y=0.0;
  for(int_t i=0;i<=num_vertices_;++i){
    x = vertex_coordinates_x_[i];
    y = vertex_coordinates_y_[i];
    lx = x - cx;
    ly = y - cy;
    dx = (1.0+dudx)*lx + gxy*ly;
    dy = (1.0+dvdy)*ly + gxy*lx;
    X = std::cos(theta)*dx - std::sin(theta)*dy + u + cx;
    Y = std::sin(theta)*dx + std::cos(theta)*dy + v + cy;
    new_vertices_x[i] = (int_t)X;
    new_vertices_y[i] = (int_t)Y;
  }

  // put a white plus sign at all of the veritices:
  for(int_t i=0;i<num_vertices_;++i){
    x = new_vertices_x[i];
    y = new_vertices_y[i];

    if(x+token_size>img_width) continue;
    if(x-token_size<0) continue;
    if(y+token_size>img_width) continue;
    if(y-token_size<0) continue;
    intensity_values[y*img_width+x] = 255;
    for(int_t j=1;j<=token_size;++j){
      intensity_values[(y-j)*img_width+x] = 255;
      intensity_values[(y+j)*img_width+x] = 255;
      intensity_values[y*img_width+x+j] = 255;
      intensity_values[y*img_width+x-j] = 255;
    }
  }

  // draw a line between each segment:
  int xp=0,yp=0,numSteps=0,numPerStep=1,xs=0,ys=0,incrementX=1,incrementY=1;//,start_x=0,start_y=0;
  scalar_t slope=0;
  for(int_t i=0;i<num_vertices_;++i){
    x = new_vertices_x[i];
    y = new_vertices_y[i];
    xp = new_vertices_x[i+1];
    yp = new_vertices_y[i+1];

    if(std::abs(yp-y)>0 && std::abs(xp-x)>0){
      slope = std::abs((scalar_t)(yp-y) / (scalar_t)(xp-x));
      numSteps = std::abs(xp-x);
      assert(numSteps>0 && "  DICe ERROR: numSteps must not be zero.");
      numPerStep = std::max(std::abs(yp-y) / numSteps,1);
      incrementX = numSteps/(xp-x);
      incrementY = std::abs(yp-y)/(yp-y);
      for(int_t j=0;j<numSteps;++j){
        xs = x+j*incrementX;
        ys = (int_t)(y+slope*j*incrementY);
        for(int_t k=0;k<numPerStep;++k)
          intensity_values[(ys+k*incrementY)*img_width+xs] = 255;
      }
    }
    else{
      if(yp-y==0){
        numSteps = std::abs(xp-x);
        if(numSteps==0) continue;
        assert(xp-x!=0 && "  DICe ERROR: xp - x cannot be zero.");
        incrementX = numSteps/(xp-x);
        for(int_t j=0;j<numSteps;++j)
          intensity_values[y*img_width+x+j*incrementX] = 255;
      }
      if(xp-x==0){
        numSteps = std::abs(yp-y);
        if(numSteps==0) continue;
        assert(yp-y!=0 && "  DICe ERROR: yp - y cannot be zero.");
        incrementY = numSteps/(yp-y);
        for(int_t j=0;j<numSteps;++j)
          intensity_values[(y+j*incrementY)*img_width+x] = 255;
      }
    }
  }
  layer_0_image->replace_intensities(intensity_values);
}


Circle::Circle(const int_t centroid_x,
  const int_t centroid_y,
  const scalar_t & radius):
  centroid_x_(centroid_x),
  centroid_y_(centroid_y),
  radius_(radius),
  radius2_(radius*radius)
{
  assert(centroid_x_>0);
  assert(centroid_y_>0);
  assert(radius_>0);
  min_x_ = centroid_x_ - radius - 1;
  max_x_ = centroid_x_ + radius + 1;
  min_y_ = centroid_y_ - radius - 1;
  max_y_ = centroid_y_ + radius + 1;
}

void
Circle::deactivate_pixels(Teuchos::ArrayRCP<bool> & pixel_flags,
  Teuchos::ArrayRCP<int_t> x_coords,
  Teuchos::ArrayRCP<int_t> y_coords,
  const int_t origin_x,
  const int_t origin_y) const{

  assert(x_coords.size()==y_coords.size());
  assert(pixel_flags.size()==x_coords.size());

  int x=0,y=0;
  int dx=0,dy=0;
  for(int_t j=0;j<x_coords.size();++j){
    x = x_coords[j] + origin_x;
    y = y_coords[j] + origin_y;
    dx = (x - centroid_x_)*(x - centroid_x_);
    dy = (y - centroid_y_)*(y - centroid_y_);
    if(dx + dy <= radius2_){
      pixel_flags[j] = false;
    }
  }
}

// NOTE: The pair is (y,x) not (x,y) so that the ordering in the set will match loops over y then x
std::set<std::pair<int_t,int_t> >
Circle::get_owned_pixels(Teuchos::RCP<Def_Map> map,
  const int_t cx,
  const int_t cy,
  const scalar_t skin_factor)const{
  TEUCHOS_TEST_FOR_EXCEPTION(map!=Teuchos::null,std::runtime_error,"Error, circle deformation has not been implemented yet");
  std::set<std::pair<int_t,int_t> > coordSet;

  scalar_t dx=0,dy=0;
  // rip over the points in the extents of the circle to determine which onese are inside
  for(int_t y=min_y_;y<=max_y_;++y){
    for(int_t x=min_x_;x<=max_x_;++x){
      // x and y are the global coordinates of the point to test
      dx = (x-centroid_x_)*(x-centroid_x_);
      dy = (y-centroid_y_)*(y-centroid_y_);
      if(dx + dy <= radius2_){
        coordSet.insert(std::pair<int_t,int_t>(y,x));
      }
    }
  }
  return coordSet;
}

void
Circle::draw(Teuchos::RCP<Image> & layer_0_image,
  Teuchos::RCP<Def_Map> map)const{

  const int_t img_width = layer_0_image->width();
  const int_t img_height = layer_0_image->height();

  // get the intensities of the layer_0_image
  Teuchos::ArrayRCP<const scalar_t> img_intensities = layer_0_image->intensity_array();
  Teuchos::ArrayRCP<scalar_t> intensity_values(img_width*img_height,0.0);
  for(int_t i=0;i<img_width*img_height;++i){
    intensity_values[i] = img_intensities[i];
  }

  scalar_t u=0.0,v=0.0,theta=0.0,dudx=0.0,dvdy=0.0,gxy=0.0;
  if(map!=Teuchos::null){
    u     = map->u_;
    v     = map->v_;
    theta = map->t_;
    dudx  = map->ex_;
    dvdy  = map->ey_;
    gxy   = map->g_;
  }
  // construct a set of eight points on the perimeter
  std::vector<scalar_t> initialX(8);
  std::vector<scalar_t> initialY(8);
  scalar_t angle = 0.0;
  for(int_t i=0;i<8;++i){
    initialX[i] = centroid_x_ + radius_*std::cos(angle);
    initialY[i] = centroid_y_ + radius_*std::sin(angle);
    angle += DICE_PI/4;
  }
  // transform the coordinates to the deformed shape
  std::vector<int_t> perimX(8);
  std::vector<int_t> perimY(8);
  scalar_t X=0.0, Y=0.0;
  scalar_t dx=0,dy=0;
  scalar_t lx=0,ly=0;
  for(int_t i=0;i<8;++i){
    lx = initialX[i] - centroid_x_;
    ly = initialY[i] - centroid_y_;
    dx = (1.0+dudx)*lx + gxy*ly;
    dy = (1.0+dvdy)*ly + gxy*lx;
    X = std::cos(theta)*dx - std::sin(theta)*dy + u + centroid_x_;
    Y = std::sin(theta)*dx + std::cos(theta)*dy + v + centroid_y_;
    perimX[i] = (int_t)X;
    perimY[i] = (int_t)Y;
  }
  // put a white plus sign at all of the perimeter points:
  int x=0,y=0,token_size=3;
  for(int_t i=0;i<8;++i){
    x = perimX[i];
    y = perimY[i];
    if(x+token_size>img_width) continue;
    if(x-token_size<0) continue;
    if(y+token_size>img_width) continue;
    if(y-token_size<0) continue;
    intensity_values[y*img_width+x] = 255;
    for(int_t j=1;j<=token_size;++j){
      intensity_values[(y-j)*img_width+x] = 255;
      intensity_values[(y+j)*img_width+x] = 255;
      intensity_values[y*img_width+x+j] = 255;
      intensity_values[y*img_width+x-j] = 255;
    }
  }
  layer_0_image->replace_intensities(intensity_values);
}

Rectangle::Rectangle(const int_t centroid_x,
  const int_t centroid_y,
  const int_t width,
  const int_t height):
  centroid_x_(centroid_x),
  centroid_y_(centroid_y),
  width_(width),
  height_(height)
{
  assert(centroid_x_>0);
  assert(centroid_y_>0);
  assert(width_>0);
  assert(height_>0);

  origin_x_ = centroid_x_ - width/2;
  origin_y_ = centroid_y_ - height/2;

  assert(origin_x_ >= 0);
  assert(origin_y_ >= 0);
}

void
Rectangle::deactivate_pixels(Teuchos::ArrayRCP<bool> & pixel_flags,
  Teuchos::ArrayRCP<int_t> x_coords,
  Teuchos::ArrayRCP<int_t> y_coords,
  const int_t origin_x,
  const int_t origin_y) const{

  assert(x_coords.size()==y_coords.size());
  assert(pixel_flags.size()==x_coords.size());

  int x=0,y=0;
  int dx=0,dy=0;
  for(int_t j=0;j<x_coords.size();++j){
    x = x_coords[j] + origin_x;
    y = y_coords[j] + origin_y;
    if(x >= origin_x_ && x < origin_x_ + width_ && y >= origin_y_ && y < origin_y_  + height_)
      pixel_flags[j] = false;
  }
}

// NOTE: The pair is (y,x) not (x,y) so that the ordering in the set will match loops over y then x
std::set<std::pair<int_t,int_t> >
Rectangle::get_owned_pixels(Teuchos::RCP<Def_Map> map,
  const int_t cx,
  const int_t cy,
  const scalar_t skin_factor)const{

  std::set<std::pair<int_t,int_t> > coordSet;

  if(map!=Teuchos::null){
    int_t min_x = 0;
    int_t min_y = 0;
    int_t max_x = 0;
    int_t max_y = 0;
    scalar_t u     = map->u_;
    scalar_t v     = map->v_;
    scalar_t theta = map->t_;
    scalar_t dudx  = map->ex_;
    scalar_t dvdy  = map->ey_;
    scalar_t gxy   = map->g_;
    scalar_t dx=0.0,dy=0.0;
    scalar_t X=0.0,Y=0.0;
    int_t new_x=0,new_y=0;
    std::vector<int_t> vertex_coordinates_x(5,0.0);
    std::vector<int_t> vertex_coordinates_y(5,0.0);
    std::vector<int_t> verts_x(5,0.0);
    std::vector<int_t> verts_y(5,0.0);
    vertex_coordinates_x[0] = origin_x_;
    vertex_coordinates_x[1] = origin_x_ + width_;
    vertex_coordinates_x[2] = origin_x_ + width_;
    vertex_coordinates_x[3] = origin_x_;
    vertex_coordinates_x[4] = origin_x_;
    vertex_coordinates_y[0] = origin_y_;
    vertex_coordinates_y[1] = origin_y_;
    vertex_coordinates_y[2] = origin_y_ + height_;
    vertex_coordinates_y[3] = origin_y_ + height_;
    vertex_coordinates_y[4] = origin_y_;

    for(int_t i=0;i<vertex_coordinates_x.size();++i){
      int x = vertex_coordinates_x[i];
      int y = vertex_coordinates_y[i];
      // iterate over all the pixels in the reference blocking subset and compute the current position
      dx = (1.0+dudx)*(x-cx) + gxy*(y-cy);
      dy = (1.0+dvdy)*(y-cy) + gxy*(x-cx);
      X = std::cos(theta)*dx - std::sin(theta)*dy + u + cx;
      Y = std::sin(theta)*dx + std::cos(theta)*dy + v + cy;
      new_x = (int_t)X;
      if(X - (int_t)X >= 0.5) new_x++;
      new_y = (int_t)Y;
      if(Y - (int_t)Y >= 0.5) new_y++;
      verts_x[i] = new_x;
      verts_y[i] = new_y;
    } // vertex loop
    // compute the geometric centroid of the new vertices:
    int_t centroid_x = 0;
    int_t centroid_y = 0;
    for(int_t i=0;i<verts_x.size()-1;++i){
      centroid_x+=verts_x[i];
      centroid_y+=verts_y[i];
    }
    centroid_x /= (verts_x.size()-1);
    centroid_y /= (verts_y.size()-1);
    // apply the skin factor
    for(int_t i=0;i<verts_x.size();++i){
      // add the skin to the new vertex (applied as a stretch in x and y):
      new_x = skin_factor*(verts_x[i] - centroid_x) + centroid_x;
      new_y = skin_factor*(verts_y[i] - centroid_y) + centroid_y;
      verts_x[i] = new_x;
      verts_y[i] = new_y;
      if(i==0){
        min_x = new_x;
        max_x = new_x;
        min_y = new_y;
        max_y = new_y;
      }
      else{
        if(new_x < min_x){min_x = new_x;}
        if(new_x > max_x){max_x = new_x;}
        if(new_y < min_y){min_y = new_y;}
        if(new_y > max_y){max_y = new_y;}
      }
    } // vertex_loop

    int dx1=0,dx2=0,dy1=0,dy2=0;
    scalar_t angle=0;
    // rip over the points in the extents of the polygon to determine which onese are inside
    for(int y=min_y;y<=max_y;++y){
      for(int x=min_x;x<=max_x;++x){
        // x and y are the global coordinates of the point to test
        angle=0.0;
        for (int_t i=0;i<4;i++) {
          // get the two end points of the polygon side and construct
          // a vector from the point to each one:
          dx1 = verts_x[i] - x;
          dy1 = verts_y[i] - y;
          dx2 = verts_x[i+1] - x;
          dy2 = verts_y[i+1] - y;
          angle += angle_2d(dx1,dy1,dx2,dy2);
        }
        // if the angle is greater than PI, the point is in the polygon
        if(std::abs(angle) >= DICE_PI){
          coordSet.insert(std::pair<int_t,int_t>(y,x));
        }
      }
    }
  } // has deformation
  else{
    scalar_t dx=0,dy=0;
    // rip over the points in the extents of the circle to determine which onese are inside
    for(int_t y=0;y<height_;++y){
      for(int_t x=0;x<=width_;++x){
        coordSet.insert(std::pair<int_t,int_t>(origin_y_+y,origin_x_+x));
      }
    }
  }
  return coordSet;
}

void Rectangle::draw(Teuchos::RCP<Image> & layer_0_image,
  Teuchos::RCP<Def_Map> map)const{

  const int_t img_width = layer_0_image->width();
  const int_t img_height = layer_0_image->height();

  // get the intensities of the layer_0_image
  Teuchos::ArrayRCP<const scalar_t> img_intensities = layer_0_image->intensity_array();
  Teuchos::ArrayRCP<scalar_t> intensity_values(img_width*img_height,0.0);
  for(int_t i=0;i<img_width*img_height;++i){
    intensity_values[i] = img_intensities[i];
  }

  scalar_t u=0.0,v=0.0,theta=0.0,dudx=0.0,dvdy=0.0,gxy=0.0;
  if(map!=Teuchos::null){
    u     = map->u_;
    v     = map->v_;
    theta = map->t_;
    dudx  = map->ex_;
    dvdy  = map->ey_;
    gxy   = map->g_;
  }
  const int_t num_vertices = 5;
  std::vector<int_t> new_vertices_x(num_vertices);
  std::vector<int_t> new_vertices_y(num_vertices);
  std::vector<int_t> vertices_x(num_vertices);
  std::vector<int_t> vertices_y(num_vertices);
  vertices_x[0] = origin_x_; vertices_y[0] = origin_y_;
  vertices_x[1] = origin_x_ + width_ - 1; vertices_y[1] = origin_y_;
  vertices_x[2] = origin_x_ + width_ - 1; vertices_y[2] = origin_y_ + height_ -1;
  vertices_x[3] = origin_x_; vertices_y[3] = origin_y_ + height_ -1;
  vertices_x[4] = origin_x_; vertices_y[4] = origin_y_;

  int x=0,y=0,token_size=3;
  int dx=0,dy=0;
  int lx=0,ly=0;
  scalar_t X=0.0, Y=0.0;
  for(int_t i=0;i<=num_vertices;++i){
    x = vertices_x[i];
    y = vertices_y[i];
    lx = x - centroid_x_;
    ly = y - centroid_y_;
    dx = (1.0+dudx)*lx + gxy*ly;
    dy = (1.0+dvdy)*ly + gxy*lx;
    X = std::cos(theta)*dx - std::sin(theta)*dy + u + centroid_x_;
    Y = std::sin(theta)*dx + std::cos(theta)*dy + v + centroid_y_;
    new_vertices_x[i] = (int_t)X;
    new_vertices_y[i] = (int_t)Y;
  }

  // put a white plus sign at all of the veritices:
  for(int_t i=0;i<num_vertices;++i){
    x = new_vertices_x[i];
    y = new_vertices_y[i];

    if(x+token_size>img_width) continue;
    if(x-token_size<0) continue;
    if(y+token_size>img_width) continue;
    if(y-token_size<0) continue;
    intensity_values[y*img_width+x] = 255;
    for(int_t j=1;j<=token_size;++j){
      intensity_values[(y-j)*img_width+x] = 255;
      intensity_values[(y+j)*img_width+x] = 255;
      intensity_values[y*img_width+x+j] = 255;
      intensity_values[y*img_width+x-j] = 255;
    }
  }

  // draw a line between each segment:
  int xp=0,yp=0,numSteps=0,numPerStep=1,xs=0,ys=0,incrementX=1,incrementY=1;//,start_x=0,start_y=0;
  scalar_t slope=0;
  for(int_t i=0;i<num_vertices;++i){
    x = new_vertices_x[i];
    y = new_vertices_y[i];
    xp = new_vertices_x[i+1];
    yp = new_vertices_y[i+1];

    if(std::abs(yp-y)>0 && std::abs(xp-x)>0){
      slope = std::abs((scalar_t)(yp-y) / (scalar_t)(xp-x));
      numSteps = std::abs(xp-x);
      assert(numSteps>0 && "  DICe ERROR: numSteps must not be zero.");
      numPerStep = std::max(std::abs(yp-y) / numSteps,1);
      incrementX = numSteps/(xp-x);
      incrementY = std::abs(yp-y)/(yp-y);
      for(int_t j=0;j<numSteps;++j){
        xs = x+j*incrementX;
        ys = (int_t)(y+slope*j*incrementY);
        for(int_t k=0;k<numPerStep;++k)
          intensity_values[(ys+k*incrementY)*img_width+xs] = 255;
      }
    }
    else{
      if(yp-y==0){
        numSteps = std::abs(xp-x);
        if(numSteps==0) continue;
        assert(xp-x!=0 && "  DICe ERROR: xp - x cannot be zero.");
        incrementX = numSteps/(xp-x);
        for(int_t j=0;j<numSteps;++j)
          intensity_values[y*img_width+x+j*incrementX] = 255;
      }
      if(xp-x==0){
        numSteps = std::abs(yp-y);
        if(numSteps==0) continue;
        assert(yp-y!=0 && "  DICe ERROR: yp - y cannot be zero.");
        incrementY = numSteps/(yp-y);
        for(int_t j=0;j<numSteps;++j)
          intensity_values[(y+j*incrementY)*img_width+x] = 255;
      }
    }
  }
  layer_0_image->replace_intensities(intensity_values);
}

}// End DICe Namespace
