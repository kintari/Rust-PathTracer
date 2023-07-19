use crate::numeric::*;

pub trait Sdf {
	fn distance(&self, p: Float3) -> f32;
}



pub struct Plane {
	n: Float3,
	d: f32,
}

impl Plane {
	
	pub fn new(n: Float3, d: f32) -> Plane {
		return Plane { n, d };
	}

}

impl Sdf for Plane {

	fn distance(&self, p: Float3) -> f32 {
		return dot(p,self.n)-self.d;
	}

}



pub struct Sphere {
	center: Float3,
	radius: f32
}

impl Sphere {

	pub fn new(center: Float3, radius: f32) -> Sphere {
		return Sphere { center, radius };
	}

}

impl Sdf for Sphere {

	fn distance(&self, p: Float3) -> f32 {
		return length(p-self.center)-self.radius;
	}

}
