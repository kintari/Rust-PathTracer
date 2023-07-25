use crate::float3::*;

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
	pub center: Float3,
	pub radius: f32
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

pub struct Box {
	center: Float3,
	size: Float3
}

impl Box {
	pub fn new(center: Float3, size: Float3) -> Self {
		return Box { center, size };
	}
}

impl Sdf for Box {
	fn distance(&self, p: Float3) -> f32 {
		let b = 0.5*self.size;
		let q = abs(p-self.center)-b;
		return length(max(q, float3![0])) +
			f32::min(f32::max(q.x, f32::max(q.y, q.z)), 0.0);
	}	
}