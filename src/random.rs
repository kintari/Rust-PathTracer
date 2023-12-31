use std::cell::RefCell;
use rand_pcg::Pcg32;
use rand_core::SeedableRng;
use rand::Rng;
use std::f32::consts::PI;

use super::float3::*;

pub struct Random {
	rng: RefCell<Pcg32>
}

impl Random {

	pub fn new() -> Random {
		return Random {
			rng: RefCell::new(Pcg32::from_entropy())
		};
	}

	pub fn uniform(&self) -> f32 {
		return self.rng.borrow_mut().gen();
	}

	pub fn uniform_in_range(&self, low: f32, high: f32) -> f32 {
		return low+(high-low)*self.rng.borrow_mut().gen::<f32>()
	}

	pub fn unit_vector(&self) -> Float3 {
		let theta = self.uniform_in_range(0.0, 2.0*PI);
		let z = self.uniform_in_range(-1.0, 1.0);
		let r = f32::sqrt(f32::max(0.0,1.0-z*z));
		let x = r*f32::cos(theta);
		let y = r*f32::sin(theta);
		return float3![x,y,z];
	}

	pub fn cosine_weighted(&self, dir: Float3) -> Float3 {

		let v = self.unit_vector();
		
		// project onto plane orthogonal to 'dir'
		let u = v-dot(dir,v)*dir;

		let len = length(u);
		let z = f32::sqrt(f32::max(0.0,1.0-len*len));
		
		let result = u+z*dir;

		return result;
	}

}
