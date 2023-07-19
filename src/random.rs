use rand_pcg::Pcg32;
use rand_core::SeedableRng;
use rand::Rng;

use crate::numeric::*;
use crate::numeric::float3::*;

use std::f32::consts::PI;

pub struct Random {
	rng: Pcg32
}

impl Random {

	pub fn new() -> Random {
		return Random {
			rng: Pcg32::from_entropy()
		};
	}

	pub fn uniform(&mut self) -> f32 {
		return self.rng.gen();
	}

	pub fn uniform_in_range(&mut self, low: f32, high: f32) -> f32 {
		return low+(high-low)*self.rng.gen::<f32>()
	}

	pub fn unit_vector(&mut self) -> Float3 {
		let theta = self.uniform_in_range(0.0, 2.0*PI);
		let z = self.uniform_in_range(-1.0, 1.0);
		let r = f32::sqrt(f32::max(0.0,1.0-z*z));
		let x = r*f32::cos(theta);
		let y = r*f32::sin(theta);
		return float3![x,y,z];
	}

	pub fn cosine_weighted(&mut self, dir: Float3) -> Float3 {
		let v = self.unit_vector();
		
		// project onto plane orthogonal to 'dir'
		let u = v-Float3::dot(dir,v)*dir;

		let len = length(u);
		let z = f32::sqrt(f32::max(0.0,1.0-len*len));
		return u+z*dir;
	}

}
