
use std::ops::{Add,Sub,Mul,Div};

use super::Vector;

#[derive(Copy,Clone)]
pub struct Float3 {
	pub x: f32,
	pub y: f32,
	pub z: f32
}

#[macro_export]
macro_rules! float3 {
	($x:expr,$y:expr,$z:expr) => {
		Float3::new($x as f32, $y as f32, $z as f32)
	};
	($x:expr) => {
		Float3::new($x as f32, $x as f32, $x as f32)
	};
}

pub(crate) use float3;

impl Vector for Float3 {
	
	fn dot(u: Float3, v: Float3) -> f32 {
		return u.x*v.x + u.y*v.y + u.z*v.z;
	}

}

impl Float3 {
	
	pub fn new(x: f32, y: f32, z: f32) -> Self {
		return Self { x, y, z };
	}

}

fn abs(v: Float3) -> Float3 {
	return Float3 { x: f32::abs(v.x), y: f32::abs(v.y), z: f32::abs(v.z) };
}

impl Mul<f32> for Float3 {
	type Output = Float3;
	fn mul(self, rhs: f32) -> Self::Output {
		return Self {
			x: self.x * rhs,
			y: self.y * rhs,
			z: self.z * rhs
		};
	}
}

impl Mul<Float3> for f32 {
	type Output = Float3;
	fn mul(self, rhs: Float3) -> Self::Output {
		return rhs * self;
	}
}

impl Mul<Float3> for Float3 {
	type Output = Float3;
	fn mul(self, rhs: Float3) -> Self::Output {
		return Self::Output {
			x: self.x * rhs.x,
			y: self.y * rhs.y,
			z: self.z * rhs.z
		}
	}
}

impl Add<Float3> for Float3 {
	type Output = Float3;
	fn add(self, rhs: Self) -> Self::Output {
		return Self {
			x: self.x + rhs.x,
			y: self.y + rhs.y,
			z: self.z + rhs.z
		};
	}
}

impl Add<f32> for Float3 {
	type Output = Float3;
	fn add(self, rhs: f32) -> Self::Output {
		return self + Float3::new(rhs,rhs,rhs);
	}
}

impl Sub<Float3> for Float3 {
	type Output = Float3;
	fn sub(self, rhs: Self) -> Self::Output {
		return Self {
			x: self.x - rhs.x,
			y: self.y - rhs.y,
			z: self.z - rhs.z
		};
	}
}

impl Sub<f32> for Float3 {
	type Output = Float3;
	fn sub(self, rhs: f32) -> Self::Output {
		return self - Float3::new(rhs, rhs, rhs);
	}
}

impl Div<f32> for Float3 {
	type Output = Float3;
	fn div(self, rhs: f32) -> Self::Output {
		return self / Float3::new(rhs, rhs, rhs);
	}
}

impl Div<Float3> for Float3 {
	type Output = Float3;
	fn div(self, rhs: Float3) -> Float3 {
		let x = self.x / rhs.x;
		let y = self.y / rhs.y;
		let z = self.z / rhs.z;
		return Float3::new(x, y, z);
	}
}

impl From<f32> for Float3 {
	fn from(value: f32) -> Float3 {
		return Float3::new(value, value, value);
	}
}
