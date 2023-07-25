
use std::ops::*;

#[derive(Copy,Clone,Debug)]
pub struct Float3 {
	pub x: f32,
	pub y: f32,
	pub z: f32
}

pub fn dot(u: Float3, v: Float3) -> f32 {
	return u.x*v.x + u.y*v.y + u.z*v.z;
}

pub fn length(u: Float3) -> f32 {
	return f32::sqrt(dot(u,u));
}

pub fn normalize(v: Float3) -> Float3 {
	let rcp_len = 1.0 / length(v);
	return v * rcp_len;
}

pub fn lerp(a: Float3, b: Float3, t: f32) -> Float3 {
	let t = f32::clamp(t, 0.0, 1.0);
	return t*b + (1.0-t)*a;
}

pub trait Abs {
	fn abs(v: Self) -> Self;
}

impl Abs for Float3 {
	fn abs(v: Self) -> Self {
		return Self {
			x: f32::abs(v.x),
			y: f32::abs(v.y),
			z: f32::abs(v.z)
		};
	}
}

pub fn abs<T: Abs>(val: T) -> T {
	return T::abs(val);
}

pub trait Max {
	fn max(u: Self, v: Self) -> Self;
}

/*
impl Max for f32 {
	fn max(u: Self, v: Self) -> Self {
		return f32::max(u, v);
	}
}
*/

impl Max for Float3 {
	fn max(u: Self, v: Self) -> Self {
		return Self {
			x: f32::max(u.x, v.x),
			y: f32::max(u.y, v.y),
			z: f32::max(u.z, v.z)
		};
	}
}

pub fn max<T: Max>(u: T, v: T) -> T {
	return T::max(u, v);
}

pub trait Saturate {
	fn saturate(self: Self) -> Self;
}

impl Saturate for f32 {
	fn saturate(self) -> Self {
		return f32::clamp(self, 0.0, 1.0);
	}
}

impl Saturate for Float3 {
	fn saturate(self: Self) -> Self {
		return Self {
			x: self.x.saturate(),
			y: self.y.saturate(),
			z: self.z.saturate()
		};
	}
}

pub fn saturate<V: Saturate>(v: V) -> V {
	return v.saturate();
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

impl Float3 {
	pub fn new(x: f32, y: f32, z: f32) -> Self {
		return Self { x, y, z };
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
/*
impl Add<f32> for Float3 {
	type Output = Float3;
	fn add(self, rhs: f32) -> Self::Output {
		return self + Float3::new(rhs,rhs,rhs);
	}
}
*/
impl AddAssign<Float3> for Float3 {
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl AddAssign<f32> for Float3 {
	fn add_assign(&mut self, rhs: f32) {
		*self = *self + Float3::new(rhs, rhs, rhs);	
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

impl MulAssign<Float3> for Float3 {
	fn mul_assign(&mut self, rhs: Float3) {
		*self = Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z);
	}
}

impl Neg for Float3 {
	type Output = Self;
	fn neg(self) -> Self {
		return Self::new(-self.x, -self.y, -self.z);
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
