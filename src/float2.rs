use std::ops::*;

#[derive(Debug,Clone,Copy)]
pub struct Float2 {
	pub x: f32,
	pub y: f32
}

#[macro_export]
macro_rules! float2 {
	($x:expr,$y:expr) => {
		Float2::new($x as f32, $y as f32)
	};
	($x:expr) => {
		Float3::new($x as f32, $x as f32)
	};
}

pub(crate) use float2;

impl Float2 {
	pub fn new(x: f32, y: f32) -> Float2 {
		return Float2 { x, y };
	}
}

impl Add<Float2> for Float2 {
	type Output = Self;
	fn add(self, v: Self) -> Self {
		return Self::new(self.x+v.x,self.y+v.y);
	}
}

impl Sub<Float2> for Float2 {
	type Output = Self;
	fn sub(self, v: Self) -> Self {
		return Self::new(self.x-v.x,self.y-v.y);
	}
}

impl Sub<f32> for Float2 {
	type Output = Self;
	fn sub(self, f: f32) -> Self::Output {
		return Self::new(self.x-f,self.y-f);
	}
}

impl Mul<f32> for Float2 {
	type Output = Self;
	fn mul(self, f: f32) -> Self {
		return Self::new(f*self.x,f*self.y);
	}
}

impl Mul<Float2> for f32 {
	type Output = Float2;
	fn mul(self, v: Float2) -> Self::Output {
		return v * self;
	}
}

impl Mul<Float2> for Float2 {
	type Output = Self;
	fn mul(self, v: Self) -> Self {
		return Self::new(self.x*v.x,self.y*v.y);
	}
}

impl Div<Float2> for Float2 {
	type Output = Self;
	fn div(self, v: Self) -> Self::Output {
		return Self::new(self.x/v.x,self.y/v.y);
	}
}