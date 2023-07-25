/*
#[derive(Clone,Copy)]
pub struct Color {
	pub r: f32,
	pub g: f32,
	pub b: f32,
	pub a: f32
}

impl Color {

	pub fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
		return Color { r, g, b, a };
	}

}

use std::ops::Mul;

impl Mul<f32> for Color {
	type Output = Color;
	fn mul(self, value: f32) -> Color {
		return Color::new(
			self.r*value,
			self.g*value,
			self.b*value,
			self.a*value
		);
	}
}
*/