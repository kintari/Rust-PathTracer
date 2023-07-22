pub mod float3;

pub use float3::Float3;
pub use float3::Abs;

pub fn length<T>(v: T) -> f32
	where T: Vector
{
	return f32::sqrt(Vector::dot(v,v));
}

pub fn normalize<T>(v: T) -> T
	where T: Vector
{
	return v / length::<T>(v);
}

pub fn abs<T: Abs>(val: T) -> T {
	return T::abs(val);
}

pub fn dot<T: Vector>(u: T, v: T) -> f32 {
	return T::dot(u, v);
}

pub trait Saturate {
	fn saturate(x: Self) -> Self;
}

impl Saturate for f32 {
	fn saturate(x: f32) -> f32 {
		if x != x {
			return x;
		}
		else if x < 0.0 {
			return 0.0;
		}
		else if x > 1.0 {
			return 1.0;
		}
		else {
			return x;
		}
	}
}

impl Saturate for Float3 {
	fn saturate(v: Float3) -> Float3 {
		return Self::new(f32::saturate(v.x), f32::saturate(v.y), f32::saturate(v.z));
	}
}

pub fn saturate<T>(value: T) -> T
	where T: Copy+Saturate
{
	return T::saturate(value);
}


use std::ops::{Add,Sub,Mul,Div};

pub trait Vector
	where Self: Sized+Copy+Mul<f32,Output=Self>+Div<f32,Output=Self>
{
	fn dot(u: Self, v: Self) -> f32;	
}