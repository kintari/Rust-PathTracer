pub mod float3;

pub use float3::Float3;

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

pub fn dot<T: Vector>(u: T, v: T) -> f32 {
	return T::dot(u, v);
}

use std::ops::{Add,Sub,Mul,Div};

pub trait Vector
	where Self: Sized+Copy+Mul<f32,Output=Self>+Div<f32,Output=Self>
{
	fn dot(u: Self, v: Self) -> f32;	
}