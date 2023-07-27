use crate::float3::*;
use crate::ray::*;

pub trait Shader {
	fn main(&self, ray: Ray) -> Float3;
}
