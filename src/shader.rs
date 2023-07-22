use crate::numeric::*;

pub trait Shader {
	fn main(&mut self, frag_coord: Float3, resolution: Float3, prev_color: Float3) -> Float3;
}
