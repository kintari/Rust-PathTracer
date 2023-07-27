
use crate::float3::Float3;

#[derive(Debug,Clone,Copy)]
pub struct Ray {
	pub p: Float3,
	pub d: Float3
}

impl FnOnce<(f32,)> for Ray {
	type Output = Float3;
	extern "rust-call" fn call_once(self, args: (f32,)) -> Self::Output {
		return self.p + args.0 * self.d;
	}
}