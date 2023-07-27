#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

mod color;
mod float3;
mod buffer;
mod random;
mod shader;
mod sdf;

use color::*;
use buffer::*;
use shader::*;

use sdf::Sdf;

use std::path::{Path,PathBuf};
use std::fs::File;
use std::io::BufWriter;
use std::f32::consts::PI;
use std::ops::*;
use std::cmp::{min,max,Ordering};
use std::iter::zip;

use random::Random;
use float3::*;

macro_rules! cond {
	($test:expr,$true_expr:expr,$false_expr:expr) => {
		if $test { $true_expr } else { $false_expr }
	}
}

struct Hit {
	distance: f32,
	location: Float3,
	normal: Float3,
}

struct MyShader {
	random: Random,
	sphere: sdf::Sphere,
	walls: [sdf::Plane;6],
	half_dims: Float3,
	contact_distance: f32,
	light: sdf::Box,
	max_iterations: usize
}

impl MyShader {

	fn new() -> Self {
		let dx = 6.0;
		let dy = 6.0;
		let dz = 4.0;
		let walls = [
			sdf::Plane::new(float3![-1, 0, 0], -dx),
			sdf::Plane::new(float3![ 1, 0, 0], -dx),
			sdf::Plane::new(float3![ 0, 1, 0], -dy),
			sdf::Plane::new(float3![ 0,-1, 0], -dy),
			sdf::Plane::new(float3![ 0, 0,-1], -dz),
			sdf::Plane::new(float3![ 0, 0, 1], -25.0)
		];
		let light = sdf::Box::new(float3!(0,5.75,-2),float3!(4,0.50,4));
		let sphere = sdf::Sphere::new(float3![0,-1.5,0], 2.0);
		return Self {
			random: Random::new(),
			sphere,
			walls,
			half_dims: float3![dx, dy, dz],
			contact_distance: 1.0 / 1024.0,
			light,
			max_iterations: 250
		};
	}

	fn scene_distance(&self, p: Float3) -> f32 {
		let d_sphere = self.sphere.distance(p);
		let d_walls = self.walls.iter()
			.map(|sdf| sdf.distance(p))
			.reduce(|a,x| cond![a < x, a, x])
			.unwrap();
		let d_light = self.light.distance(p);
		let d = f32::min(f32::min(d_sphere, d_walls), d_light);
		return d;
	}

	fn emissive(&self, p: Float3) -> Float3 {
		let epsilon = 0.001;
		if self.light.distance(p) < epsilon {
			return float3![5.0];
		}
		else if self.walls[5].distance(p) < epsilon {
			return float3![1.0];
		}
		return float3![0.0];
	}

	fn diffuse_color(&self, p: Float3) -> Float3 {
		let d = 0.01;
		if f32::abs(p.x) + d >= self.half_dims.x {
			return cond![
				p.x > 0.0,
				float3![1,0.4,0.4],
				float3![0.4,1,0.4]
			];
		}
		else {
			return float3![0.5];
		}
	}

	fn calc_normal(&self, p: Float3) -> Float3 {
		let h = 1.0 / 512.0;
		let dx = h*float3![1,0,0];
		let dy = h*float3![0,1,0];
		let dz = h*float3![0,0,1];
		let x = self.scene_distance(p+dx)-self.scene_distance(p-dx);
		let y = self.scene_distance(p+dy)-self.scene_distance(p-dy);
		let z = self.scene_distance(p+dz)-self.scene_distance(p-dz);
		return normalize(float3![x, y, z]);
	}
	
	fn ray_march(&self, pos: Float3, dir: Float3, threshold: f32) -> Option<Hit> {
		let mut t = 0.0;
		let mut i = 0;
		while i < self.max_iterations {
			let p = pos + t * dir;
			let d = self.scene_distance(p);
			if d < threshold {
				let normal = self.calc_normal(p);
				let hit = Hit {
					distance: t,
					location: p,
					normal,
				};
				return Some(hit);
			}
			else {
				t += d;
				i += 1;
			}
		}
		return None;
	}
		
	fn unproject(&self, uv: Float3) -> Float3 {
		let clip = 2.0*uv-1.0;
		return normalize(float3![clip.x, clip.y, 2.4]);
	}
	
	fn env(&self, v: Float3) -> Float3 {
		return float3![0.0, 1.0, 1.0];
	}

}

impl Shader for MyShader {


	fn main(&mut self, fragCoord: Float3, resolution: Float3) -> Float3 {

		let uv = fragCoord / resolution;
		let camera = float3![0, 0, -20];
		let dir = self.unproject(float3![uv.x, uv.y, 0.0]);

		let mut P = camera;
		let mut D = dir;
		let mut Kd = float3![1];
		let roughness = 0.5;
		let mut I = float3![0];

		let mut iterations = 5;
		while iterations > 0 {
			if let Some(hit) = self.ray_march(P, D, self.contact_distance) {
				I += Float3::max(Kd * self.emissive(hit.location), float3![0]);
				
				D = normalize(lerp(
					hit.normal,
					self.random.unit_vector(),
					roughness));

				let cos_theta = dot(D, hit.normal);

				Kd *= cos_theta * self.diffuse_color(hit.location);

/*
				D = normalize(lerp(
					hit.normal,
					self.random.cosine_weighted(hit.normal),
					roughness));
*/

				P = hit.location + 4.0 * self.contact_distance * D;
			}
			else {
				I += Kd * self.diffuse_color(P) * self.env(D);
				break;
			}
			iterations -= 1;
		}
		return I;
	}

} // impl Shader

fn ACESFilm(x: Float3) -> Float3 {
	let a = float3![2.51];
	let b = float3![0.03];
	let c = float3![2.43];
	let d = float3![0.59];
	let e = float3![0.14];
	return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

fn exposure(value: Float3, e: f32) -> Float3 {
	f32::exp2(e) * value
}

struct PathTracer<T> {
	shader: T,
	image: Buffer<Float3>,
	exposure: f32,
	samples_per_pixel: usize,
	random: Random,
	iteration: usize
}

impl<T: Shader> PathTracer<T> {

	fn new(shader: T, image: Buffer::<Float3>) -> Self {
		return Self {
			shader,
			image,
			exposure: 0.0,
			samples_per_pixel: 64,
			iteration: 0,
			random: Random::new()
		};
	}

	fn iteration(&self) -> usize {
		return self.iteration;
	}

	fn render(&mut self) {
		let w = self.image.get_width();
		let h = self.image.get_height();
		let resolution = float3![w,h,0];
		let fac = 1.0 / (self.samples_per_pixel as f32);
		let mut f = |i,j| {
			let mut I = float3![0];
			for _ in 0 .. self.samples_per_pixel {
				let jitter = float3!(self.random.uniform(), self.random.uniform(), 0.0) - float3!(0.5, 0.5, 0.0);
				let frag_coord = float3![i, h-j-1, 0] + jitter;
				I += fac * self.shader.main(frag_coord, resolution);
			}
			return I;
		};
		self.image.fill(|i,j,&prev_color| prev_color + f(i,j));
		self.iteration += 1;
	}

	fn image(&self) -> &Buffer<Float3> {
		return &self.image;
	}

	fn postprocess(&self) -> Buffer<Float3> {
		let fac = 1.0 / (self.iteration as f32);
		return self.image
			.map(|&color| ACESFilm(fac*color));
	}

} // impl PathTracer

fn quantize_to_8bits(value: f32) -> u8 {
	let i: i32 = f32::floor(value * 255.0) as i32;
	let i = i32::clamp(i, 0, 255);
	return i as u8;
}

fn save_image(image: &Buffer<Float3>, path: &Path) {
	let data: Vec<u8> = image.get_pixels().iter()
		.flat_map(|px| [px.x,px.y,px.z,1.0])
		.map(quantize_to_8bits)
		.collect();
	image::save_buffer(
		path,
		&data,
		image.get_width() as u32,
		image.get_height() as u32,
		image::ColorType::Rgba8).unwrap();
}

fn main() {
	let shader = MyShader::new();
	let w = 512;
	let h = w;
	let mut pt = PathTracer::new(
		shader,
		Buffer::<Float3>::new(w,h,float3![0,0,0]));
	loop {
		pt.render();
		let post_image = pt.postprocess();
		save_image(&post_image, Path::new("image.bmp"));
		println!("iteration {} saved", pt.iteration());
	}
}
