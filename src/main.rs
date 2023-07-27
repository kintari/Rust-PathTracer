#![feature(fn_traits)]
#![feature(unboxed_closures)]

#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

mod buffer;
mod color;
mod float2;
mod float3;
mod random;
mod ray;
mod sdf;
mod shader;

use buffer::*;
use color::*;
use float2::*;
use float3::*;
use random::Random;
use ray::Ray;
use sdf::Sdf;
use shader::*;

use std::path::{Path,PathBuf};
use std::fs::File;
use std::io::BufWriter;
use std::f32::consts::PI;
use std::ops::*;
use std::cmp::{min,max,Ordering};
use std::iter::zip;


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
			return float3![2.0];
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
		let h = 1.0 / 1024.0;
		let dx = h*float3![1,0,0];
		let dy = h*float3![0,1,0];
		let dz = h*float3![0,0,1];
		let x = self.scene_distance(p+dx)-self.scene_distance(p-dx);
		let y = self.scene_distance(p+dy)-self.scene_distance(p-dy);
		let z = self.scene_distance(p+dz)-self.scene_distance(p-dz);
		return normalize(float3![x, y, z]);
	}
	
	fn ray_march(&self, ray: Ray, threshold: f32) -> Option<Hit> {
		let mut t = 0.0;
		let mut i = 0;
		while i < 250 {
			let p = ray(t);
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

}

impl Shader for MyShader {

	fn main(&self, camera_ray: Ray) -> Float3 {

		let mut ray = camera_ray;
		let mut Kd = float3![1];
		let roughness = 0.5;
		let mut I = float3![0];

		let mut iterations = 6;
		while iterations > 0 {
			if let Some(hit) = self.ray_march(ray, self.contact_distance) {
				I += Kd * self.emissive(hit.location);
				Kd *= self.diffuse_color(hit.location);
				let d = normalize(lerp(
					hit.normal,
					self.random.cosine_weighted(hit.normal),
					roughness));
				let mut offset = 2.0 * 0.001 * self.random.unit_vector();
				if dot(offset,d) < 0.0 {
					offset = offset * -1.0;
				}
				let p = offset + hit.location + 8.0 * self.contact_distance * hit.normal;
				ray = Ray { p, d };
			}
			else {
				//I += Kd * self.diffuse_color(P) * self.env(D);
				break;
			}
			iterations -= 1;
		}
		return I;
	} // main

}

fn quantize_to_8bits(value: f32) -> u8 {
	let i: i32 = f32::floor(value * 255.0) as i32;
	let i = i32::clamp(i, 0, 255);
	return i as u8;
}

fn save_image(path: &Path, image: &Buffer<Float3>) {
	let data: Vec<u8> = image.pixels().iter()
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

struct Renderer<T> {
	shader: T,
	random: Random,
	samples_per_pixel: usize,
	camera_pos: Float3
}

impl<T: Shader> Renderer<T> {

	fn render(&self, image: &mut Buffer<Float3>) {
		let w = image.get_width();
		let h = image.get_height();
		let resolution = float2![w-1,h-1];
		let fac = 1.0 / (self.samples_per_pixel as f32);
		let f = |i,j| {
			let mut I = float3![0];
			for _ in 0 .. self.samples_per_pixel {
				let jitter = float2![
					self.random.uniform_in_range(-0.5,0.5),
					self.random.uniform_in_range(-0.5,0.5)
				];
				let u = i as f32;
				let v = resolution.y-(j as f32);
				let uv = float2![u,v];
				let uv = (jitter+uv) / resolution;
				let ray = self.camera_ray(uv);
				I += fac * self.shader.main(ray);
			}
			return I;
		};
		image.fill(|i,j,&prev_color| prev_color + f(i,j));
	}

	fn camera_ray(&self, uv: Float2) -> Ray {
		let clip = 2.0*uv-1.0;
		let p = self.camera_pos;
		let d = normalize(float3![clip.x, clip.y, 2.4]);
		return Ray { p, d };
	}

}

fn ACESFilm(x: Float3) -> Float3 {
	let a = float3![2.51];
	let b = float3![0.03];
	let c = float3![2.43];
	let d = float3![0.59];
	let e = float3![0.14];
	return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}

fn main() {

	let shader = MyShader::new();
	let image_size = 128;
	let exposure = 0.0;

	let mut color_buffer = Buffer::<Float3>::new(image_size, image_size, float3![0]);
	let mut post_buffer = color_buffer.clone();

	let mut iteration = 1;
	
	let renderer = Renderer {
		shader,
		random: Random::new(),
		samples_per_pixel: 16,
		camera_pos: float3![0,0,-20]
	};

	loop {

		println!("iteration {}", iteration);
		println!("  rendering ...");
		
		renderer.render(&mut color_buffer);

		println!("  post-processing ...");

		let iter = zip(
			post_buffer.pixels_mut().iter_mut(),
			color_buffer.pixels().iter()
		);

		let scale = 1.0 / (iteration as f32);
		iter.for_each(|(dst, src)| {
			let e = f32::exp2(exposure);
			*dst = ACESFilm(*src*scale*e);
		});

		println!("  saving ...");

		save_image(Path::new("image.bmp"), &post_buffer);
		println!("  complete");

		iteration += 1;
	}
}
