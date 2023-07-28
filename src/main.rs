#![feature(fn_traits)]
#![feature(unboxed_closures)]

#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unreachable_code)]
#![allow(unused_macros)]

mod buffer;
mod color;
mod float2;
mod float3;
mod random;
mod ray;
mod sdf;

use buffer::*;
use color::*;
use float2::*;
use float3::*;
use random::Random;
use ray::Ray;
use sdf::Sdf;

use std::path::{Path,PathBuf};
use std::fs::File;
use std::io::BufWriter;
use std::f32::consts::PI;
use std::ops::*;
use std::cmp::{min,max,Ordering};
use std::iter::zip;
use std::thread;
use std::sync::mpsc::channel;
use std::time::Duration;


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

struct Scene {
	sphere: sdf::Sphere,
	walls: [sdf::Plane;6],
	light: sdf::Box,
	camera_pos: Float3
}

impl Scene {
	
	fn new() -> Scene {
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
		//let half_dims = float3![dx, dy, dz];
		let camera_pos = float3![0,0,-20];
		return Scene { sphere, walls, light, camera_pos };
	}

	fn distance(&self, p: Float3) -> f32 {
		/*
		let d_walls = self.walls.iter()
			.map(|sdf| sdf.distance(p))
			.reduce(|a,x| f32::min(a,x))
			.unwrap();
		*/
		let d_sphere = self.sphere.distance(p);
		let d_light = self.light.distance(p);
		let d_walls = self.walls.iter()
			.map(|sdf| sdf.distance(p))
			.reduce(|a,x| f32::min(a,x))
			.unwrap();
		return f32::min(d_sphere, f32::min(d_light, d_walls));
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
		if self.walls[0].distance(p) < d {
			return float3![1,0.4,0.4];
		}
		else if self.walls[1].distance(p) < d {
			return float3![0.4,1,0.4];
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
		let x = self.distance(p+dx)-self.distance(p-dx);
		let y = self.distance(p+dy)-self.distance(p-dy);
		let z = self.distance(p+dz)-self.distance(p-dz);
		return normalize(float3![x, y, z]);
	}

}

struct Renderer<'a> {
	scene: &'a Scene,
	random: Random
}

impl<'a> Renderer<'a> {

	fn new(scene: &Scene) -> Renderer {
		return Renderer { scene, random: Random::new() };
	}
	
	fn ray_march(&self, ray: Ray, contact_distance: f32) -> Option<Hit> {
		let mut t = 0.0;
		let mut i = 0;
		while i < 250 {
			let p = ray(t);
			let d = self.scene.distance(p);
			if d < contact_distance {
				let normal = self.scene.calc_normal(p);
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

	fn shader(&self, camera_ray: Ray) -> Float3 {
		let contact_distance = 1.0 / 1024.0;
		let mut ray = camera_ray;
		let mut Kd = float3![1];
		let mut I = float3![0];
		let roughness = 0.5;
		let mut iterations = 6;
		while iterations > 0 {
			if let Some(hit) = self.ray_march(ray, contact_distance) {
				I += Kd * self.scene.emissive(hit.location);
				Kd *= self.scene.diffuse_color(hit.location);
				let d = normalize(lerp(
					hit.normal,
					self.random.cosine_weighted(hit.normal),
					roughness));
				let mut offset = 2.0 * 0.001 * self.random.unit_vector();
				if dot(offset,d) < 0.0 {
					offset = offset * -1.0;
				}
				let p = offset + hit.location + 8.0 * contact_distance * hit.normal;
				ray = Ray { p, d };
				iterations -= 1;
			}
			else {
				//I += Kd * self.diffuse_color(P) * self.env(D);
				break;
			}
		}
		return I;
	} // main
	
	fn camera_ray(&self, uv: Float2) -> Ray {
		let clip = 2.0*uv-1.0;
		let p = self.scene.camera_pos;
		let d = normalize(float3![clip.x, clip.y, 2.4]);
		return Ray { p, d };
	}
	
	fn render(&self, image: &mut Buffer<Float3>) {
		let camera_pos = float3![0,0,-20];
		let w = image.get_width();
		let h = image.get_height();
		let resolution = float2![w-1,h-1];
		image.fill(|i,j,&prev_color| {
			let jitter = float2![self.random.uniform_in_range(-0.5,0.5), self.random.uniform_in_range(-0.5,0.5)];
			let u = i as f32;
			let v = resolution.y-(j as f32);
			let uv = (jitter+float2![u,v]) / resolution;
			let ray = self.camera_ray(uv);
			return prev_color + self.shader(ray);
		});
	}

}

struct Compositor {
	pub buffer: Buffer::<Float3>,
	divisor: f32
}

impl Compositor {
	fn new(w: usize, h: usize) -> Compositor {
		return Compositor {
			buffer: Buffer::<Float3>::new(w, h, float3![0]),
			divisor: 0.0
		};
	}
	fn composite(&mut self, buffer: &Buffer<Float3>) {
		self.buffer.combine::<Float3>(
			buffer, |c0, c1| { c0 + c1 });
		self.divisor += 1.0;
	}
}

fn main() {

	let exposure = 0.0;
	let image_size = 128;
	let samples_per_pixel = 16;

	let scene = Scene::new();
	let mut compositor = Compositor::new(image_size, image_size);

	let num_threads = 4;
	let (tx, rx) = channel();

	thread::scope(|s| {
		for i in 0..num_threads {
			let tx = tx.clone();
			let scene_ref = &scene;
			s.spawn(move || {
				loop {
					let mut buffer = Buffer::<Float3>::new(image_size, image_size, float3![0]);
					let renderer = Renderer::new(scene_ref);	
					for sample in 0..samples_per_pixel {
						renderer.render(&mut buffer);
					}
					let b = buffer.map(|&c| c/(samples_per_pixel as f32));
					tx.send(b).unwrap();
				}
			});
			thread::sleep(Duration::from_millis(500));
		}
		loop {
			let buffer = rx.recv().unwrap();
			println!("got a buffer");
			compositor.composite(&buffer);
			let ldr = compositor.buffer
				.map(|&c| ACESFilm(c/compositor.divisor));
			save_image(Path::new("image.bmp"), &ldr);
		}
	});

	
	//let mut iteration = 1;

	/*
	loop {

		//println!("iteration {}", iteration);
		//println!("  rendering ...");
		

		//println!("  post-processing ...");

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
 */
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

fn ACESFilm(x: Float3) -> Float3 {
	let a = float3![2.51];
	let b = float3![0.03];
	let c = float3![2.43];
	let d = float3![0.59];
	let e = float3![0.14];
	return saturate((x*(a*x+b))/(x*(c*x+d)+e));
}
