#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(dead_code)]

mod color;
mod buffer;
mod numeric;
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

use numeric::*;

use random::Random;

macro_rules! cond {
	($test:expr,$true_expr:expr,$false_expr:expr) => {
		if $test { $true_expr } else { $false_expr }
	}
}

macro_rules! color {
	($r:expr,$g:expr,$b:expr,$a:expr) => {
		Color::new($r as f32, $g as f32, $b as f32, $a as f32)
	};
	($x:expr) => {
		Color::new($x as f32, $x as f32, $x as f32, $x as f32)
	};
}

struct Hit {
	distance: f32,
	location: Float3,
	normal: Float3,
}

struct MyShader {
	random: Random,
	sphere: sdf::Sphere,
	walls: [sdf::Plane;5],
	ray_depth: u32,
	half_dims: Float3
}

impl MyShader {

	fn new() -> Self {
		let dx = 6.0;
		let dy = 6.0;
		let dz = 4.0;
		let walls = [
			sdf::Plane::new(float3![-1,0,0], -dx),
			sdf::Plane::new(float3![ 1,0,0], -dx),
			sdf::Plane::new(float3![ 0,1,0], -dy),
			sdf::Plane::new(float3![0,-1,0], -dy),
			sdf::Plane::new(float3![0,0,-1], -dz)
		];
		return Self {
			random: Random::new(),
			sphere: sdf::Sphere::new(float3![0,-1,0], 2.0),
			walls,
			ray_depth: 0,
			half_dims: float3![dx, dy, dz]
		};
	}

	fn scene_distance(&self, p: Float3) -> f32 {
		let d_sphere = self.sphere.distance(p);
		let d_walls = self.walls.iter()
			.map(|sdf| sdf.distance(p))
			.reduce(|a,x| cond![a < x, a, x])
			.unwrap();
		let d = f32::min(d_sphere, d_walls);
		return d;
	}

	fn emissive(&self, p: Float3) -> Float3 {
		let light_size = 2.0;
		if f32::abs(p.x) < light_size && f32::abs(p.z) < light_size {
			if p.y + 0.01 > self.half_dims.y {
				return float3![1.0];
			}
		}
		return float3![0.0];
	}

	fn diffuse_color(&self, p: Float3) -> Float3 {
		let d = 0.01;
		if f32::abs(p.x) + d >= self.half_dims.x {
			return cond![
				p.x > 0.0,
				float3![1,0,0],
				float3![0,1,0]
			];
		}
		else {
			return float3![0.5];
		}
	}

	fn calc_normal(&self, p: Float3) -> Float3 {
		let h = 0.0005;
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
		while i < 250 {
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
	
	fn primary_ray(&self, pos: Float3, dir: Float3) -> Option<Hit> {
		return self.ray_march(pos, dir, 0.001);
	}
	
	fn unproject(&self, uv: Float3) -> Float3 {
		let clip = 2.0*uv-1.0;
		return normalize(float3![clip.x, clip.y, 2.4]);
	}
	
	fn env(&self, _v: Float3) -> Float3 {
		return float3![0.25];
	}

}

impl Shader for MyShader {

	fn main(&mut self, fragCoord: Float3, resolution: Float3) -> Float3 {
		let u =       fragCoord.x / resolution.x;
		let v = 1.0 - fragCoord.y / resolution.y;
		let camera = float3![0, 0, -20];
		let dir = self.unproject(float3![u, v, 0.0]);

		let mut P = camera;
		let mut D = dir;
		let mut Kd = float3![1];
		let mut I = float3![0];

		let mut iterations = 5;
		while iterations > 0 {
			if let Some(hit) = self.ray_march(P, D, 0.001) {
				I = I + Kd * self.emissive(hit.location);
				Kd = Kd * self.diffuse_color(hit.location);
				D = self.random.cosine_weighted(hit.normal);
				P = hit.location + 0.005 * D;
			}
			else {
				break;
			}
			iterations -= 1;
		}
		return I;
	}

} // impl Shader

struct PathTracer<T> {
	shader: T,
	image: Buffer<Color>,
}

impl<T: Shader> PathTracer<T> {
	
	fn new(shader: T, image: Buffer::<Color>) -> Self {
		return Self { shader, image };
	}

	fn render(&mut self, iterations: u32) {
		let resolution = float3![self.image.width(), self.image.height(), 0];
		let factor = 1.0 / (iterations as f32);
		let f = |i,j,color: &Color| {
			let frag_coord = float3![i, j, 0];
			let mut frag_color = float3![0];
			for _ in 0..iterations {
				frag_color = frag_color + factor * self.shader.main(frag_coord, resolution);
			}
			return Color::new(color.r+frag_color.x, color.g+frag_color.y, color.b+frag_color.z, 1.0);
		};
		self.image.fill(f);
	}

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

	fn save(&self, path: &Path) -> Result<(), png::EncodingError> {

		let file = File::create(path).unwrap();
		let ref mut output = BufWriter::new(file);

		let w = self.image.width();
		let h = self.image.height();

		let mut encoder = png::Encoder::new(output, w as u32, h as u32);
		encoder.set_color(png::ColorType::Rgba);
		encoder.set_depth(png::BitDepth::Eight);
		encoder.set_srgb(png::SrgbRenderingIntent::RelativeColorimetric);
	
		let mut writer = encoder.write_header().unwrap();

		let data: Vec<u8> = self.image.pixels().iter()
			.flat_map(|px| [px.r,px.g,px.b,px.a])
			.map(|x| f32::floor(Self::saturate(x)*255.5_f32) as u8)
			.collect();

		return writer.write_image_data(&data);
	}

}

fn main() {
	let shader = MyShader::new();
	let res = 512;
	let num_samples = 16;
	let mut pt = PathTracer::new(
		shader,
		Buffer::<Color>::new(res,res,color![0,0,0,0]));
	let mut i = 1;
	loop {
		pt.render(num_samples);
		pt.save(Path::new("image.png")).unwrap();
		println!("iteration {} saved", i);
		i += 1;
	}
}




/*
trait Shape {
	fn sdf(&self, p: Float3) -> f32;
	fn normal(&self, p: Float3) -> Float3;
}

struct Sphere {
	center: Float3,
	radius: f32
}

impl Shape for Sphere {
	fn sdf(&self, p: Float3) -> f32 {
		return length(self.center-p)-self.radius;
	}
	fn normal(&self, p: Float3) -> Float3 {
		return normalize(p-self.center);
	}
}

struct Plane {
	normal: Float3,
	distance: f32
}

impl Shape for Plane {
	fn sdf(&self, p: Float3) -> f32 {
		return Vector::dot(self.normal,p)-self.distance;
	}
	fn normal(&self, _: Float3) -> Float3 {
		return self.normal;
	}
}

struct Scene {
	shapes: Vec<Box<dyn Shape>>
}
*/