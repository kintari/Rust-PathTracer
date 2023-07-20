pub struct Buffer<T> {
	width: usize,
	height: usize,
	pixels: Vec<T>
}

impl<T: Copy> Buffer<T> {

	pub fn new(width: usize, height: usize, value: T) -> Buffer<T> {
		let count = width * height;
		return Buffer {
			width,
			height,
			pixels: vec![value; count]
		};
	}

	pub fn fill<F>(&mut self, mut f: F)
		where F: FnMut(usize,usize,&T) -> T
	{
		for i in 0..self.width {
			for j in 0..self.height {
				let index = self.width * j + i;
				self.pixels[index] = f(i, j, &self.pixels[index]);
			}
		}
	}

	pub fn map<F,R>(&self, f: F) -> Buffer<R>
		where F: FnMut(&T) -> R
	{
		let pixels: Vec<_> = self.get_pixels().iter().map(f).collect();
		return Buffer {
			width: self.get_width(),
			height: self.get_height(),
			pixels: pixels
		};
	}

	pub fn get_width(&self) -> usize {
		return self.width;
	}

	pub fn get_height(&self) -> usize {
		return self.height;
	}

	pub fn get_pixels(&self) -> &Vec<T> {
		return &self.pixels;
	}

}

impl<T: Copy> Clone for Buffer<T> {
	fn clone(&self) -> Self {
		return Buffer {
			width: self.width,
			height: self.height,
			pixels: self.pixels.to_vec()
		}
	}
}