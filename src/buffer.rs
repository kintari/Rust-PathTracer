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

	pub fn width(&self) -> usize {
		return self.width;
	}

	pub fn height(&self) -> usize {
		return self.height;
	}

	pub fn pixels(&self) -> &Vec<T> {
		return &self.pixels;
	}

}