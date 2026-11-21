//! Texturas de imagen (framebuffers) y relación píxel/pantalla exacta (GOG-111).
//!
//! - [`Texture`]: imagen validada `RGBA8` o `Indexed8` (paleta RGBA). Los bytes se guardan **sin
//!   modificar**; [`Texture::pixel_hash`] (FNV-1a 64) permite comprobarlo.
//! - [`ImageViewport`]: política de ajuste imagen -> ventana.
//!
//! # Política de escala (documentada)
//! 1. `s = floor(min(vw/iw, vh/ih))`. Si `s >= 1` la escala es **entera** (pixel-perfect) y el
//!    desplazamiento (letterbox/pillarbox) es **entero**: `off = floor((v - i*s)/2)`. Las barras
//!    quedan fuera del rectángulo de imagen.
//! 2. Si la ventana es menor que la imagen en algún eje (`s == 0`) se usa la escala fraccionaria
//!    `min(vw/iw, vh/ih)` centrada; no es pixel-exacta (`is_integer_scale() == false`).
//!
//! Con escala entera, `image -> screen -> image` y `screen -> image -> screen` son exactos en
//! `f64` para coordenadas de píxel enteras y para cualquier punto de pantalla múltiplo de `s`.

use std::fmt::Write;

use crate::error::ProjectionError;

/// Límite por eje para validar texturas.
pub const MAX_TEXTURE_DIM: u32 = 4096;

/// Formato de píxel de una [`Texture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 4 bytes por píxel (R, G, B, A).
    Rgba8,
    /// 1 byte por píxel: índice en la paleta.
    Indexed8,
}

impl PixelFormat {
    /// Nombre canónico estable.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rgba8 => "rgba8",
            Self::Indexed8 => "indexed8",
        }
    }
}

/// Imagen validada, fila a fila, origen arriba-izquierda.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Texture {
    width: u32,
    height: u32,
    format: PixelFormat,
    palette: Vec<[u8; 4]>,
    pixels: Vec<u8>,
}

fn invalid(reason: &'static str) -> ProjectionError {
    ProjectionError::InvalidTexture { reason }
}

fn check_dims(width: u32, height: u32) -> Result<usize, ProjectionError> {
    if width == 0 || height == 0 {
        return Err(invalid("zero dimension"));
    }
    if width > MAX_TEXTURE_DIM || height > MAX_TEXTURE_DIM {
        return Err(invalid("dimension too large"));
    }
    Ok(width as usize * height as usize)
}

impl Texture {
    /// Textura `RGBA8`; `pixels.len()` debe ser `w*h*4`.
    pub fn rgba8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, ProjectionError> {
        let n = check_dims(width, height)?;
        if pixels.len() != n * 4 {
            return Err(invalid("pixel buffer length mismatch"));
        }
        Ok(Self {
            width,
            height,
            format: PixelFormat::Rgba8,
            palette: Vec::new(),
            pixels,
        })
    }

    /// Textura indexada; `pixels.len() == w*h`, paleta de 1..=256 entradas y todos los índices
    /// dentro de la paleta.
    pub fn indexed8(
        width: u32,
        height: u32,
        palette: Vec<[u8; 4]>,
        pixels: Vec<u8>,
    ) -> Result<Self, ProjectionError> {
        let n = check_dims(width, height)?;
        if pixels.len() != n {
            return Err(invalid("pixel buffer length mismatch"));
        }
        if palette.is_empty() || palette.len() > 256 {
            return Err(invalid("palette must have 1..=256 entries"));
        }
        if pixels.iter().any(|&i| usize::from(i) >= palette.len()) {
            return Err(invalid("pixel index outside palette"));
        }
        Ok(Self {
            width,
            height,
            format: PixelFormat::Indexed8,
            palette,
            pixels,
        })
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn format(&self) -> PixelFormat {
        self.format
    }

    /// Paleta (vacía en `Rgba8`).
    pub fn palette(&self) -> &[[u8; 4]] {
        &self.palette
    }

    /// Bytes originales, sin modificar.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// FNV-1a 64 de [`Texture::pixels`] (mismo algoritmo que el hash de frames del emulador).
    pub fn pixel_hash(&self) -> u64 {
        self.pixels.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// Color RGBA del píxel `(x, y)`, o `None` fuera de rango.
    pub fn rgba_at(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = y as usize * self.width as usize + x as usize;
        match self.format {
            PixelFormat::Indexed8 => self.palette.get(usize::from(self.pixels[i])).copied(),
            PixelFormat::Rgba8 => {
                let p = &self.pixels[i * 4..i * 4 + 4];
                Some([p[0], p[1], p[2], p[3]])
            }
        }
    }

    pub(crate) fn write_canonical_json(&self, out: &mut String) {
        let _ = write!(
            out,
            "{{\"format\":\"{}\",\"width\":{},\"height\":{},\"hash\":\"{:016x}\",\"palette\":\"",
            self.format.as_str(),
            self.width,
            self.height,
            self.pixel_hash()
        );
        for c in &self.palette {
            for b in c {
                let _ = write!(out, "{b:02x}");
            }
        }
        out.push_str("\",\"pixels\":\"");
        for b in &self.pixels {
            let _ = write!(out, "{b:02x}");
        }
        out.push_str("\"}");
    }
}

/// Mapeo imagen <-> ventana según la política de escala del módulo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageViewport {
    image_w: u32,
    image_h: u32,
    view_w: u32,
    view_h: u32,
    scale: f64,
    off_x: f64,
    off_y: f64,
    integer: bool,
}

impl ImageViewport {
    /// Calcula el ajuste. Todas las dimensiones deben ser `> 0`.
    pub fn fit(
        image_w: u32,
        image_h: u32,
        view_w: u32,
        view_h: u32,
    ) -> Result<Self, ProjectionError> {
        if image_w == 0 || image_h == 0 || view_w == 0 || view_h == 0 {
            return Err(invalid("zero dimension in viewport fit"));
        }
        let (iw, ih, vw, vh) = (
            u64::from(image_w),
            u64::from(image_h),
            u64::from(view_w),
            u64::from(view_h),
        );
        let s_int = (vw / iw).min(vh / ih);
        let (scale, off_x, off_y, integer) = if s_int >= 1 {
            let s = s_int as f64;
            (
                s,
                ((vw - iw * s_int) / 2) as f64,
                ((vh - ih * s_int) / 2) as f64,
                true,
            )
        } else {
            let s = (f64::from(view_w) / f64::from(image_w))
                .min(f64::from(view_h) / f64::from(image_h));
            (
                s,
                (f64::from(view_w) - f64::from(image_w) * s) / 2.0,
                (f64::from(view_h) - f64::from(image_h) * s) / 2.0,
                false,
            )
        };
        Ok(Self {
            image_w,
            image_h,
            view_w,
            view_h,
            scale,
            off_x,
            off_y,
            integer,
        })
    }

    /// Píxeles de pantalla por píxel de imagen.
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// `true` si la escala es entera (pixel-perfect).
    pub const fn is_integer_scale(&self) -> bool {
        self.integer
    }

    /// Desplazamiento `(x, y)` de la esquina superior-izquierda de la imagen en pantalla.
    pub const fn offset(&self) -> (f64, f64) {
        (self.off_x, self.off_y)
    }

    /// Tamaño de ventana.
    pub const fn view_size(&self) -> (u32, u32) {
        (self.view_w, self.view_h)
    }

    /// Coordenadas de imagen (px, fraccionarias permitidas) -> pantalla.
    pub fn image_to_screen(&self, x: f64, y: f64) -> (f64, f64) {
        (self.off_x + x * self.scale, self.off_y + y * self.scale)
    }

    /// Pantalla -> coordenadas de imagen (inversa exacta de [`Self::image_to_screen`]).
    pub fn screen_to_image(&self, sx: f64, sy: f64) -> (f64, f64) {
        (
            (sx - self.off_x) / self.scale,
            (sy - self.off_y) / self.scale,
        )
    }

    /// Píxel de imagen bajo un punto de pantalla; `None` en las barras (letterbox).
    pub fn screen_to_pixel(&self, sx: f64, sy: f64) -> Option<(u32, u32)> {
        let (x, y) = self.screen_to_image(sx, sy);
        if x < 0.0 || y < 0.0 || x >= f64::from(self.image_w) || y >= f64::from(self.image_h) {
            return None;
        }
        Some((x.floor() as u32, y.floor() as u32))
    }

    /// Rectángulo de imagen `(x, y, w, h)` -> rectángulo de pantalla.
    pub fn image_rect_to_screen(&self, x: f64, y: f64, w: f64, h: f64) -> (f64, f64, f64, f64) {
        let (sx, sy) = self.image_to_screen(x, y);
        (sx, sy, w * self.scale, h * self.scale)
    }

    /// Rectángulo que ocupa la imagen completa en pantalla.
    pub fn image_screen_rect(&self) -> (f64, f64, f64, f64) {
        self.image_rect_to_screen(0.0, 0.0, f64::from(self.image_w), f64::from(self.image_h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_validation() {
        assert!(Texture::rgba8(0, 1, vec![]).is_err());
        assert!(Texture::rgba8(1, 1, vec![0; 3]).is_err());
        assert!(Texture::indexed8(2, 1, vec![[0; 4]], vec![0, 1]).is_err());
        assert!(Texture::indexed8(2, 1, vec![], vec![0, 0]).is_err());
        let t = Texture::indexed8(2, 1, vec![[1, 2, 3, 4], [5, 6, 7, 8]], vec![1, 0]).unwrap();
        assert_eq!(t.rgba_at(0, 0), Some([5, 6, 7, 8]));
        assert_eq!(t.rgba_at(2, 0), None);
        assert_eq!(t.pixels(), &[1, 0]);
    }

    #[test]
    fn integer_scale_letterbox_policy() {
        let v = ImageViewport::fit(32, 32, 100, 70).unwrap();
        assert!(v.is_integer_scale());
        assert_eq!(v.scale(), 2.0);
        assert_eq!(v.offset(), (18.0, 3.0));
        assert_eq!(v.image_to_screen(5.0, 7.0), (28.0, 17.0));
        assert_eq!(v.screen_to_pixel(0.0, 0.0), None);
        assert_eq!(v.screen_to_pixel(18.0, 3.0), Some((0, 0)));
    }

    #[test]
    fn smaller_window_falls_back_to_fractional() {
        let v = ImageViewport::fit(32, 32, 16, 20).unwrap();
        assert!(!v.is_integer_scale());
        assert_eq!(v.scale(), 0.5);
    }
}
