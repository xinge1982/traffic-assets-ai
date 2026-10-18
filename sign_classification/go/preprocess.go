package signclassification

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"strings"

	"github.com/disintegration/imaging"
	_ "golang.org/x/image/webp"
)

// Decode corrects JPEG EXIF orientation. The exported model expects RGB CHW.
func preprocess(path string, m Manifest) ([]float32, error) {
	img, err := imaging.Open(path, imaging.AutoOrientation(true))
	if err != nil {
		return nil, err
	}
	return preprocessImage(img, m), nil
}

func preprocessImage(img image.Image, m Manifest) []float32 {
	b := img.Bounds()
	side := b.Dx()
	if b.Dy() > side {
		side = b.Dy()
	}
	canvas := image.NewNRGBA(image.Rect(0, 0, side, side))
	for i := range canvas.Pix {
		canvas.Pix[i] = 255
	}
	x, y := (side-b.Dx())/2, (side-b.Dy())/2
	for iy := 0; iy < b.Dy(); iy++ {
		for ix := 0; ix < b.Dx(); ix++ {
			p := color.NRGBAModel.Convert(img.At(b.Min.X+ix, b.Min.Y+iy)).(color.NRGBA)
			a := int(p.A)
			blend := func(v uint8) uint8 { return uint8((int(v)*a + 255*(255-a) + 127) / 255) }
			canvas.SetNRGBA(x+ix, y+iy, color.NRGBA{blend(p.R), blend(p.G), blend(p.B), 255})
		}
	}
	// Catmull-Rom is a=-0.5 bicubic with antialiasing when downsampling.
	resized := imaging.Resize(canvas, m.ImageSize, m.ImageSize, imaging.CatmullRom)
	plane := m.ImageSize * m.ImageSize
	data := make([]float32, 3*plane)
	for y := 0; y < m.ImageSize; y++ {
		for x := 0; x < m.ImageSize; x++ {
			pixel := resized.NRGBAAt(x, y)
			for ch, v := range []uint8{pixel.R, pixel.G, pixel.B} {
				data[ch*plane+y*m.ImageSize+x] = (float32(v)/255 - float32(m.Mean[ch])) / float32(m.Std[ch])
			}
		}
	}
	return data
}

func ImagePaths(input string) ([]string, error) {
	info, err := os.Stat(input)
	if err != nil {
		return nil, err
	}
	supported := func(path string) bool {
		switch strings.ToLower(filepath.Ext(path)) {
		case ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp":
			return true
		}
		return false
	}
	if !info.IsDir() {
		if !supported(input) {
			return nil, fmt.Errorf("unsupported image: %s", input)
		}
		return []string{input}, nil
	}
	var paths []string
	err = filepath.WalkDir(input, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && supported(path) {
			paths = append(paths, path)
		}
		return nil
	})
	if err == nil && len(paths) == 0 {
		err = fmt.Errorf("no supported images in %s", input)
	}
	return paths, err
}
