package signclassification

import (
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
)

func testManifest() Manifest {
	return Manifest{ImageSize: 224, Mean: []float64{0.48145466, 0.4578275, 0.40821073}, Std: []float64{0.26862954, 0.26130258, 0.27577711}}
}

func TestPythonPreprocessing(t *testing.T) {
	// Golden PNGs were produced by common.load_image + Pillow BICUBIC.
	for _, name := range []string{"alpha", "downsample"} {
		actual, err := preprocess(filepath.Join("testdata", name+".png"), testManifest())
		if err != nil {
			t.Fatal(err)
		}
		expected, err := imaging.Open(filepath.Join("testdata", name+"_pillow.png"))
		if err != nil {
			t.Fatal(err)
		}
		maxDiff := float64(0)
		for y := 0; y < 224; y++ {
			for x := 0; x < 224; x++ {
				p := color.NRGBAModel.Convert(expected.At(x, y)).(color.NRGBA)
				for ch, v := range []uint8{p.R, p.G, p.B} {
					want := (float64(v)/255 - testManifest().Mean[ch]) / testManifest().Std[ch]
					maxDiff = math.Max(maxDiff, math.Abs(float64(actual[ch*224*224+y*224+x])-want))
				}
			}
		}
		// Pillow uses fixed-point coefficient rounding; allow at most one pixel level.
		if maxDiff > 1.01/(255*0.26130258) {
			t.Fatalf("%s normalized max difference: %g", name, maxDiff)
		}
	}
}

func TestPaddingAndNormalization(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 224, 112))
	for y := 0; y < 112; y++ {
		for x := 0; x < 224; x++ {
			img.SetNRGBA(x, y, color.NRGBA{255, 0, 0, 255})
		}
	}
	m := testManifest()
	data := preprocessImage(img, m)
	for ch := 0; ch < 3; ch++ {
		want := (float32(1) - float32(m.Mean[ch])) / float32(m.Std[ch])
		if data[ch*224*224] != want {
			t.Fatal("padding must be white")
		}
	}
	if data[112*224+112] <= 0 || data[224*224+112*224+112] >= 0 {
		t.Fatal("RGB channels or CHW layout incorrect")
	}
}

func TestRank(t *testing.T) {
	r, err := rank("a.png", []string{"禁止左转", "禁止右转"}, []float32{0.2, 0.8}, 5, 0.9)
	if err != nil || r.ClassName != "禁止右转" || r.Accepted || len(r.TopK) != 2 {
		t.Fatalf("%+v %v", r, err)
	}
	if _, err = rank("a.png", []string{"a", "b"}, []float32{float32(math.NaN()), 0}, 1, 0); err == nil {
		t.Fatal("NaN accepted")
	}
}

func TestONNXInference(t *testing.T) {
	library := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	if library == "" {
		t.Skip("set ONNXRUNTIME_SHARED_LIBRARY_PATH for native ONNX integration")
	}
	ort.SetSharedLibraryPath(library)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer ort.DestroyEnvironment()
	c, err := New("testdata/tiny")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	r, err := c.Predict("testdata/alpha.png", 2, 0.5)
	if err != nil || r.ClassName != "B" || !r.Accepted || math.Abs(float64(r.TopK[0].Probability)-0.8) > 1e-6 {
		t.Fatalf("%+v %v", r, err)
	}
	if err = c.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = c.Predict("testdata/alpha.png", 1, 0); err == nil {
		t.Fatal("prediction after Close succeeded")
	}
}
