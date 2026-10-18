// Package signclassification runs exported CLIP classifiers without Python.
// Initialize ONNX Runtime once per process before New, and destroy it after Close.
package signclassification

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type Manifest struct {
	FormatVersion int         `json:"format_version"`
	ModelFile     string      `json:"model_file"`
	InputName     string      `json:"input_name"`
	OutputName    string      `json:"output_name"`
	ImageSize     int         `json:"image_size"`
	Preprocessing string      `json:"preprocessing"`
	Mean          []float64   `json:"mean"`
	Std           []float64   `json:"std"`
	Classes       []string    `json:"classes"`
	References    []Reference `json:"references"`
}

type Reference struct {
	Path          string    `json:"path"`
	Probabilities []float64 `json:"probabilities"`
}

func LoadManifest(dir string) (Manifest, error) {
	var m Manifest
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return m, err
	}
	if err = json.Unmarshal(raw, &m); err != nil {
		return m, err
	}
	if m.FormatVersion != 1 || m.Preprocessing != "exif_alpha_white_square_pad_bicubic_v1" || m.ImageSize != 224 || len(m.Classes) < 2 || len(m.Mean) != 3 || len(m.Std) != 3 || m.InputName == "" || m.OutputName == "" || m.ModelFile == "" || filepath.Base(m.ModelFile) != m.ModelFile {
		return m, fmt.Errorf("unsupported or invalid manifest")
	}
	seen := make(map[string]bool)
	for _, c := range m.Classes {
		if c == "" || seen[c] {
			return m, fmt.Errorf("empty or duplicate class: %q", c)
		}
		seen[c] = true
	}
	for i := range m.Mean {
		if math.IsNaN(m.Mean[i]) || math.IsInf(m.Mean[i], 0) || math.IsNaN(m.Std[i]) || math.IsInf(m.Std[i], 0) || m.Std[i] <= 0 {
			return m, fmt.Errorf("invalid normalization")
		}
	}
	return m, nil
}

type Candidate struct {
	ClassName   string  `json:"class_name"`
	Probability float32 `json:"probability"`
}
type Result struct {
	Path      string      `json:"path"`
	ClassName string      `json:"class_name"`
	Accepted  bool        `json:"accepted"`
	TopK      []Candidate `json:"top_k"`
}

type Classifier struct {
	manifest Manifest
	session  *ort.DynamicAdvancedSession
	mu       sync.Mutex
}

func New(modelDir string) (*Classifier, error) {
	m, err := LoadManifest(modelDir)
	if err != nil {
		return nil, err
	}
	s, err := ort.NewDynamicAdvancedSession(filepath.Join(modelDir, m.ModelFile), []string{m.InputName}, []string{m.OutputName}, nil)
	if err != nil {
		return nil, fmt.Errorf("load ONNX model: %w", err)
	}
	return &Classifier{manifest: m, session: s}, nil
}

// Close is idempotent and waits for an in-flight prediction.
func (c *Classifier) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session == nil {
		return nil
	}
	err := c.session.Destroy()
	c.session = nil
	return err
}

func (c *Classifier) Predict(path string, topK int, minProbability float32) (Result, error) {
	var empty Result
	if topK < 1 || math.IsNaN(float64(minProbability)) || minProbability < 0 || minProbability > 1 {
		return empty, fmt.Errorf("topK must be positive and threshold in [0,1]")
	}
	data, err := preprocess(path, c.manifest)
	if err != nil {
		return empty, fmt.Errorf("preprocess %s: %w", path, err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session == nil {
		return empty, fmt.Errorf("classifier is closed")
	}
	input, err := ort.NewTensor(ort.NewShape(1, 3, int64(c.manifest.ImageSize), int64(c.manifest.ImageSize)), data)
	if err != nil {
		return empty, err
	}
	defer input.Destroy()
	output, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(c.manifest.Classes))))
	if err != nil {
		return empty, err
	}
	defer output.Destroy()
	if err = c.session.Run([]ort.Value{input}, []ort.Value{output}); err != nil {
		return empty, fmt.Errorf("ONNX inference: %w", err)
	}
	return rank(path, c.manifest.Classes, output.GetData(), topK, minProbability)
}

func rank(path string, classes []string, probabilities []float32, topK int, threshold float32) (Result, error) {
	if len(classes) != len(probabilities) || len(classes) == 0 || topK < 1 {
		return Result{}, fmt.Errorf("invalid prediction shape or topK")
	}
	candidates := make([]Candidate, len(classes))
	sum := float64(0)
	for i, p := range probabilities {
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) || p < 0 || p > 1 {
			return Result{}, fmt.Errorf("invalid probability")
		}
		sum += float64(p)
		candidates[i] = Candidate{classes[i], p}
	}
	if math.Abs(sum-1) > 1e-3 {
		return Result{}, fmt.Errorf("probabilities do not sum to one")
	}
	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].Probability > candidates[j].Probability })
	if topK > len(candidates) {
		topK = len(candidates)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return Result{}, err
	}
	return Result{abs, candidates[0].ClassName, candidates[0].Probability >= threshold, candidates[:topK]}, nil
}
