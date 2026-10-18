package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	signs "github.com/xinge1982/traffic-assets-ai/sign_classification/go"
	ort "github.com/yalue/onnxruntime_go"
)

func run() error {
	modelDir := flag.String("model-dir", "../output/onnx", "exported ONNX model directory")
	input := flag.String("input", "", "image or image directory")
	runtime := flag.String("ort-lib", "", "ONNX Runtime DLL/.so absolute path")
	topK := flag.Int("top-k", 5, "number of candidates")
	threshold := flag.Float64("min-probability", 0, "acceptance threshold")
	output := flag.String("output", "", "optional output JSON file")
	verify := flag.Bool("verify", false, "check Python reference images from manifest")
	tolerance := flag.Float64("tolerance", 0.02, "maximum Go/Python probability difference for -verify")
	flag.Parse()
	if *topK < 1 || math.IsNaN(*threshold) || *threshold < 0 || *threshold > 1 || math.IsNaN(*tolerance) || math.IsInf(*tolerance, 0) || *tolerance < 0 {
		return fmt.Errorf("invalid top-k, min-probability or tolerance")
	}
	if *input == "" && !*verify {
		return fmt.Errorf("-input is required unless using -verify")
	}
	if *runtime != "" {
		ort.SetSharedLibraryPath(*runtime)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}
	defer ort.DestroyEnvironment()
	classifier, err := signs.New(*modelDir)
	if err != nil {
		return err
	}
	defer classifier.Close()
	if *verify {
		manifest, err := signs.LoadManifest(*modelDir)
		if err != nil {
			return err
		}
		if len(manifest.References) == 0 {
			return fmt.Errorf("export with --verify-image first")
		}
		for _, reference := range manifest.References {
			if len(reference.Probabilities) != len(manifest.Classes) {
				return fmt.Errorf("invalid reference: %s", reference.Path)
			}
			result, err := classifier.Predict(reference.Path, len(manifest.Classes), 0)
			if err != nil {
				return err
			}
			got := make(map[string]float64)
			for _, p := range result.TopK {
				got[p.ClassName] = float64(p.Probability)
			}
			maxDiff := float64(0)
			for i, name := range manifest.Classes {
				maxDiff = math.Max(maxDiff, math.Abs(got[name]-reference.Probabilities[i]))
			}
			fmt.Fprintf(os.Stderr, "%s: maximum probability difference %.8f\n", reference.Path, maxDiff)
			if maxDiff > *tolerance {
				return fmt.Errorf("Go/Python parity failed (tolerance %.8f)", *tolerance)
			}
		}
		if *input == "" {
			return nil
		}
	}
	paths, err := signs.ImagePaths(*input)
	if err != nil {
		return err
	}
	results := make([]signs.Result, 0, len(paths))
	for _, path := range paths {
		result, err := classifier.Predict(path, *topK, float32(*threshold))
		if err != nil {
			return err
		}
		results = append(results, result)
	}
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	if *output != "" {
		if err = os.WriteFile(*output, append(data, '\n'), 0644); err != nil {
			return err
		}
	}
	fmt.Println(string(data))
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
