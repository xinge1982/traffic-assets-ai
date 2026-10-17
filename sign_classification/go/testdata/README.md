Test fixtures only, not trained traffic sign models.

- `alpha.png`: synthetic 29×17 RGBA pattern, to test white compositing and upsampling.
- `downsample.png`: synthetic 317×251 RGBA pattern, to test antialiased downsampling.
- `*_pillow.png`: `common.load_image(path).resize((224, 224), Image.Resampling.BICUBIC)`.
- `tiny/sign_classifier.onnx`: ONNX opset 17 Constant graph, input images `[1,3,224,224]`, output probabilities `[[0.2,0.8]]`. Used only to exercise the real Go/native Runtime boundary and lifecycle; no classification accuracy claim.

Source pattern for both images, where x/y are pixel column/row:
`R=(17*x+3*y)%256; G=(11*x+23*y)%256; B=(7*x+13*y)%256; A=(19*x+29*y)%256`.
