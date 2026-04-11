# Detecting Unsafe Surfaces on Mars  Surfaces using Computer Vision
We are attempting to determine unsafe surfaces for the Mars Rover to land on using computationally efficient methods in Computer Vision.

# Part 1: Finding Unsafe Surfaces by Slopes
This Program tries to mark each pixel with a safety score to detect Cliffs, Shadows, Craters, etc. which are unsafe to land on for a Space Rover. We do this using a 2 Step Process:

- Gradient Scoring
- Contouring for Large Shadows

## Gradient Scoring
We calculate the Sobel Gradients of each Pixel using a Kernel Size of 3 and Blurring Kernel Size of 5. Found algorithm from [1].

$$ g(x, y) = \sqrt{I_x^2 + I_y^2} $$

$$ G(x, y) = \min \left( 1,\ \frac{g(x, y)}{g_{99}} \right) $$

## Contouring for finding Large Shadows
The Algorithm was ignoring Large Shadows and would mark it as safe even though these were part of craters or cliffs. We used Contouring and tried to find areas with very dark pixels of area >= 500 pixels. This sufficiently marked large unsafe areas as small shadow areas were marked as unsafe by gradient scoring already.

## Sample Results

![Sample Output 1](./GradientScoring/Output/Output1.png)
![Sample Output 2](./GradientScoring/Output/Output2.png)
![Sample Output 3](./GradientScoring/Output/Output3.png)

## Citation
[1] J. He, H. Cui, and J. Feng, “Edge information based crater detection and matching for lunar exploration,” in Proc. International Conference on Intelligent Control and Information Processing, Dalian, China, Aug. 13–15, 2010, pp. 302–307

# Part 2: Estimating Surface Hardness
We found Cliffs, Craters, etc. previously but we do not know whether the land given is hard or soft. We use Fast-Fourier Transform to estimate if a surface is soft or hard.

Hard surfaces, such as exposed bedrock or boulder fields, introduce "high-frequency noise." These sharp edges create spikes in the high-frequency spectrum. A smooth, soft surface like a dust mantle or a sand dune has very few sharp changes in brightness. Mathematically, this means the "signal" of the image is dominated by low spatial frequencies. Hard surfaces, such as exposed bedrock or boulder fields, introduce "high-frequency noise." Building upon this theory, I have manually set a tunable ratio threshold of 1 for a surface to be hard.

## Sample Results
![Sample Outputs Stacked](./TerrainClassification/prediction_output_stacked.png)

We can see that the process works generally on both one-type hardness/softness dominated surface or a uniform mixture of both.
