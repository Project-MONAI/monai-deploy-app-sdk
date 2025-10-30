# Development Notes

## Implementation Notes for nnUNet MAP


* Initial Tests show volume and Dice agreement with Bundle, need to do more thorough testing.

1. For each model configuration the output gets written to .npz file by nnunet inference functions.

2. These file paths are then used by the EnsembleProbabilities Transform function to create the final output.

3. If nnunet postprocessing is used, use the largest connected component transform in the MAP. There could be minor differences in the implementation, will do thorough analysis later.

3. Need to better understand the use of "context" in compute and compute_impl as input arguments.

4. Investigate keeping the probabilities in the memory, to help with speedup.

5. Need to investigate the current traceability provisions in the operators implemented.


## Implementation Details

### Testing Strategy

Tests should be conducted to:
1. Compare MAP output with native nnUNet output
2. Measure performance (time, memory usage)
3. Validate with various input formats and sizes
4. Test error handling and edge cases


### nnUNet Integration

The current implementation relies on the nnUNet's native inference approach which outputs intermediate .npz files for each model configuration. While this works, it introduces file I/O overhead which could potentially be optimized.

### Ensemble Prediction Flow

1. Multiple nnUNet models (3d_fullres, 3d_lowres, 3d_cascade_fullres) are loaded
2. Each model performs inference separately
3. Results are written to temporary .npz files
4. EnsembleProbabilitiesToSegmentation transform reads these files
5. Final segmentation is produced by combining results

### Potential Optimizations

- Keep probability maps in memory instead of writing to disk
- Parallelize model inference where applicable
- Streamline the ensemble computation process

### Context Usage

The `context` parameter in `compute` and `compute_impl` functions appears to be used for storing and retrieving models. Further investigation is needed to fully understand how this context is managed and whether it's being used optimally.

### Traceability

Current traceability in the operators may need improvement. Consider adding:

- More detailed logging
- Performance metrics
- Input/output validation steps
- Error handling with informative messages

