# Copyright 2021-2026 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm

# Set matplotlib to use non-interactive backend for containerized environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from monai.deploy.core import ConditionType, Fragment, Operator, OperatorSpec


class SegmentationZScoreOperator(Operator):
    """This operator computes z-scores and percentiles for segmentation metrics against normative data.
    
    Takes metrics from SegmentationMetricsOperator and compares them to age/sex-specific normative
    values stored in CSV files. Outputs z-scores, percentiles, and generates PDF visualization.
    
    Named Input:
        metrics_dict: Dictionary of metrics from SegmentationMetricsOperator
        patient_age: Patient age in years (float)
        patient_sex: Patient sex ("Male" or "Female")
        assets_path: Path to assets folder containing normative data CSVs
    
    Named Output:
        zscore_dict: Dictionary with z-scores and percentiles for each organ
        pdf_bytes: Bytes of PDF file with quantile curves and patient values (optional)
        zscore_text: Formatted text summary for DICOM SR (filtered organs)
        
    Notes:
        - Normative data CSVs should be organized in subfolders under assets_path.
        - Refer to examples/apps/cchmc_ped_abd_ct_seg_app/assets for expected structure.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        assets_path: Optional[str] = None,
        organ_name_mapping: Optional[Dict[str, str]] = None,
        generate_plots: bool = True,
        additional_metrics_map: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs
    ):
        """Create an instance for a containing application object.
        
        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            assets_path (str, optional): Path to assets folder with normative data. 
                If None, must be provided as input to compute().
            organ_name_mapping (dict, optional): Maps organ names from metrics_dict to asset subfolder names.
                Example: {"liver_volume": "liver", "liver_density": "liver_HU"}
                If None, uses exact matching.
            generate_plots (bool): If True, generates matplotlib visualization and outputs as PDF bytes. Default True.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.input_name_metrics = "metrics_dict"
        self.input_name_dcm_series = "study_selected_series_list"
        self.input_name_assets = assets_path 
        self.additional_metrics_map = additional_metrics_map
        
        self.output_name_zscore = "zscore_dict"
        self.output_name_pdf_bytes = "pdf_bytes"
        
        self.assets_path = assets_path
        self.organ_name_mapping = organ_name_mapping or {}
        self.generate_plots = generate_plots
        
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        
        spec.input(self.input_name_metrics)
        spec.input(self.input_name_dcm_series).condition(ConditionType.NONE)  # Optional input
        
        spec.output(self.output_name_zscore).condition(ConditionType.NONE)
        spec.output(self.output_name_pdf_bytes).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""
        
        # Receive inputs
        metrics_dict = op_input.receive(self.input_name_metrics)
        study_selected_series_list = op_input.receive(self.input_name_dcm_series)

    
        # Extract patient demographics from DICOM series
        dicom_series = None
        for study_selected_series in study_selected_series_list:
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            break
        
        if dicom_series is None:
            raise ValueError("Could not extract DICOM series from study_selected_series_list")
        
        # Get patient info from first SOP instance
        orig_ds = dicom_series.get_sop_instances()[0].get_native_sop_instance()
        patient_sex = orig_ds.get("PatientSex", "")
        patient_age_str = orig_ds.get("PatientAge", "")
        
        # Convert DICOM age string (e.g., "012Y") to numeric value
        if patient_age_str and patient_age_str.endswith("Y"):
            patient_age = float(patient_age_str[:-1])
        else:
            raise ValueError(f"Invalid or missing PatientAge: {patient_age_str}")
        
        self._logger.info(f"Extracted PatientSex: {patient_sex}, PatientAge: {patient_age}")
        
        if self.assets_path is None:
            raise ValueError("assets_path must be provided in constructor if not passed as input")
        
        # Validate inputs
        if metrics_dict is None or not isinstance(metrics_dict, dict):
            raise ValueError("metrics_dict must be a dictionary from SegmentationMetricsOperator")
        
        if patient_age is None or not isinstance(patient_age, (int, float)):
            raise ValueError("patient_age must be a numeric value")
        
        # If patient_sex is m or f, convert to full string
        if patient_sex.lower() == "m":
            patient_sex = "Male"
        elif patient_sex.lower() == "f":
            patient_sex = "Female"
        
        if patient_sex is None or not isinstance(patient_sex, str):
            raise ValueError("patient_sex must be a string ('Male' or 'Female')")
        
        patient_sex = patient_sex.strip().capitalize()
        if patient_sex not in ["Male", "Female"]:
            raise ValueError(f"patient_sex must be 'Male' or 'Female', got: {patient_sex}")
        
        # If additional_metrics_map is provided, augment metrics_dict
        if self.additional_metrics_map:
            for metric_key, mapping in self.additional_metrics_map.items():
                organ = mapping.get("organ")
                metric = mapping.get("metric")
                if organ in metrics_dict and metric in metrics_dict[organ]:
                    metrics_dict[metric_key] = {
                        "biomarker_value": metrics_dict[organ][metric]
                    }
        
        # Calculate z-scores and percentiles for all organs
        zscore_dict, processed_organs, units_dict = self.calculate_zscores_batch(
            metrics_dict, patient_age, patient_sex, self.assets_path
        )
        self._logger.info(f"Z-score calculation complete, z_score_dict: {zscore_dict}")
        
        # Add units to zscore_dict
        for organ_name, data in zscore_dict.items():
            unit = units_dict.get(organ_name, "")
            data["unit"] = unit
        
        # Generate plot if requested and data is available
        pdf_bytes = None
        if self.generate_plots and processed_organs:
            try:
                # Create and save visualization to BytesIO buffer
                pdf_bytes = self.create_visualization(
                    processed_organs, patient_age, patient_sex
                )
                
                self._logger.info(f"PDF report generated in memory ({len(pdf_bytes)} bytes)")
            except Exception as e:
                self._logger.error(f"Error generating PDF visualization: {e}")
                pdf_bytes = None
        
        self._logger.info("Emitting outputs")
        
        # Emit outputs
        op_output.emit(zscore_dict, self.output_name_zscore)
        op_output.emit(pdf_bytes, self.output_name_pdf_bytes)
        
    def _load_biomarker_data(
        self, biomarker_name: str, assets_path: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load the quantile regression data for a specific biomarker.
        
        Args:
            biomarker_name: Name of the biomarker (should match subfolder in assets)
            assets_path: Path to assets folder
            
        Returns:
            Tuple of (male_df, female_df) or (None, None) if loading fails
        """
        try:
            biomarker_path = Path(assets_path) / biomarker_name
            df_m = pd.read_csv(biomarker_path / "results_m_fine.csv", index_col=0)
            df_f = pd.read_csv(biomarker_path / "results_f_fine.csv", index_col=0)
            return df_m, df_f
        except Exception as e:
            self._logger.error(f"Error loading data for biomarker '{biomarker_name}': {e}")
            return None, None

    def _calculate_percentile(
        self,
        age: float,
        sex: str,
        biomarker_value: float,
        df_m: pd.DataFrame,
        df_f: pd.DataFrame
    ) -> Optional[float]:
        """Calculate the percentile for a biomarker value given age and sex.
        
        Args:
            age: Patient age in years
            sex: Patient sex ("Male" or "Female")
            biomarker_value: Measured biomarker value
            df_m: Male quantile DataFrame
            df_f: Female quantile DataFrame
            
        Returns:
            Percentile as float between 0 and 1, or None if calculation fails
        """
        # Select appropriate dataframe
        df = df_m if sex == "Male" else df_f
        
        # Get quantile column names (exclude 'Age' and index columns)
        quantile_cols = [col for col in df.columns if col not in ['Age', 'Unnamed: 0'] and col != 'index']
        quantile_cols_sorted = sorted(quantile_cols, key=float)
        quantiles = [float(q) for q in quantile_cols_sorted]
        
        # Sort by age for interpolation
        df = df.sort_values('Age').reset_index(drop=True)
        
        # Create interpolation functions for each quantile level
        interp_functions = {}
        for i, q in enumerate(quantiles):
            col_name = quantile_cols_sorted[i]
            interp_functions[q] = interp1d(
                df['Age'],
                df[col_name],
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
        
        # Get the biomarker values at each quantile for the patient's age
        biomarker_quantiles = []
        for q in quantiles:
            biomarker_val = interp_functions[q](age)
            biomarker_quantiles.append(biomarker_val)
        
        # Pair quantiles with their corresponding biomarker values
        quantile_biomarker = list(zip(quantiles, biomarker_quantiles))
        
        # Sort by biomarker values to ensure proper ordering
        quantile_biomarker_sorted = sorted(quantile_biomarker, key=lambda x: x[1])
        sorted_quantiles = [q for q, b in quantile_biomarker_sorted]
        sorted_biomarkers = [b for q, b in quantile_biomarker_sorted]
        
        # Handle edge cases
        if np.isnan(biomarker_value):
            return None
        
        percentile = None
        
        # Below lowest quantile
        if biomarker_value <= sorted_biomarkers[0]:
            q_low, q_high = sorted_quantiles[0], sorted_quantiles[1]
            b_low, b_high = sorted_biomarkers[0], sorted_biomarkers[1]
            if b_high == b_low:
                percentile = q_low
            else:
                slope = (q_high - q_low) / (b_high - b_low)
                percentile = q_low + slope * (biomarker_value - b_low)
        
        # Above highest quantile
        elif biomarker_value >= sorted_biomarkers[-1]:
            q_low, q_high = sorted_quantiles[-2], sorted_quantiles[-1]
            b_low, b_high = sorted_biomarkers[-2], sorted_biomarkers[-1]
            if b_high == b_low:
                percentile = q_high
            else:
                slope = (q_high - q_low) / (b_high - b_low)
                percentile = q_high + slope * (biomarker_value - b_high)
        
        # Between quantiles - interpolate
        else:
            for i in range(len(sorted_biomarkers) - 1):
                b_low = sorted_biomarkers[i]
                b_high = sorted_biomarkers[i + 1]
                q_low = sorted_quantiles[i]
                q_high = sorted_quantiles[i + 1]
                
                if b_low <= biomarker_value <= b_high:
                    percentile = q_low + (q_high - q_low) * (biomarker_value - b_low) / (b_high - b_low)
                    break
        
        # Clamp to valid range to avoid numerical issues with norm.ppf
        if percentile is not None:
            epsilon = 0.001
            percentile = np.clip(percentile, epsilon, 1 - epsilon)
        
        return percentile

    def calculate_zscores_batch(
        self,
        metrics_dict: Dict[str, Dict],
        patient_age: float,
        patient_sex: str,
        assets_path: str
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        """Calculate z-scores and percentiles for all organs in metrics_dict.
        
        Args:
            metrics_dict: Dictionary from SegmentationMetricsOperator
            patient_age: Patient age in years
            patient_sex: Patient sex ("Male" or "Female")
            assets_path: Path to assets folder
            
        Returns:
            Tuple of (zscore_dict, processed_organs_dict, units_dict) where:
            - zscore_dict contains results for output
            - processed_organs_dict contains data needed for plotting
        """
        zscore_dict = {}
        processed_organs = {}
        # Establish units dict
        units_dict = {}
            
        for organ_name, metrics in metrics_dict.items():
            # Skip if metrics contain an error
            if "error" in metrics:
                self._logger.warning(f"Skipping organ '{organ_name}' due to error in metrics: {metrics['error']}")
                zscore_dict[organ_name] = {"error": metrics["error"]}
                continue
            
            # Map organ name to asset folder name if mapping provided
            asset_name = self.organ_name_mapping.get(organ_name, organ_name)
            
            # Extract biomarker value (volume_ml or area_cm2)
            biomarker_value = metrics.get("volume_ml") or metrics.get("area_cm2") or metrics.get("biomarker_value")
            
            units_dict[organ_name] = "mL" if "volume_ml" in metrics else "cm²" if "area_cm2" in metrics else ""
            
            if biomarker_value is None or biomarker_value == 0:
                self._logger.warning(f"No valid biomarker value for organ '{organ_name}', skipping")
                zscore_dict[organ_name] = {
                    "percentile": None,
                    "z_score": None,
                    "biomarker_value": biomarker_value,
                    "message": "No segmentation detected"
                }
                continue
            
            # Load normative data
            df_m, df_f = self._load_biomarker_data(asset_name, assets_path)
            
            if df_m is None or df_f is None:
                self._logger.error(f"Could not load normative data for '{asset_name}', skipping '{organ_name}'")
                zscore_dict[organ_name] = {
                    "error": f"Normative data not available for '{asset_name}'"
                }
                continue
            
            try:
                # Calculate percentile
                percentile = self._calculate_percentile(
                    patient_age, patient_sex, biomarker_value, df_m, df_f
                )
                
                if percentile is None:
                    zscore_dict[organ_name] = {
                        "error": "Failed to calculate percentile"
                    }
                    continue
                
                # Calculate z-score from percentile
                z_score = norm.ppf(percentile)
                
                # Store results
                zscore_dict[organ_name] = {
                    "biomarker_value": float(biomarker_value),
                    "percentile": float(percentile),
                    "percentile_pct": float(percentile * 100),
                    "z_score": float(z_score),
                    "patient_age": float(patient_age),
                    "patient_sex": patient_sex,
                    "interpretation": self._generate_interpretation(percentile, z_score)
                }
                
                # Store data for plotting
                processed_organs[organ_name] = {
                    "asset_name": asset_name,
                    "biomarker_value": biomarker_value,
                    "percentile": percentile,
                    "z_score": z_score,
                    "df_m": df_m,
                    "df_f": df_f
                }
                
            except Exception as e:
                self._logger.error(f"Error calculating z-score for '{organ_name}': {e}")
                zscore_dict[organ_name] = {
                    "error": str(e)
                }
        
        return zscore_dict, processed_organs, units_dict

    def _generate_interpretation(self, percentile: float, z_score: float) -> str:
        """Generate human-readable interpretation of results.
        
        Args:
            percentile: Percentile value (0-1)
            z_score: Z-score value
            
        Returns:
            Interpretation string
        """
        direction = "above" if z_score > 0 else "below"
        return (
            f"This value is at the {percentile*100:.1f}th percentile, "
            f"{abs(z_score):.2f} standard deviations {direction} the population mean"
        )
    
    def create_visualization(
        self,
        processed_organs: Dict[str, Dict],
        patient_age: float,
        patient_sex: str
    ) -> bytes:
        """Create matplotlib visualization with quantile curves in an Nx2 grid.
        
        Args:
            processed_organs: Dictionary with organ data
            patient_age: Patient age in years
            patient_sex: Patient sex ("Male" or "Female")
            
        Returns:
            bytes: PDF file content as bytes
        """
        if not processed_organs:
            self._logger.warning("No organs to visualize")
            return b"" 

        num_organs = len(processed_organs)
        
        # Calculate rows needed for 2 columns (No math library needed)
        ncols = 2
        nrows = (num_organs + 1) // ncols
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(15, 5 * nrows),
            squeeze=False  # Ensures axes is always a 2D array
        )
        
        # Flatten axes for easy 1D iteration
        axes_flat = axes.flatten()
        
        # Quantiles configuration
        quantiles_to_plot = [0.05, 0.25, 0.50, 0.75, 0.95]
        colors = ['red', 'orange', 'blue', 'green', 'purple']
        labels = ['5th', '25th', '50th', '75th', '95th']
        
        for i, (organ_name, organ_data) in enumerate(processed_organs.items()):
            ax = axes_flat[i]
            
            # Select relevant dataframe based on sex
            if patient_sex == "Male":
                df = organ_data["df_m"]
            else:
                df = organ_data["df_f"]
                
            biomarker_value = organ_data["biomarker_value"]
            percentile = organ_data["percentile"]
            z_score = organ_data["z_score"]
            
            # Extract only quantile columns (exclude metadata)
            quantile_cols = [
                col for col in df.columns 
                if col not in ['Age', 'Unnamed: 0', 'index']
            ]
            quantile_mapping = {float(col): col for col in quantile_cols}
            
            # Plot quantile curves
            for j, q in enumerate(quantiles_to_plot):
                if q in quantile_mapping:
                    col_name = quantile_mapping[q]
                    ax.plot(
                        df['Age'],
                        df[col_name],
                        color=colors[j],
                        label=f'{labels[j]} percentile' if i == 0 else "", 
                        linewidth=2
                    )
            
            # Plot Patient Point
            ax.scatter(
                [patient_age],
                [biomarker_value],
                color='red',
                s=150,
                marker='X',
                zorder=5,
                edgecolors='black',
                linewidths=1.5,
                label='Patient' if i == 0 else ""
            )
            
            # Annotation Box: Value, Percentile, Z-score
            ax.annotate(
                f'Val: {biomarker_value:.1f}\nP: {percentile*100:.1f}%\nZ: {z_score:.2f}',
                (patient_age, biomarker_value),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                fontsize=9,
                fontweight='bold'
            )
            
            # Styling
            ax.set_xlabel('Age (years)', fontsize=10)
            ax.set_ylabel('Volume (mL)', fontsize=10)
            ax.set_title(f'{organ_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend outside the plot area (top center), only on the first plot
            if i == 0:
                ax.legend(
                    loc='lower center', 
                    bbox_to_anchor=(0.5, 1.05), # Moves legend above the plot
                    ncol=3, # Arranges items in 3 columns for a flatter look
                    fontsize=9, 
                    frameon=False
                )

        # Hide any unused subplots
        for j in range(num_organs, nrows * ncols):
            axes_flat[j].axis('off')

        # Overall Title
        fig.suptitle(
            f'Organ Volume Quantile Curves - {patient_sex} Patient, Age {patient_age:.1f} years',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout()
        
        # Save to BytesIO buffer
        buffer = BytesIO()
        fig.savefig(buffer, format='pdf', bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
def test():
    """Test function for the SegmentationZScoreOperator."""
    import numpy as np
    from monai.deploy.core import Fragment
    
    print("Testing SegmentationZScoreOperator...")
    print("=" * 60)
    
    # Create mock metrics_dict (as if from SegmentationMetricsOperator)
    metrics_dict = {
        "liver": {
            "volume_ml": 1200.0,  # 1200 mL
            "num_slices": 80,
            "slice_range": (20, 100),
            "pixel_count": 1200000,
            "mean_intensity_hu": 55.3,
            "std_intensity_hu": 12.1
        },
        "spleen": {
            "volume_ml": 180.0,  # 180 mL
            "num_slices": 40,
            "slice_range": (30, 70),
            "pixel_count": 180000,
            "mean_intensity_hu": 48.7,
            "std_intensity_hu": 10.5
        }
    }
    
    # Patient information
    patient_age = 12.5
    patient_sex = "Female"
    
    # Assets path (adjust to your local path)
    assets_path = "/mnt/projects/monai-deploy-app-sdk/examples/apps/cchmc_ped_abd_ct_seg_app/assets"
    
    # Check if assets path exists
    if not os.path.exists(assets_path):
        print(f"Warning: Assets path does not exist: {assets_path}")
        print("Please update the assets_path variable in the test function.")
        return
    
    # Create operator instance
    fragment = Fragment()
    operator = SegmentationZScoreOperator(
        fragment,
        assets_path=assets_path,
        generate_plots=True
    )
    
    print(f"\nPatient Info:")
    print(f"  Age: {patient_age} years")
    print(f"  Sex: {patient_sex}")
    print(f"\nInput Metrics:")
    for organ, metrics in metrics_dict.items():
        print(f"  {organ}: {metrics['volume_ml']} mL")
    
    # Calculate z-scores
    print("\nCalculating z-scores and percentiles...")
    zscore_dict, processed_organs = operator.calculate_zscores_batch(
        metrics_dict, patient_age, patient_sex, assets_path
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Z-Score Results:")
    print("=" * 60)
    for organ_name, results in zscore_dict.items():
        print(f"\n{organ_name.upper()}:")
        for key, value in results.items():
            if key != "interpretation":
                print(f"  {key}: {value}")
        if "interpretation" in results:
            print(f"\n  Interpretation: {results['interpretation']}")
    
    # Generate visualization
    if processed_organs:
        print("\n" + "=" * 60)
        print("Generating PDF visualization...")
        try:
            pdf_bytes = operator.create_visualization(
                processed_organs, patient_age, patient_sex, assets_path
            )
            print(f"PDF generated successfully: {len(pdf_bytes)} bytes")
            
            # Optionally save for testing
            test_output_path = Path("/tmp/pdfs_test")
            test_output_path.mkdir(parents=True, exist_ok=True)
            test_pdf_file = test_output_path / "zscore_report_test.pdf"
            with open(test_pdf_file, "wb") as f:
                f.write(pdf_bytes)
            print(f"Test PDF saved to: {test_pdf_file}")
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nVisualization skipped (no organs processed)")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    test()
