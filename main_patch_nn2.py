"""
This script runs a pipeline for prediction and analysis of the cleaned CMB signal using <TODO: Name it>

The pipeline consists of the following steps:
1. <TODO: Steps here>

And also generating various analysis figures, throughout.

Final comparison is performed in the main_analysis_compare.py script.

Usage:
    python main_<TODO: Name it>.py
"""
import logging

import hydra

from cmbml.utils.check_env_var import validate_environment_variable
from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims import MaskCreatorExecutor
from cmbml.demo_patch_nn import (
    MakeLutExecutor,

    ChoosePatchesExecutor,
    TryShowPatchExecutor,
    TryShowPatchDistExecutor,

    FindDatasetStatsSerialExecutor,
    FindDatasetStatsParallelExecutor,
    PreprocessPatchesExecutor,

    TrainingTryDataloaderExecutor,
    TrainingTryNetworkExecutor,
    TrainingExecutor,
    TrainingNoPreprocessExecutor,

    PredictTryDataloaderExecutor,
    PredictTryModelLoadExecutor,
    PredictExectutor
    )

from cmbml.analysis import (
    CommonRealPostExecutor,
    CommonNNPredPostExecutor,
    CommonNNShowSimsPostExecutor,
    CommonCMBNNCSShowSimsPostIndivExecutor,
    PixelAnalysisExecutor,
    PixelSummaryExecutor,
    PixelSummaryFigsExecutor,
    ConvertTheoryPowerSpectrumExecutor,
    MakeTheoryPSStats,
    NNMakePowerSpectrumExecutor,
    PowerSpectrumAnalysisExecutor,
    PowerSpectrumSummaryExecutor,
    PowerSpectrumSummaryFigsExecutor,
    PostAnalysisPsFigExecutor,
    ShowOnePSExecutor)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_patch_nn")
def run_cmbnncs(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    ##########################
    # CLEANING: PREPROCESSING  
    # (Simulations must be downloaded or generated previously)
    ##########################

    # We create a lookup table so that patches of pixels can be extracted from maps more quickly.
    pipeline_context.add_pipe(MakeLutExecutor)

    # The mask is used so that patches are chosen without excess contamination.
    pipeline_context.add_pipe(MaskCreatorExecutor)

    # We choose patches of pixels from the maps to use for training. For each simulation, we use just a single
    #   patch. This is a naive method to avoid overfitting, but suffices.
    pipeline_context.add_pipe(ChoosePatchesExecutor)

    # These executors are useful for debugging, but can not be run in the final pipeline as they display figures
    # pipeline_context.add_pipe(TryShowPatchExecutor)
    # pipeline_context.add_pipe(TryShowPatchDistExecutor)

    # We find the statistics of the dataset so that we can scale the data to the domain [0,1]
    pipeline_context.add_pipe(FindDatasetStatsSerialExecutor)  # A slower, but more easily understood method

    # Preprocessing speeds up the training process by snipping the patches from the maps; 
    #   This pipeline does not use preprocessed data for training.
    # pipeline_context.add_pipe(PreprocessPatchesExecutor)

    # These executors are useful for demonstration/debugging
    # pipeline_context.add_pipe(TrainingTryDataloaderExecutor)
    # pipeline_context.add_pipe(TrainingTryNetworkExecutor)

    ##########################
    # CLEANING: TRAINING
    ##########################

    # Train the model using the preprocessed dataset
    pipeline_context.add_pipe(TrainingNoPreprocessExecutor)  # This is an alternative method. It is significantly slower.
    # pipeline_context.add_pipe(TrainingExecutor)  # This is the normal training method, but requires a preprocessed dataset.

    # These executors are useful for demonstration/debugging
    # pipeline_context.add_pipe(PredictTryDataloaderExecutor)
    # pipeline_context.add_pipe(PredictTryModelLoadExecutor)

    # Predict the cleaned CMB signal using the trained model
    pipeline_context.add_pipe(PredictExectutor)

    ##########################
    # ANALYSIS
    ##########################

    # Apply to the target (CMB realization)
    pipeline_context.add_pipe(CommonRealPostExecutor)
    # Apply to CMBNNCS's predictions
    pipeline_context.add_pipe(CommonNNPredPostExecutor)
    # Show results of cleaning
    pipeline_context.add_pipe(CommonNNShowSimsPostExecutor)

    # pipeline_context.add_pipe(PixelAnalysisExecutor)
    # pipeline_context.add_pipe(PixelSummaryExecutor)
    # pipeline_context.add_pipe(PixelSummaryFigsExecutor)

    # # # These two do not need to run individually for all models (but they're fast, so it doesn't matter unless you're actively changing them)
    # pipeline_context.add_pipe(ConvertTheoryPowerSpectrumExecutor)
    # pipeline_context.add_pipe(MakeTheoryPSStats)

    # # # # # CMBNNCS's Predictions as Power Spectra Anaylsis
    # pipeline_context.add_pipe(NNMakePSExecutor)
    # # # pipeline_context.add_pipe(ShowOnePSExecutor)  # Used for debugging; does not require full set of theory ps for simulations
    # pipeline_context.add_pipe(PSAnalysisExecutor)
    # pipeline_context.add_pipe(PowerSpectrumSummaryExecutor)
    # pipeline_context.add_pipe(PowerSpectrumSummaryFigsExecutor)
    # pipeline_context.add_pipe(PostAnalysisPsFigExecutor)

    pipeline_context.prerun_pipeline()

    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        logger.exception("An exception occured during the pipeline.", exc_info=e)
        raise e
    finally:
        logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    run_cmbnncs()
