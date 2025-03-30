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
from patch_nn import (
    MakeLutExecutor,
    ChoosePatchesExecutor,
    FindDatasetStatsParallelExecutor,
    PreprocessPatchesExecutor,
    TrainingExecutor,
    PredictExectutor
    )

from cmbml.analysis import (
    # ShowSimsPrepExecutor, 
    CommonRealPostExecutor,
    CommonPredPostExecutor,
    CommonShowSimsPostExecutor,
    # CommonCMBNNCSPredPostExecutor,
    # CommonCMBNNCSShowSimsPostExecutor,
    # CommonCMBNNCSShowSimsPostIndivExecutor,
    # CMBNNCSShowSimsPredExecutor, 
    # CMBNNCSShowSimsPostExecutor,
    PixelAnalysisExecutor,
    PixelSummaryExecutor,
    # ConvertTheoryPowerSpectrumExecutor,
    # MakeTheoryPSStats,
    # CMBNNCSMakePSExecutor,
    # PixelSummaryFigsExecutor,
    # PowerSpectrumAnalysisExecutor,
    # PowerSpectrumSummaryExecutor,
    # PowerSpectrumSummaryFigsExecutor,
    # PostAnalysisPsFigExecutor,
    # ShowOnePSExecutor
    )


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_patch_nn")
def run(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    # Cleaning: Preprocessing
    pipeline_context.add_pipe(MakeLutExecutor)
    pipeline_context.add_pipe(MaskCreatorExecutor)
    pipeline_context.add_pipe(ChoosePatchesExecutor)
    pipeline_context.add_pipe(FindDatasetStatsParallelExecutor)
    pipeline_context.add_pipe(PreprocessPatchesExecutor)

    # Cleaning: Training and Prediction
    pipeline_context.add_pipe(TrainingExecutor)
    pipeline_context.add_pipe(PredictExectutor)

    # Analysis: processing the maps in a common way
    pipeline_context.add_pipe(CommonRealPostExecutor)
    pipeline_context.add_pipe(CommonPredPostExecutor)

    # Analysis: showing the maps (pixel-level result)
    pipeline_context.add_pipe(CommonShowSimsPostExecutor)

    # Analysis: getting & presenting pixel-level statistics
    pipeline_context.add_pipe(PixelAnalysisExecutor)
    pipeline_context.add_pipe(PixelSummaryExecutor)
    # pipeline_context.add_pipe(PixelSummaryFigsExecutor)  # Deactivated during repo separation

    # # Analysis: generating the power spectra and power spectra statistics
    # pipeline_context.add_pipe(NNMakePowerSpectrumExecutor)  # Deactivated during repo separation
    # pipeline_context.add_pipe(PowerSpectrumAnalysisExecutor)  # Deactivated during repo separation
    # pipeline_context.add_pipe(PowerSpectrumSummaryExecutor)  # Deactivated during repo separation
    # pipeline_context.add_pipe(PowerSpectrumSummaryFigsExecutor)  # Deactivated during repo separation
    # pipeline_context.add_pipe(PostAnalysisPsFigExecutor)  # Deactivated during repo separation

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
    run()
