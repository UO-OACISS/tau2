from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *
from java.util import HashSet
from java.util import ArrayList
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Metric
from edu.uoregon.tau.perfexplorer.glue import PerformanceResult
from edu.uoregon.tau.perfexplorer.glue import PerformanceAnalysisOperation
from edu.uoregon.tau.perfexplorer.glue import ExtractEventOperation
from edu.uoregon.tau.perfexplorer.glue import Utilities
from edu.uoregon.tau.perfexplorer.glue import BasicStatisticsOperation
from edu.uoregon.tau.perfexplorer.glue import DeriveMetricOperation
from edu.uoregon.tau.perfexplorer.glue import ScaleMetricOperation
from edu.uoregon.tau.perfexplorer.glue import DeriveMetricEquation
from edu.uoregon.tau.perfexplorer.glue import DeriveMetricsFileOperation
from edu.uoregon.tau.perfexplorer.glue import MergeTrialsOperation
from edu.uoregon.tau.perfexplorer.glue import TrialResult
from edu.uoregon.tau.perfexplorer.glue import AbstractResult
from edu.uoregon.tau.perfexplorer.glue import DrawGraph
from edu.uoregon.tau.perfexplorer.glue import DrawMetadataGraph
from edu.uoregon.tau.perfexplorer.glue import SaveResultOperation

True = 1
False = 0


def load(inApp, inExp, inTrial):
  trial1 = Utilities.getTrial(inApp, inExp, inTrial)
  result1 = TrialResult(trial1)
  return result1


def main():
        print "--------------- JPython test script start ------------"
        inputs = load("Application","Experiment","Trial")

        grapher = DrawMetadataGraph(inputs)
        grapher.setMetadataField("cluster-membership")
        #grapher.setTitle("My Title")
        #grapher.setXAxisLabel("")
        #grapher.setYAxisLabel("")
        grapher.processData()

        print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()

