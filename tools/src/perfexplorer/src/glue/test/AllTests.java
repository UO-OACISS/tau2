package edu.uoregon.tau.perfexplorer.glue.test;

import junit.framework.Test;
import junit.framework.TestSuite;

public class AllTests {

	public static Test suite() {
		TestSuite suite = new TestSuite("Test for glue");
		//$JUnit-BEGIN$
		suite.addTestSuite(SaveResultOperationTest.class);
		suite.addTestSuite(CorrelateEventsWithMetadataTest.class);
		suite.addTestSuite(PCAOperationTest.class);
		suite.addTestSuite(DeriveMetricOperationTest.class);
		suite.addTestSuite(CopyOperationTest.class);
		suite.addTestSuite(BasicStatisticsOperationTest.class);
		suite.addTestSuite(NaiveBayesOperationTest.class);
		suite.addTestSuite(MergeTrialsOperationTest.class);
		suite.addTestSuite(DrawBoxChartGraphTest.class);
		suite.addTestSuite(ScaleMetricOperationTest.class);
		suite.addTestSuite(SmartKMeansOperationTest.class);
		suite.addTestSuite(TrialMetadataTest.class);
		suite.addTestSuite(DrawGraphTest.class);
		suite.addTestSuite(SplitTrialPhasesOperationTest.class);
		suite.addTestSuite(DeriveAllMetricsOperationTest.class);
		suite.addTestSuite(TrialThreadMetadataTest.class);
		suite.addTestSuite(DifferenceMetadataOperationTest.class);
		suite.addTestSuite(TrialResultTest.class);
		suite.addTestSuite(LinearRegressionOperationTest.class);
		suite.addTestSuite(ExtractEventOperationTest.class);
		suite.addTestSuite(KMeansOperationTest.class);
		suite.addTestSuite(ProvenanceTest.class);
		suite.addTestSuite(DifferenceOperationTest.class);
		suite.addTestSuite(LogarithmicOperationTest.class);
		suite.addTestSuite(SupportVectorOperationTest.class);
		suite.addTestSuite(ScalabilityOperationTest.class);
		suite.addTestSuite(RatioOperationTest.class);
		suite.addTestSuite(TopXEventsTest.class);
		suite.addTestSuite(ExtractPhasesOperationTest.class);
		suite.addTestSuite(TopXPercentEventsTest.class);
		suite.addTestSuite(ExtractMetricOperationTest.class);
		suite.addTestSuite(DrawMMMGraphTest.class);
		suite.addTestSuite(ExtractRankOperationTest.class);
		suite.addTestSuite(DataSourceResultTest.class);
		suite.addTestSuite(CorrelationOperationTest.class);
		suite.addTestSuite(CQoSClassifierOperationTest.class);
		suite.addTestSuite(DefaultOperationTest.class);
		//$JUnit-END$
		return suite;
	}

}
