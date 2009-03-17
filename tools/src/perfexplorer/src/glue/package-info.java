/** 
 * The glue package provides the classes necessary to construct PerfExplorer scripts.
 * <p>  
 * The glue package provides the classes which can be used to control PerfExplorer.
 * These classes all derive from two base interfaces, 
 * {@link edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation} and
 * {@link edu.uoregon.tau.perfexplorer.glue.PerformanceResult}.  In addition to the
 * glue package, some classes from PerfDMF may also need to be included, such
 * as the {@link edu.uoregon.tau.perfdmf.Application},
 * {@link edu.uoregon.tau.perfdmf.Experiment}, and 
 * {@link edu.uoregon.tau.perfdmf.Trial} classes.  
 * <p>  
 * <center><img width="50%" src="doc-files/PerfExplorerGlueObjects.png"/></center>
 * <p>  
 * The glue package provides a user-accessible programming interface,
 * with limited exposure to a number analysis data objects and process
 * objects.  As an example, take the hierarchy which constructs
 * the {@link edu.uoregon.tau.perfexplorer.glue.DeriveMetricOperation} class.
 * The top level interface for the processing classes is the 
 * {@link edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation}, which
 * defines the interface for all process objects.  The interface consists of
 * methods to define input data, process the inputs, get output data objects,
 * and reset the process.  The 
 * {@link edu.uoregon.tau.perfexplorer.glue.AbstractPerformanceOperation} is an
 * abstract implementation of the 
 * {@link edu.uoregon.tau.perfexplorer.glue.PerformanceAnalysisOperation} interface,
 * and provides basic internal member variables, such as the input data and
 * output data.  Finally, the 
 * {@link edu.uoregon.tau.perfexplorer.glue.DeriveMetricOperation} class is an
 * example of a concrete extension of the 
 * {@link edu.uoregon.tau.perfexplorer.glue.AbstractPerformanceOperation}
 * class, and will take
 * one more more input data sets with two or more metrics each, and generate
 * a derived metric representing either the addition, subtraction, multiplication,
 * or division of one metric with the other.  
 * <p>
 * Corresponding with the operation
 * hierarchy is the data hierarchy.  At the top of the hierarchy is the
 * {@link edu.uoregon.tau.perfexplorer.glue.PerformanceResult} 
 * interface, which defines basic methods for
 * accessing the profile data within.  The abstract implementation of the
 * interface is {@link edu.uoregon.tau.perfexplorer.glue.AbstractResult} class,
 * which defines many internal data structures and static constants.  The
 * {@link edu.uoregon.tau.perfexplorer.glue.TrialResult} class is
 * an example of a class which is a concrete implementation of the
 * abstract class, and provides an object which holds the performance
 * profile data for a given trial, when loaded from PerfDMF.
 *
 * 
 * @since 2.0 
 * @see edu.uoregon.tau.perfdmf
 */ 
package edu.uoregon.tau.perfexplorer.glue;


