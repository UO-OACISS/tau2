/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue.psl;

/**
 * @author khuck
 *
 */
public interface Property {
	boolean holds();
	double getConfidence();
	double getSeverity();
}
