package edu.uoregon.tau.paraprof;

import edu.uoregon.tau.dms.dss.*;

/**
 * CallPathDrawObject: This object is used to hold data for the CallPathTextWindowPanel
 *   
 * <P>CVS $Id: CallPathDrawObject.java,v 1.6 2005/01/19 02:33:25 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.6 $
 * @see		CallPathTextWindow
 * @see		CallPathTextWindowPanel
 */
public class CallPathDrawObject {

    private Function function = null;
    private boolean parent = false;
    private boolean child = false;
    private boolean spacer = false;
    private boolean expanded = false;

    private double exclusiveValue = 0.0;
    private double inclusiveValue = 0.0;
    private double numberOfCallsFromCallPathObjects = 0;
    private double numberOfCalls = 0;
    
    public CallPathDrawObject(Function function, boolean parent, boolean child, boolean spacer) {
        this.function = function;
        this.parent = parent;
        this.child = child;
        this.spacer = spacer;
    }

    public String getName() {
        if (function == null) {
            return "Spacer - name not set";
        } else {
            return function.getName();
        }
    }
    
    public Function getFunction() {
        return function;
    }

    public void setExclusiveValue(double exclusiveValue) {
        this.exclusiveValue = exclusiveValue;
    }

    public double getExclusiveValue() {
        return this.exclusiveValue;
    }

    public void setInclusiveValue(double inclusiveValue) {
        this.inclusiveValue = inclusiveValue;
    }

    public double getInclusiveValue() {
        return this.inclusiveValue;
    }

    public void setNumberOfCallsFromCallPathObjects(double numberOfCallsFromCallPathObjects) {
        this.numberOfCallsFromCallPathObjects = numberOfCallsFromCallPathObjects;
    }

    public double getNumberOfCallsFromCallPathObjects() {
        return numberOfCallsFromCallPathObjects;
    }

    public void setNumberOfCalls(double numberOfCalls) {
        this.numberOfCalls = numberOfCalls;
    }

    public double getNumberOfCalls() {
        return numberOfCalls;
    }

    public boolean isParent() {
        return parent;
    }

    public boolean isChild() {
        return child;
    }

    public boolean isParentChild() {
        return (parent || child);
    }

    public boolean isSpacer() {
        return spacer;
    }

    public void setExpanded(boolean expanded) {
        this.expanded = expanded;
    }

    public boolean isExpanded() {
        return this.expanded;
    }
}