/* 
   CallPathDrawObject.java

   Title:      ParaProf
   Author:     Robert Bell
   
   Used in the CallPathTextWindowPanel class to keep track of what is currently
   being drawn.

   Things to do: Class is complete.
*/

package edu.uoregon.tau.paraprof;

import edu.uoregon.tau.dms.dss.*;

public class CallPathDrawObject implements Mapping{

    public CallPathDrawObject(Mapping mapping, boolean parent, boolean child, boolean spacer){
	this.mapping = mapping;
	this.parent = parent;
	this.child = child;
	this.spacer = spacer;
    }
    
    public void setMappingName(String mappingName){}
    public String getMappingName(){
	if(mapping==null)
	    return "Spacer - name not set";
	else
	    return mapping.getMappingName();
    }

    public void setMappingID(int mappingID){}
    public int getMappingID(){
	if(mapping==null)
	    return -1;
	else
	    return mapping.getMappingID();
    }

    public void setExclusiveValue(double exclusiveValue){
	this.exclusiveValue = exclusiveValue;}

    public double getExclusiveValue(){
	return this.exclusiveValue;}

    public void setInclusiveValue(double inclusiveValue){
	this.inclusiveValue = inclusiveValue;}

    public double getInclusiveValue(){
	return this.inclusiveValue;}

    public void setNumberOfCallsFromCallPathObjects(double numberOfCallsFromCallPathObjects){
	this.numberOfCallsFromCallPathObjects = numberOfCallsFromCallPathObjects;}

    public double getNumberOfCallsFromCallPathObjects(){
	return numberOfCallsFromCallPathObjects;}

    public void setNumberOfCalls(double numberOfCalls){
	this.numberOfCalls = numberOfCalls;}

    public double getNumberOfCalls(){
	return numberOfCalls;}

    public boolean isParent(){
	return parent;}

    public boolean isChild(){
	return child;}

    public boolean isParentChild(){
	return (parent||child);}

    public boolean isSpacer(){
	return spacer;}

    public void setExpanded(boolean expanded){
	this.expanded = expanded;}

    public boolean isExpanded(){
	return this.expanded;}
    //####################################
    //Instance Data.
    //####################################
    Mapping mapping = null;
    boolean parent = false;
    boolean child = false;
    boolean spacer = false;
    boolean expanded = false;

    private double exclusiveValue = 0.0;
    private double inclusiveValue = 0.0;
    private double numberOfCallsFromCallPathObjects = 0;
    private double numberOfCalls = 0;
    //####################################
    //End - Instance Data.
    //####################################
}
