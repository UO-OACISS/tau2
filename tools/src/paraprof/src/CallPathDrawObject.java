/* 
   CallPathDrawObject.java

   Title:      ParaProf
   Author:     Robert Bell
   
   Used in the CallPathTextWindowPanel class to keep track of what is currently
   being drawn.

   Things to do: Class is complete.
*/

package paraprof;

public class CallPathDrawObject{

    public CallPathDrawObject(GlobalThreadDataElement globalThreadDataElement, boolean parentChild, boolean spacer){
	this.globalThreadDataElement = globalThreadDataElement;
	this.parentChild = parentChild;
	this.spacer = spacer;
    }

    public int getMappingID(){
	if(globalThreadDataElement==null)
	    return -1;
	else
	    return globalThreadDataElement.getMappingID();
    }

    public String getMappingName(){
	if(globalThreadDataElement==null)
	    return "Spacer - name not set";
	else
	    return globalThreadDataElement.getMappingName();
    }

    public void setExclusiveValue(double exclusiveValue){
	this.exclusiveValue = exclusiveValue;}

    public double getExclusiveValue(){
	return this.exclusiveValue;}

    public void setInclusiveValue(double inclusiveValue){
	this.inclusiveValue = inclusiveValue;}

    public double getInclusiveValue(){
	return this.inclusiveValue;}

    public void setNumberOfCallsFromCallPathObjects(int numberOfCallsFromCallPathObjects){
	this.numberOfCallsFromCallPathObjects = numberOfCallsFromCallPathObjects;}

    public int getNumberOfCallsFromCallPathObjects(){
	return numberOfCallsFromCallPathObjects;}

    public void setNumberOfCalls(int numberOfCalls){
	this.numberOfCalls = numberOfCalls;}

    public int getNumberOfCalls(){
	return numberOfCalls;}

    public boolean isParentChild(){
	return parentChild;}

    public boolean isSpacer(){
	return spacer;}
    //####################################
    //Instance Data.
    //####################################
    GlobalThreadDataElement globalThreadDataElement = null;
    boolean parentChild = false;
    boolean spacer = false;

    private double exclusiveValue = 0.0;
    private double inclusiveValue = 0.0;
    private int numberOfCallsFromCallPathObjects = 0;
    private int numberOfCalls = 0;
    //####################################
    //End - Instance Data.
    //####################################
}
