/* 
   Name:        GlobalSortedMappingElement.java
   Author:      Robert Bell
   
   Description: A simple class used to maintain a name-id mapping.
   Used by the GlobalMapping class to support efficient id lookups
   to sorted lists.

   Things to do: Class is complete.
*/

package edu.uoregon.tau.dms.dss;

public class GlobalSortedMappingElement implements Comparable{
    public GlobalSortedMappingElement(String mappingName, int mappingID){
	this.mappingName = mappingName;
	this.mappingID = mappingID;
    }

    public void setMappingName(String mappingName){
	this.mappingName = mappingName;}

    public String getMappingName(){
	return mappingName;}

    public void setMappingID(int mappingID){
	this.mappingID = mappingID;}

    public int getMappingID(){
	return mappingID;}

    public int compareTo(Object inObject){
	return  mappingName.compareTo(((GlobalSortedMappingElement)inObject).getMappingName());
    }

    //####################################
    //Instance Data.
    //####################################
    String mappingName = null;
    int mappingID = -1;
    //####################################
    //End - Instance Data.
    //####################################
}
