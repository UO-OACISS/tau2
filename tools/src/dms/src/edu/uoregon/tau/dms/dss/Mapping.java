/* 
   Name:        Mapping.java
   Author:      Robert Bell
   Description: Many entities in the dms represent a mapping between a name and
                an id.  This interface guarantees that mapping.
*/

package edu.uoregon.tau.dms.dss;

public interface Mapping{
    public void setMappingName(String mappingName);
    public String getMappingName();
  
    public void setMappingID(int mappingID);
    public int getMappingID();
}

