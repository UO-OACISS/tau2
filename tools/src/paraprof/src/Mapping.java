/* 
   Mapping.java

   Title:      ParaProf
   Author:     Robert Bell
   Description: Many entities in ParaProf reprsent a mapping between a name and
                an id.  This interface guarantees that mapping.
*/

package paraprof;

interface Mapping{
    public void setMappingName(String mappingName);
    public String getMappingName();
  
    public void setMappingID(int mappingID);
    public int getMappingID();
}

