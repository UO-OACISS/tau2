/* 
  Name:        ParaProfObserver.java
  Author:      Robert Bell
  Description: A more flexible implementation than that given in the 
               standard Observable/Observer of the jdk.
*/

package edu.uoregon.tau.dms.dss;

public interface ParaProfObserver{

    void update() throws DatabaseException;
    void update(Object obj) throws DatabaseException;
}
