/* 
  Name:        ParaProfObserver.java
  Author:      Robert Bell
  Description: A more flexible implementation than that given in the 
               standard Observable/Observer of the jdk.
*/

package edu.uoregon.tau.dms.dss;

public interface ParaProfObserver{

    void update();
    void update(Object obj);
}
