/* 
  
  ParaProfObserver.java
  
  Title:       ParaProfObserver.java
  Author:      Robert Bell
  Description: A more flexible implementation than that given in the 
               standard Observable/Observer of the jdk.
*/

package paraprof;

public interface ParaProfObserver{

    void update();
    void update(Object obj);
}
