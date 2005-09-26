/* 
  
  ParaProfTreeNodeUserObject.java
  
  Title:       ParaProfTreeUserObject.java
  Author:      Robert Bell
  Description: All implementations of this interface promise to clear
               DefaultMutableTreeNode references for self and to call
               clearDefaultMutableTreeNodes() on all known 
               ParaProfTreeNodeUserObject implementers self contains.
*/

package edu.uoregon.tau.paraprof;

interface ParaProfTreeNodeUserObject{
    public void clearDefaultMutableTreeNodes();
}
