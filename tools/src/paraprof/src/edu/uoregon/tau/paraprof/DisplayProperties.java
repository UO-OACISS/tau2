/* 
   Name:        DisplayProperties.java
   Author:      Robert Bell
   Description: 
*/

package edu.uoregon.tau.paraprof;

public interface DisplayProperties{

    public void setValueType(int valueType);
    public int getValueType();
    public void setOrder(int order);
    public int getOrder();
    public void setUnits(int units);
    public int getUnits();

    public static final int EXCLUSIVE = 2;
    public static final int INCLUSIVE = 4;
    public static final int NUMBER_OF_CALLS = 6;
    public static final int NUMBER_OF_SUBROUTINES = 8;
    public static final int INCLUSIVE_PER_CALL = 10;
    public static final int DESCENDING_ORDER = 0;
    public static final int ASCENDING_ORDER = 1;
    public static final int MICROSECONDS = 0;
    public static final int MILLISECONDS = 1;
    public static final int SECONDS = 2;
}
