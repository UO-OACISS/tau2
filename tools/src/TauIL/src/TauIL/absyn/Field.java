/***********************************************************************
 *
 * File        : Field.java
 * Author      : Tyrel Datwyler
 *
 * Description : Class represents a type of profile data to use in
 *               analysis.
 *
 ***********************************************************************/

package TauIL.absyn;

public class Field implements SyntaxAttribute {
    public static final int NUMCALLS = 0, NUMSUBRS = 1, PERCENT = 2, USEC = 3,
	CUMUSEC = 4, USECS_CALL = 5, COUNT = 6, TOTCOUNT = 7, COUNTS_CALL = 8, STDDEV = 9;

    public static final int NA = -1, TIMER = 0, COUNTER = 1;

    public static final String [] literals = { "numcalls", "numsubrs", "percent", 
					       "usec", "cumusec", "usecs/call", "count", 
					       "totalcount", "counts/call", "stddev" };

    public int field;
    public int metric = NA;

    public Field(int field) {
	this.field = field;
	if ((field >= USEC) && (field <= USECS_CALL))
	    this.metric = TIMER;
	else if ((field >= COUNT) && (field <= COUNTS_CALL))
	    this.metric = COUNTER;
    }

    public String generateSyntax() {
	return literals[field];
    }

    
}
