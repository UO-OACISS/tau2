package edu.uoregon.TAU.dexInjector;

public class TypeDescriptor {
    public static final int UNKNOWN = 0;
    public static final int VOID    = 1;
    public static final int BOOL    = 2;
    public static final int BYTE    = 3;
    public static final int SHORT   = 4;
    public static final int CHAR    = 5;
    public static final int INT     = 6;
    public static final int LONG    = 7;
    public static final int FLOAT   = 8;
    public static final int DOUBLE  = 9;
    public static final int CLASS   = 10;
    public static final int ARRAY   = 11;

    public int type;    // as defined above
    public int size;    // dalvik registers needed
    public String desc; // string description

    public TypeDescriptor(int t, String d) {
	type = t;
	desc = d;

	switch (type) {
	case VOID:
	    size = 0;
	    break;

	case BOOL:
	case BYTE:
	case SHORT:
	case CHAR:
	case INT:
	case FLOAT:
	case CLASS:
	case ARRAY:
	    size = 1;
	    break;

	case LONG:
	case DOUBLE:
	    size = 2;
	    break;

	default:
	    size = 0;
	    break;
	}
    }

    public String toString() {
	return desc;
    }
}
