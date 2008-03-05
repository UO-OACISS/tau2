/**
 * Created on Feb 13, 2006
 *
 */
package common;

import java.io.Serializable;


/**
 * This class is used as a typesafe enumeration.
 *
<<<<<<< SelectionType.java
 * <P>CVS $Id: SelectionType.java,v 1.3 2008/03/05 00:25:53 khuck Exp $</P>
=======
 * <P>CVS $Id: SelectionType.java,v 1.3 2008/03/05 00:25:53 khuck Exp $</P>
>>>>>>> 1.1.10.1
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public final class SelectionType implements Serializable {

    /**
     * One attribute, the value - it is transient so it is not serialized
     */
    private final transient int _value;

    /**
     * Static instances of the engine.
     */
    public final static SelectionType NO_MULTI = new SelectionType(0);
    public final static SelectionType APPLICATION = new SelectionType(1);
    public final static SelectionType EXPERIMENT = new SelectionType(2);
    public final static SelectionType TRIAL = new SelectionType(3);
    public final static SelectionType METRIC = new SelectionType(4);
    public final static SelectionType VIEW = new SelectionType(5);
    public final static SelectionType EVENT = new SelectionType(6);
    
    
    /**
     * The constructor is private, so this class cannot be instantiated.
     * @param name
     */
    private SelectionType(int value) {
        this._value = value;
    }
    
    // The following declarations are necessary for serialization
    private static int nextOrdinal = 0;
    private final int ordinal = nextOrdinal++;
    private static final SelectionType[] VALUES = 
        {NO_MULTI, APPLICATION, EXPERIMENT, TRIAL, METRIC, VIEW, EVENT};
    
    /**
     * This method is necessary, because we are serializing the object.
     * When the object is serialized, a NEW INSTANCE is created, and we
     * can't compare reference pointers.  When that is the case, replace the
     * newly instantiated object with a static reference.
     * 
     * @return
     * @throws java.io.ObjectStreamException
     */
    Object readResolve () throws java.io.ObjectStreamException
    {
        return VALUES[ordinal];    // Canonicalize
    }
}
