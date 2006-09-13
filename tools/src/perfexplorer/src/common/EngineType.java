/**
 * Created on Feb 13, 2006
 *
 */
package common;

import java.io.Serializable;


/**
 * This class is used as a typesafe enumeration.
 *
 * <P>CVS $Id: EngineType.java,v 1.1 2006/09/13 23:28:20 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public final class EngineType implements Serializable {

    /**
     * One attribute, the name - it is transient so it is not serialized
     */
    private final transient String _name;

    /**
     * Static instances of the engine.
     */
    public final static EngineType WEKA = new EngineType("Weka");
    public final static EngineType RPROJECT = new EngineType("R Project");
    public final static EngineType OCTAVE = new EngineType("Octave");
    
    /**
     * The constructor is private, so this class cannot be instantiated.
     * @param name
     */
    private EngineType(String name) {
        this._name = name;
    }
    
    /**
     * Only one public method, to return the name of the engine
     * @return
     */
    public String toString() {
        return this._name;
    }
    
    // The following declarations are necessary for serialization
    private static int nextOrdinal = 0;
    private final int ordinal = nextOrdinal++;
    private static final EngineType[] VALUES = {WEKA, RPROJECT, OCTAVE};
    
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
