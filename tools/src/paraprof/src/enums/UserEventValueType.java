package edu.uoregon.tau.paraprof.enums;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.perfdmf.*;

/**
 * type-safe enum pattern for value type
 *    
 * TODO : nothing, this class is complete
 *
 * <P>CVS $Id: UserEventValueType.java,v 1.1 2005/09/26 21:12:46 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public abstract class UserEventValueType {

    private final String name;

    private UserEventValueType(String name) {
        this.name = name;
    }

    public String toString() {
        return name;
    }

    public static final UserEventValueType NUMSAMPLES = new UserEventValueType("Number of Samples") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventNumberValue();
        }
        public double getMaxValue(UserEvent ue) {
            return ue.getMaxUserEventNumberValue();
        }
    };

    public static final UserEventValueType MAX = new UserEventValueType("Max Value") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventMaxValue();
        }
        public double getMaxValue(UserEvent ue) {
            return ue.getMaxUserEventMaxValue();
        }
    };

    public static final UserEventValueType MIN = new UserEventValueType("Min Value") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventMinValue();
        }
        public double getMaxValue(UserEvent ue) {
            return ue.getMaxUserEventMinValue();
        }
    };

    public static final UserEventValueType MEAN = new UserEventValueType("Mean Value") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventMeanValue();
        }
        public double getMaxValue(UserEvent ue) {
            return ue.getMaxUserEventMeanValue();
        }
    };

    public static final UserEventValueType STDDEV = new UserEventValueType("Standard Deviation") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventStdDev();
        }
        public double getMaxValue(UserEvent ue) {
            return ue.getMaxUserEventStdDev();
        }
    };

    public abstract double getValue(UserEventProfile uep);

    public abstract double getMaxValue(UserEvent ue);
    
    
}
