package edu.uoregon.tau.paraprof.enums;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.dms.dss.*;

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
    };

    public static final UserEventValueType MAX = new UserEventValueType("Max Value") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventMaxValue();
        }
    };

    public static final UserEventValueType MIN = new UserEventValueType("Min Value") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventMinValue();
        }
    };

    public static final UserEventValueType MEAN = new UserEventValueType("Mean Value") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventMeanValue();
        }
    };

    public static final UserEventValueType STDDEV = new UserEventValueType("Standard Deviation") {
        public double getValue(UserEventProfile uep) {
            return uep.getUserEventStdDev();
        }
    };

    public abstract double getValue(UserEventProfile uep);

}
