/*
 * Created on Mar 15, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.paraprof.enums;

import edu.uoregon.tau.paraprof.ParaProfMetric;

/**
 * @author amorris
 *
 * TODO ...
 */
public abstract class UnitType {

    private final String name;
    private UnitType(String name) { this.name = name; }
    public String toString() { return name; }
    
    public static final UnitType MICROSECONDS = new UnitType("Microseconds") {
        public String getUnitString(double value, ParaProfMetric metric) {
            return "";
        }
    };
    
    
    
    public abstract String getUnitString(double value, ParaProfMetric metric);
    
}
