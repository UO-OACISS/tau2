/*
 * DerivedMetrics.java
 * 
 * 
 * Title: ParaProf Author: Robert Bell Description:
 */

package edu.uoregon.tau.paraprof;

import javax.swing.*;
import edu.uoregon.tau.dms.dss.*;
import java.util.*;

public class DerivedMetrics {

    public DerivedMetrics() {
    }

    public static ParaProfMetric applyOperation(ParaProfMetric operand1, Object operand2, String inOperation) {

        try {
            boolean constant = false; //Indicates whether we are just applying
            // a constant
            //as an argument for the second operand.
            double constantValue = 0.00;
            ParaProfTrial trialOpA = null;
            ParaProfTrial trialOpB = null;
            int opA = -1;
            int opB = -1;

            if (operand2 instanceof String) {
                constantValue = Double.parseDouble((((String) operand2).substring(4)).trim());
                constant = true;
            }

            trialOpA = operand1.getTrial();
            opA = operand1.getID();
            if (!constant) {
                trialOpB = ((ParaProfMetric) operand2).getTrial();
                opB = ((ParaProfMetric) operand2).getID();
            }

            //We do not support metric from different trials yet. Check for
            // this.
            if ((!constant) && (trialOpA != trialOpB)) {
                JOptionPane.showMessageDialog(ParaProf.paraProfManager,
                        "Sorry, please select metrics from the same trial!", "ParaProf Error",
                        JOptionPane.ERROR_MESSAGE);
                return null;
            }

            String newMetricName = null;
            int operation = -1;
            if (inOperation.equals("Add")) {
                operation = 0;
                newMetricName = " + ";
            } else if (inOperation.equals("Subtract")) {
                operation = 1;
                newMetricName = " - ";
            } else if (inOperation.equals("Multiply")) {
                operation = 2;
                newMetricName = " * ";
            } else if (inOperation.equals("Divide")) {
                operation = 3;
                newMetricName = " / ";
            } else {
                System.out.println("Wrong operation type");
            }

            if (constant)
                newMetricName = ((ParaProfMetric) trialOpA.getMetrics().elementAt(opA)).getName()
                        + newMetricName + constantValue;
            else
                newMetricName = ((ParaProfMetric) trialOpA.getMetrics().elementAt(opA)).getName()
                        + newMetricName + ((ParaProfMetric) trialOpA.getMetrics().elementAt(opB)).getName();

            System.out.println("Metric name is: " + newMetricName);

            ParaProfMetric newMetric = trialOpA.addMetric();
            newMetric.setTrial(trialOpA);
            newMetric.setName(newMetricName);
            newMetric.setDerivedMetric(true);
            int metric = newMetric.getID();
            trialOpA.setSelectedMetricID(metric);

            Iterator l = trialOpA.getTrialData().getFunctions();

            //TODO: I'm confused as to whether or not these both need to be
            // done, shouldn't the bottom one do the top stuff anyway?
            while (l.hasNext()) {
                Function f = (Function) l.next();
                f.incrementStorage();
            }
            trialOpA.getTrialData().increaseVectorStorage();

            edu.uoregon.tau.dms.dss.Thread meanThread;
            
            meanThread = trialOpA.getDataSource().getMeanData();
            
            meanThread.incrementStorage();
            
            
            l = meanThread.getFunctionListIterator();
            while (l.hasNext()) {
                FunctionProfile functionProfile = (FunctionProfile) l.next();
                if (functionProfile != null) {
                    functionProfile.incrementStorage();
                }
            }

            //######
            //Calculate the raw values.
            //We only need establish exclusive and inclusive time.
            //The rest of the data can either be computed from these,
            //or is already in the system (number of calls as an example
            //of the latter.
            //######

            Node node;
            Context context;
            edu.uoregon.tau.dms.dss.Thread thread;

            for (Enumeration e1 = trialOpA.getNCT().getNodes().elements(); e1.hasMoreElements();) {
                node = (Node) e1.nextElement();
                for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
                    context = (Context) e2.nextElement();
                    for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
                        thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
                        thread.incrementStorage();
                        l = thread.getFunctionListIterator();
                        while (l.hasNext()) {
                            FunctionProfile functionProfile = (FunctionProfile) l.next();
                            if (functionProfile != null) {
                                Function function = functionProfile.getFunction();
                                functionProfile.incrementStorage();

                                double d1 = 0.0;
                                double d2 = 0.0;
                                double result = 0.0;

                                d1 = functionProfile.getExclusive(opA);
                                if (!constant) {
                                    d2 = functionProfile.getExclusive(opB);
                                    result = DerivedMetrics.apply(operation, d1, d2);
                                } else
                                    result = DerivedMetrics.apply(operation, d1, constantValue);

                                functionProfile.setExclusive(metric, result);
                                //Now do the global mapping element exclusive
                                // stuff.
                                if ((function.getMaxExclusive(metric)) < result)
                                    function.setMaxExclusive(metric, result);

                                d1 = functionProfile.getInclusive(opA);
                                if (!constant) {
                                    d2 = functionProfile.getInclusive(opB);
                                    result = DerivedMetrics.apply(operation, d1, d2);
                                } else
                                    result = DerivedMetrics.apply(operation, d1, constantValue);

                                functionProfile.setInclusive(metric, result);
                                //Now do the global mapping element inclusive
                                // stuff.

                                if ((result > thread.getMaxInclusive(metric))) {
                                    thread.setMaxInclusive(metric, result);
                                }

                                if ((function.getMaxInclusive(metric)) < result)
                                    function.setMaxInclusive(metric, result);
                            }
                        }
                        thread.setThreadData(metric);
                    }
                }
            }
            //Done with this metric, let the global mapping compute the mean
            // values.
            trialOpA.setMeanData(metric);
            return newMetric;
        } catch (Exception e) {
            if (e instanceof NumberFormatException) {
                //Display an error
                JOptionPane.showMessageDialog(ParaProf.paraProfManager,
                        "Did not recognize arguments! Note: DB apply not supported.",
                        "Argument Error!", JOptionPane.ERROR_MESSAGE);
            } else {
                UtilFncs.systemError(new ParaProfError(DerivedMetrics.staticToString()
                        + ": applyOperation(...)", "An error occurred ... please see console!",
                        "An error occured while trying to apply this operation!", null, e,
                        ParaProf.paraProfManager, null, null, true, false, false), null, null);
            }
            return null;
        }
    }

    public static double apply(int op, double arg1, double arg2) {
        double d = 0.0;
        switch (op) {
        case (0):
            d = arg1 + arg2;
            break;
        case (1):
            if (arg1 > arg2) {
                d = arg1 - arg2;
            }
            break;
        case (2):
            d = arg1 * arg2;
            break;
        case (3):
            if (arg2 != 0) {
                return arg1 / arg2;
            }
            break;
        default:
            UtilFncs.systemError(null, null, "Unexpected opertion - PPML01 value: " + op);
        }
        return d;
    }

    public static String staticToString() {
        return (new DerivedMetrics()).toString();
    }

    public String toString() {
        return this.getClass().getName();
    }
}