/*
 * DerivedMetrics.java
 * 
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
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

            //System.out.println("Metric name is: " + newMetricName);

            ParaProfMetric newMetric = trialOpA.addMetric();
            newMetric.setTrial(trialOpA);
            newMetric.setName(newMetricName);
            newMetric.setDerivedMetric(true);
            int metric = newMetric.getID();
//            trialOpA.setSelectedMetricID(metric);

            Iterator l = trialOpA.getDataSource().getFunctions();


            edu.uoregon.tau.dms.dss.Thread meanThread;
            
            meanThread = trialOpA.getDataSource().getMeanData();
            meanThread.incrementStorage();

            edu.uoregon.tau.dms.dss.Thread totalThread;
            
            totalThread = trialOpA.getDataSource().getTotalData();
            
            totalThread.incrementStorage();
            
            
            l = meanThread.getFunctionProfileIterator();
            while (l.hasNext()) {
                FunctionProfile functionProfile = (FunctionProfile) l.next();
                if (functionProfile != null) {
                    functionProfile.incrementStorage();
                }
            }


            l = totalThread.getFunctionProfileIterator();
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

            
            for (Iterator it = trialOpA.getDataSource().getNodes(); it.hasNext();) {
                Node node = (Node) it.next();
                for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                    Context context = (Context) it2.next();
                    for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                        edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) it3.next();
                        thread.incrementStorage();
                        l = thread.getFunctionProfileIterator();
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

//                                if ((function.getMaxExclusive(metric)) < result)
//                                    function.setMaxExclusive(metric, result);

                                d1 = functionProfile.getInclusive(opA);
                                if (!constant) {
                                    d2 = functionProfile.getInclusive(opB);
                                    result = DerivedMetrics.apply(operation, d1, d2);
                                } else
                                    result = DerivedMetrics.apply(operation, d1, constantValue);

                                functionProfile.setInclusive(metric, result);

                                functionProfile.setInclusivePerCall(metric, functionProfile.getInclusive(metric) / functionProfile.getNumCalls());

//                                if ((result > thread.getMaxInclusive(metric))) {
//                                    thread.setMaxInclusive(metric, result);
//                                }

//                                if ((function.getMaxInclusive(metric)) < result)
//                                    function.setMaxInclusive(metric, result);
                            }
                        }
                        //thread.setThreadData(metric);
                    }
                }
            }
            //Done with this metric, compute the mean values.
            trialOpA.setMeanData(metric);
            return newMetric;
        } catch (Exception e) {
            if (e instanceof NumberFormatException) {
                //Display an error
                JOptionPane.showMessageDialog(ParaProf.paraProfManager,
                        "Did not recognize arguments! Note: DB apply not supported.",
                        "Argument Error!", JOptionPane.ERROR_MESSAGE);
            } else {
                ParaProfUtils.handleException(e);
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
            throw new RuntimeException("Unexpected operation type: " + op);
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