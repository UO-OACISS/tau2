/*
 * ParaProfLispPrimitives.java
 * 
 * Title: ParaProf Author: Robert Bell Description: Lisp primatives for Jatha's
 * lisp interpreter.
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import org.jatha.Jatha;
import org.jatha.dynatype.*;
import org.jatha.compile.*;
import org.jatha.machine.*;
import edu.uoregon.tau.dms.dss.*;

public class ParaProfLispPrimitives {

    public ParaProfLispPrimitives(boolean debug) {
        super();
    }

    public static DssIterator getPrimitiveList(Jatha lisp, boolean debug) {
        Vector primatives = new Vector();
        primatives.add(new showThreadDataWindow(lisp, debug));
        primatives.add(new showMeanDataWindow(lisp, debug));
        primatives.add(new showMeanCallPathWindow(lisp, debug));
        primatives.add(new showGroupLedgerWindow(lisp, debug));
        primatives.add(new getExclusiveValues(lisp, debug));
        primatives.add(new wait(lisp, debug));
        primatives.add(new exitParaProf(lisp, debug));

        return new DssIterator(primatives);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

class showThreadDataWindow extends LispPrimitive {
    public showThreadDataWindow(Jatha lisp, boolean debug) {
        super(lisp, "SHOWTHREADDATAWINDOW", 6);
        this.lisp = lisp;
    }

    public void Execute(SECDMachine machine) {
        LispValue arg6 = machine.S.pop();
        LispValue arg5 = machine.S.pop();
        LispValue arg4 = machine.S.pop();
        LispValue arg3 = machine.S.pop();
        LispValue arg2 = machine.S.pop();
        LispValue arg1 = machine.S.pop();

        machine.S.push(result(arg1, arg2, arg3, arg4, arg5, arg6));
        machine.C.pop();
    }

    LispValue result(LispValue arg1, LispValue arg2, LispValue arg3,
            LispValue arg4, LispValue arg5, LispValue arg6) {
        System.out.println("Applying showThreadDataWindow. Args: " + arg1 + ","
                + arg2 + "," + arg3 + "," + arg4 + "," + arg5 + "," + arg6);
        ParaProfTrial trial = ParaProf.applicationManager.getTrial(
                                                                   Integer.parseInt(arg1.toString()),
                                                                   Integer.parseInt(arg2.toString()),
                                                                   Integer.parseInt(arg3.toString()));
        System.out.println("Got trial: " + trial);

        ThreadDataWindow threadDataWindow = new ThreadDataWindow(trial,
                Integer.parseInt(arg4.toString()),
                Integer.parseInt(arg5.toString()),
                Integer.parseInt(arg6.toString()), new DataSorter(
                        trial));

        trial.getSystemEvents().addObserver(threadDataWindow);
        threadDataWindow.show();
        /*
         * if(arg1.basic_integerp()) System.out.println("Integer");
         * if(arg1.basic_numberp()) System.out.println("Number");
         */
        return lisp.makeInteger(0);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

class showMeanDataWindow extends LispPrimitive {
    public showMeanDataWindow(Jatha lisp, boolean debug) {
        super(lisp, "SHOWMEANDATAWINDOW", 3);
        this.lisp = lisp;
    }

    public void Execute(SECDMachine machine) {
        LispValue arg3 = machine.S.pop();
        LispValue arg2 = machine.S.pop();
        LispValue arg1 = machine.S.pop();

        machine.S.push(result(arg1, arg2, arg3));
        machine.C.pop();
    }

    LispValue result(LispValue arg1, LispValue arg2, LispValue arg3) {
        System.out.println("Applying showMeanDataWindow. Args: " + arg1 + ","
                + arg2 + "," + arg3);
        ParaProfTrial trial = ParaProf.applicationManager.getTrial(
                                                                   Integer.parseInt(arg1.toString()),
                                                                   Integer.parseInt(arg2.toString()),
                                                                   Integer.parseInt(arg3.toString()));

        ThreadDataWindow threadDataWindow = new ThreadDataWindow(trial, -1, -1,
                -1, new DataSorter(trial));

        trial.getSystemEvents().addObserver(threadDataWindow);
        threadDataWindow.show();

        /*
         * if(arg1.basic_integerp()) System.out.println("Integer");
         * if(arg1.basic_numberp()) System.out.println("Number");
         */
        return lisp.makeInteger(0);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

class showMeanCallPathWindow extends LispPrimitive {
    public showMeanCallPathWindow(Jatha lisp, boolean debug) {
        super(lisp, "SHOWMEANCALLPATHWINDOW", 3);
        this.lisp = lisp;
    }

    public void Execute(SECDMachine machine) {
        LispValue arg3 = machine.S.pop();
        LispValue arg2 = machine.S.pop();
        LispValue arg1 = machine.S.pop();

        machine.S.push(result(arg1, arg2, arg3));
        machine.C.pop();
    }

    LispValue result(LispValue arg1, LispValue arg2, LispValue arg3) {
        System.out.println("Applying showMeanCallPathWindow. Args: " + arg1
                + "," + arg2 + "," + arg3);
        ParaProfTrial trial = ParaProf.applicationManager.getTrial(
                                                                   Integer.parseInt(arg1.toString()),
                                                                   Integer.parseInt(arg2.toString()),
                                                                   Integer.parseInt(arg3.toString()));

        if (trial.callPathDataPresent()) {
            CallPathTextWindow callPathTextWindow = new CallPathTextWindow(
                    trial, -1, -1, -1, new DataSorter(trial), 0);
            trial.getSystemEvents().addObserver(callPathTextWindow);
            callPathTextWindow.show();
        } else
            System.out.println("Lisp interface (SHOWMEANCALLPATHWINDOW): No callpath data present!");

        /*
         * if(arg1.basic_integerp()) System.out.println("Integer");
         * if(arg1.basic_numberp()) System.out.println("Number");
         */
        return lisp.makeInteger(0);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

class showGroupLedgerWindow extends LispPrimitive {
    public showGroupLedgerWindow(Jatha lisp, boolean debug) {
        super(lisp, "SHOWGROUPLEDGERWINDOW", 3);
        this.lisp = lisp;
    }

    public void Execute(SECDMachine machine) {
        LispValue arg3 = machine.S.pop();
        LispValue arg2 = machine.S.pop();
        LispValue arg1 = machine.S.pop();

        machine.S.push(result(arg1, arg2, arg3));
        machine.C.pop();
    }

    LispValue result(LispValue arg1, LispValue arg2, LispValue arg3) {
        System.out.println("Applying showGroupLedgerWindow. Args: " + arg1
                + "," + arg2 + "," + arg3);
        ParaProfTrial trial = ParaProf.applicationManager.getTrial(
                                                                   Integer.parseInt(arg1.toString()),
                                                                   Integer.parseInt(arg2.toString()),
                                                                   Integer.parseInt(arg3.toString()));
        if (trial.groupNamesPresent()) {
            //(new LedgerWindow(trial, 1)).show();
        } else {
            System.out.println("Lisp interface (SHOWGROUPSLEDGERWINDOW): No group data present!");
        }
        return lisp.makeInteger(0);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

class getExclusiveValues extends LispPrimitive {
    public getExclusiveValues(Jatha lisp, boolean debug) {
        super(lisp, "GETEXCLUSIVEVALUES", 4);
        this.lisp = lisp;
    }

    public void Execute(SECDMachine machine) {
        LispValue arg4 = machine.S.pop();
        LispValue arg3 = machine.S.pop();
        LispValue arg2 = machine.S.pop();
        LispValue arg1 = machine.S.pop();

        machine.S.push(result(arg1, arg2, arg3, arg4));
        machine.C.pop();
    }

    LispValue result(LispValue arg1, LispValue arg2, LispValue arg3,
            LispValue arg4) {
        System.out.println("Applying getTrial. Args: " + arg1 + "," + arg2
                + "," + arg3 + "," + arg4);
        Vector v = new Vector();

//        try {
//            //Get the exclusive values for the given metric. Put the result in
//            // a list, and pass it back.
//            ParaProfTrial trial = ParaProf.applicationManager.getTrial(
//                                                                       Integer.parseInt(arg1.toString()),
//                                                                       Integer.parseInt(arg2.toString()),
//                                                                       Integer.parseInt(arg3.toString()));
//
//            int metric = Integer.parseInt(arg4.toString());
//            if (trial != null) {
//                DataSource dataSource = trial.getDataSource();
//                for (Enumeration e1 = dataSource.getNCT().getNodes().elements(); e1.hasMoreElements();) {
//                    Node node = (Node) e1.nextElement();
//                    for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
//                        Context context = (Context) e2.nextElement();
//                        for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
//                            edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
//                            ListIterator l = thread.getFunctionListIterator();
//                            while (l.hasNext()) {
//                                FunctionProfile functionProfile = (FunctionProfile) l.next();
//                                if (functionProfile != null) {
//                                    v.add(lisp.makeReal(functionProfile.getExclusive(metric)));
//                                    System.out.println(functionProfile.getName()
//                                            + ": "
//                                            + functionProfile.getExclusive(metric));
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        } catch (Exception e) {
//            System.out.println("An exception was caught in: getExclusiveValues(...)");
//            e.printStackTrace();
//        }
        return lisp.makeList(v);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

class wait extends LispPrimitive {
    public wait(Jatha lisp, boolean debug) {
        super(lisp, "WAIT", 1);
        this.lisp = lisp;
    }

    public void Execute(SECDMachine machine) {
        LispValue arg1 = machine.S.pop();

        machine.S.push(result(arg1));
        machine.C.pop();
    }

    LispValue result(LispValue arg1) {
        System.out.println("Applying wait. Args: " + arg1);

        try {
            System.out.println("Waiting ... ");
            java.lang.Thread.sleep(Integer.parseInt(arg1.toString()));
            System.out.println("Done waiting.");
        } catch (Exception e) {
            if (e instanceof IllegalArgumentException)
                System.out.println("An IllegalArgumentException exception was caught in: wait(...): Pleae use range: 0-999999");
            else {
                e.printStackTrace();
            }
        }

        /*
         * if(arg1.basic_integerp()) System.out.println("Integer");
         * if(arg1.basic_numberp()) System.out.println("Number");
         */
        return lisp.makeInteger(0);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

class exitParaProf extends LispPrimitive {
    public exitParaProf(Jatha lisp, boolean debug) {
        super(lisp, "EXITPARAPROF", 1);
        this.lisp = lisp;
    }

    public void Execute(SECDMachine machine) {
        LispValue arg1 = machine.S.pop();

        machine.S.push(result(arg1));
        machine.C.pop();
    }

    LispValue result(LispValue arg1) {
        System.out.println("Applying exitParaProf. Args: " + arg1);

        ParaProf.exitParaProf(Integer.parseInt(arg1.toString()));

        /*
         * if(arg1.basic_integerp()) System.out.println("Integer");
         * if(arg1.basic_numberp()) System.out.println("Number");
         */
        return lisp.makeInteger(0);
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

