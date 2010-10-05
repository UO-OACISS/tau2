package edu.uoregon.tau.paraprof;

import edu.uoregon.tau.perfdmf.Function;

public class FunctionOrdering {

    private Function functions[];
    private boolean mask[];
    //private Function order[];
    
    public FunctionOrdering(DataSorter dataSorter) {
        
        
    }

    public Function[] getFunctions() {
        return functions;
    }

    public void setFunctions(Function[] functions) {
        this.functions = functions;
    }

    public boolean[] getMask() {
        return mask;
    }

    public void setMask(boolean[] mask) {
        this.mask = mask;
    }

    public Function[] getOrder() {
        //return order;
        return functions;
    }

//    public void setOrder(Function[] order) {
//        this.order = order;
//    }
    
    
    
}
