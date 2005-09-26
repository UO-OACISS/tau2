package edu.uoregon.tau.paraprof;

import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

public class ComparisonBarChartWindow extends JFrame {

    
    private List ppTrials = new ArrayList();
    
    public ComparisonBarChartWindow() {
        
        
    }
    
    
    public void AddTrial(ParaProfTrial ppTrial) {
        ppTrials.add(ppTrial);
    }
    
    
   
    
    
    

}
