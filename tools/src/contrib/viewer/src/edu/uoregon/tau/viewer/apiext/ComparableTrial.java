package edu.uoregon.tau.viewer.apiext;

import edu.uoregon.tau.dms.dss.*;
import java.util.*;

public class ComparableTrial implements  Comparable{
	private Trial trial;

	public ComparableTrial(Trial aTrial){
		this.trial = aTrial;
	}

	public Trial getTrial(){
		return trial;
	}

	public int compareTo(Object aTrial){
                int numOfProc = trial.getNodeCount()*trial.getNumContextsPerNode()*trial.getNumThreadsPerContext();
                Trial anotherTrial = ((ComparableTrial) aTrial).getTrial();
                int another = anotherTrial.getNodeCount()* anotherTrial.getNumContextsPerNode()* anotherTrial.getNumThreadsPerContext();
                return numOfProc - another;
        }
}
