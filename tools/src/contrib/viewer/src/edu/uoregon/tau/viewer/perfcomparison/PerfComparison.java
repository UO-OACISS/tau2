
package edu.uoregon.tau.viewer.perfcomparison;

import java.util.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.viewer.apiext.*;

/**
 * @author lili
 *  This class implements trial and function comparison.
 */
public class PerfComparison{
	
    final int MEAN = 0, TOTAL = 1;
    final int EXCLUSIVE=0, EXCLUSIVEPERCENTAGE=1, INCLUSIVE=2, INCLUSIVEPERCENTAGE=3, CALLS=4, SUBROUTINES=5, PERCALL=6;
    ListIterator[] functionList;
    DataSession session;
    Vector trials;
        
    boolean showMeanValue = true, 
	showTotalValue = false,
	showGroupMeanValue = false,
	showGroupTotalValue = false; // In default, mean value is shown.

    int valueOption = 0; // 0-exclusive value (default value), 1- inclusive, 2-calls, 3-inclusive/call.
    boolean compareTrial, compareIntervalEvent; // if compare trials, then compareTrial is true; if compare functions, compareIntervalEvent is true.
    Metric metric = null;
    String selectedGroup = null;

    Hashtable groupMeanHashtable = null;
    Hashtable groupTotalHashtable = null;
    Hashtable meanIntervalEventHashtable = null;
    Hashtable totalIntervalEventHashtable = null;

    // the following seven variables are intended for setting relative bar length when drawing bar graph.
    double[] maxExclusive = new double[2]; 
    double[] maxExclusivePercentage = new double[2] ;
    double[] maxInclusive = new double[2];
    double[] maxInclusivePercentage = new double[2];
    int[] maxCalls = new int[2];
    int[] maxSubrs = new int[2] ;
    double[] maxInclusivePerCall = new double[2];

    String sortKey = "value", sortOrder = "descending";
    Vector sortedHTKeys = null;

	// There are three parameters for the constructor, db session, a vector of trials or 
	// functions to be compared, and metric name of compared performance data.
    public PerfComparison(DataSession session, Vector vec, Metric metric){
	this.session = session;
	this.metric = metric;

	if (vec.elementAt(0) instanceof ComparableTrial){ // if it is trial comparison.
		
	    this.trials = vec;
	    getIntervalEventsThruAPI(vec); // retrieve functions of the trials
	    compareTrial = true;
	    compareIntervalEvent = false;
	}
	else if (vec.elementAt(0) instanceof IntervalEvent){ // if it is function comparison.
	    functionList = new ListIterator[1];
	    functionList[0] = vec.listIterator();
	    compareTrial = false;
	    compareIntervalEvent = true;
	} 
    }	

	// get functions of a vector of trials
    public void getIntervalEventsThruAPI(Vector vec){
		if ((vec.elementAt(0)) instanceof ComparableTrial){
	    	Collections.sort(vec);
	    	functionList = new ListIterator[vec.size()];
	    	int counter = 0;
	    	Trial aTrial;
	    	for (Enumeration enu = vec.elements(); enu.hasMoreElements();){
				aTrial = ((ComparableTrial) enu.nextElement()).getTrial();		
				this.session.reset();
				this.session.setTrial(aTrial);
				functionList[counter++] = this.session.getIntervalEvents();
	    	}
		}
    }

    public int getNumOfComparedTrials(){ return trials.size();}

	// get the vector of compared trials.
    public Vector getComparedTrials(){ return trials;}

	// get maximum exclusive value. 
    public double getMaxExclusive(){ 
	if (showMeanValue) 
	    return maxExclusive[MEAN]; 
	else if (showTotalValue) 
	    return maxExclusive[TOTAL];
	else if (showGroupMeanValue){
	    Vector currentGroup = (Vector) groupMeanHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(EXCLUSIVE)).doubleValue();
	}
	else {
	    Vector currentGroup = (Vector) groupTotalHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(EXCLUSIVE)).doubleValue();
	}
    } 

	//	get maximum exclusive percentage. 
    public double getMaxExclusivePercentage(){ 
	if (showMeanValue) 
	    return maxExclusivePercentage[MEAN]; 
	else if (showTotalValue) 
	    return maxExclusivePercentage[TOTAL];
	else if (showGroupMeanValue){
	    Vector currentGroup = (Vector) groupMeanHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(EXCLUSIVEPERCENTAGE)).doubleValue();
	}
	else {
	    Vector currentGroup = (Vector) groupTotalHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(EXCLUSIVEPERCENTAGE)).doubleValue();
	}
    }
    
	//	get maximum inclusive value. 
    public double getMaxInclusive(){ 
	if (showMeanValue) 
	    return maxInclusive[MEAN]; 
	else if (showTotalValue) 
	    return maxInclusive[TOTAL];
	else if (showGroupMeanValue){
	    Vector currentGroup = (Vector) groupMeanHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(INCLUSIVE)).doubleValue();
	}
	else {
	    Vector currentGroup = (Vector) groupTotalHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(INCLUSIVE)).doubleValue();
	}
    }

	//	get maximum inclusive percentage. 
    public double getMaxInclusivePercentage(){
	if (showMeanValue) 
	    return maxInclusivePercentage[MEAN]; 
	else if (showTotalValue) 
	    return maxInclusivePercentage[TOTAL];
	else if (showGroupMeanValue){
	    Vector currentGroup = (Vector) groupMeanHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(INCLUSIVEPERCENTAGE)).doubleValue();
	}
	else {
	    Vector currentGroup = (Vector) groupTotalHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(INCLUSIVEPERCENTAGE)).doubleValue();
	}
    }
    
    // get maximum number of calls
    public int getMaxCalls(){
	if (showMeanValue) 
	    return maxCalls[MEAN]; 
	else if (showTotalValue) 
	    return maxCalls[TOTAL];
	else if (showGroupMeanValue){
	    Vector currentGroup = (Vector) groupMeanHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Integer) maxVec.elementAt(CALLS)).intValue();
	}
	else {
	    Vector currentGroup = (Vector) groupTotalHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Integer) maxVec.elementAt(CALLS)).intValue();
	}
    }
    
    // get maximum number of subroutines
    public int getMaxSubrs() {
	if (showMeanValue) 
	    return maxSubrs[MEAN]; 
	else if (showTotalValue) 
	    return maxSubrs[TOTAL];
	else if (showGroupMeanValue){
	    Vector currentGroup = (Vector) groupMeanHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Integer) maxVec.elementAt(SUBROUTINES)).intValue();
	}
	else {
	    Vector currentGroup = (Vector) groupTotalHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Integer) maxVec.elementAt(SUBROUTINES)).intValue();
	}
    }

	// get maximum inclusive time per call
    public double getMaxInclusivePerCall() {
	if (showMeanValue) 
	    return maxInclusivePerCall[MEAN]; 
	else if (showTotalValue) 
	    return maxInclusivePerCall[TOTAL];
	else if (showGroupMeanValue){
	    Vector currentGroup = (Vector) groupMeanHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(PERCALL)).doubleValue();
	}
	else {
	    Vector currentGroup = (Vector) groupTotalHashtable.get(selectedGroup);
	    Vector maxVec = (Vector) currentGroup.elementAt(0);
	    return ((Double) maxVec.elementAt(PERCALL)).doubleValue();
	}
    }

    public String getMetricName() { return metric.getName(); }

    public void setShowTotalValue(){  // after the setting, must call sortBy("value")
	showMeanValue = false;
	showTotalValue = true;
	showGroupMeanValue = false;
	showGroupTotalValue = false;
    }

    public void setShowMeanValue(){ // after the setting, must call sortBy("value")
	showMeanValue = true;
	showTotalValue = false;
	showGroupMeanValue = false;
	showGroupTotalValue = false;
    }

    public void setShowGroupMeanValue(String groupName){ // after the setting, must call sortBy("value")
	showMeanValue = false; 
	showTotalValue = false;
	showGroupMeanValue = true;
	showGroupTotalValue = false;
	selectedGroup = groupName; 
    }

    public void setShowGroupTotalValue(String groupName){ // after the setting, must call sortBy("value")
	showMeanValue = false; 
	showTotalValue = false;
	showGroupMeanValue = false;
	showGroupTotalValue = true;
	selectedGroup = groupName; 
    }

    public String getSelectedGroupName() {return selectedGroup; }

    public boolean isShowMeanValueEnabled(){return showMeanValue;  }

    public boolean isShowTotalValueEnabled(){return showTotalValue;  }

    public boolean isShowGroupMeanValueEnabled(){ return showGroupMeanValue; }

    public boolean isShowGroupTotalValueEnabled(){ return showGroupTotalValue; }

    public boolean isTrialComparison(){ return compareTrial;}
    
    public boolean isIntervalEventComparison(){return compareIntervalEvent;}

    public void setShowExclusive(){ valueOption = 0; }

    public boolean isShowExclusiveEnabled(){return (valueOption==0) ;}

    public void setShowInclusive(){ valueOption = 1;}

    public boolean isShowInclusiveEnabled(){return (valueOption==1) ;}

    public void setShowCalls(){valueOption = 2;}

    public boolean isShowCallsEnabled(){return (valueOption==2) ;}

    public void setShowInclusivePerCall(){valueOption = 3;}

    public boolean isShowInclusivePerCallEnabled(){return (valueOption==3) ;}
  
    public Hashtable getMeanValues(){
	if (meanIntervalEventHashtable == null){
	    System.out.println("Please first sort out function data.");
	    return null;
	}
	
	return meanIntervalEventHashtable;
    }

    public Hashtable getMeanGroupValues(String groupName){
	if (groupName == null) // no group name specified
	    return groupMeanHashtable; // return all groups.
	else{
	    Vector groupVec = (Vector) groupMeanHashtable.get(groupName);
	    return (Hashtable) groupVec.elementAt(1);
	} 
    }

    public Hashtable getTotalValues(){
	if (totalIntervalEventHashtable == null){
	    System.out.println("Please first sort out function data.");
	    return null;
	}
	
	return totalIntervalEventHashtable;
    }

    public Hashtable getTotalGroupValues(String groupName){
	if (groupName == null)	
	    return groupTotalHashtable;
	else {
	    Vector groupVec = (Vector) groupTotalHashtable.get(groupName);
	    return (Hashtable) groupVec.elementAt(1);
	}
    }

    public void sortoutIntervalEventData(){// the functions being sorted probably come from different trials, experiments, or applications
	clearMaxValues();
	classifyIntervalEvents(true);	
	sortBy(sortKey);
    }

    public void classifyIntervalEvents(boolean setMaxValues){// classify function data according to function and group.
	if (groupMeanHashtable == null) { groupMeanHashtable = new Hashtable(); }
	if (groupTotalHashtable == null) { groupTotalHashtable = new Hashtable(); }
	if (meanIntervalEventHashtable == null){ meanIntervalEventHashtable = new Hashtable(); }
	if (totalIntervalEventHashtable == null){ totalIntervalEventHashtable = new Hashtable(); }

	ListIterator currentFuncList;
	IntervalEvent aFunc;
	String funcKey, groupKey;
	Vector groupMeanValues, groupTotalValues, meanValue,  totalValue;
	Hashtable meanIntervalEventHTOfAGroup, totalIntervalEventHTOfAGroup;

	for (int k =0; k<functionList.length; k++){
	    currentFuncList = functionList[k];
	    while (currentFuncList.hasNext()){
		aFunc = (IntervalEvent) currentFuncList.next();
		funcKey = aFunc.getName();
		groupKey = aFunc.getGroup();
		
		groupMeanValues = (Vector) groupMeanHashtable.get(groupKey);
		groupTotalValues = (Vector) groupTotalHashtable.get(groupKey);
		meanValue = (Vector) meanIntervalEventHashtable.get(funcKey);
		totalValue = (Vector) totalIntervalEventHashtable.get(funcKey);
		
		if (groupMeanValues == null){// groupMeanValues is a two-element vector, its first element is the vector of maximum values of the group functions, the second element is a hashtable of group function data.
		    groupMeanValues = new Vector(2);
		    Vector maxMeanVec = new Vector(7);

		    maxMeanVec.add(new Double(0.0));
		    maxMeanVec.add(new Double(0.0));
		    maxMeanVec.add(new Double(0.0));
		    maxMeanVec.add(new Double(0.0));
		    maxMeanVec.add(new Integer(0));
		    maxMeanVec.add(new Integer(0));
		    maxMeanVec.add(new Double(0.0));

		    // add first element
		    groupMeanValues.add(maxMeanVec);
		    
		    meanIntervalEventHTOfAGroup = new Hashtable();
		    // add second element 
		    groupMeanValues.add(meanIntervalEventHTOfAGroup);
			
		    groupMeanHashtable.put(groupKey, groupMeanValues);
		}
		else
		    meanIntervalEventHTOfAGroup = (Hashtable) (groupMeanValues.elementAt(1));		

		if (groupTotalValues == null){ // groupTotalValues is a two-element vector, its first element is the vector of maximum values of the group functions, the second element is a hashtable of group function data.
		    groupTotalValues = new Vector(2);
		    Vector maxTotalVec = new Vector(7);
		    
		    maxTotalVec.add(new Double(0.0));
		    maxTotalVec.add(new Double(0.0));
		    maxTotalVec.add(new Double(0.0));
		    maxTotalVec.add(new Double(0.0));
		    maxTotalVec.add(new Integer(0));
		    maxTotalVec.add(new Integer(0));
		    maxTotalVec.add(new Double(0.0));

		    groupTotalValues.add(maxTotalVec);

		    totalIntervalEventHTOfAGroup = new Hashtable();
		    groupTotalValues.add(totalIntervalEventHTOfAGroup);

		    groupTotalHashtable.put(groupKey, groupTotalValues);
		}
		else
		    totalIntervalEventHTOfAGroup = (Hashtable) (groupTotalValues.elementAt(1));
		
		if (meanValue == null){
		    meanValue = new Vector();
		    meanIntervalEventHashtable.put(funcKey, meanValue);

		    meanIntervalEventHTOfAGroup.put(funcKey, meanValue);		    
		}
		ComparisonWindowIntervalEvent aMeanSummary = new ComparisonWindowIntervalEvent(aFunc, aFunc.getMeanSummary());
	    
		meanValue.add(aMeanSummary);

		if (totalValue == null){
		    totalValue = new Vector();
		    totalIntervalEventHashtable.put(funcKey, totalValue);

		    totalIntervalEventHTOfAGroup.put(funcKey, totalValue);
		    
		}

		ComparisonWindowIntervalEvent aTotalSummary = new ComparisonWindowIntervalEvent(aFunc, aFunc.getTotalSummary());		
		totalValue.add(aTotalSummary);
			
		double tmpValue;
		int tmpIntValue;
		Vector maxGroupMean = (Vector) (groupMeanValues.elementAt(0));
		Vector maxGroupTotal = (Vector) (groupTotalValues.elementAt(0));

		if (setMaxValues){
				
		    // mean stuff
		    if ((tmpValue=aMeanSummary.getExclusive())>maxExclusive[MEAN])
			maxExclusive[MEAN] = tmpValue;

		    if (tmpValue>((Double)(maxGroupMean.elementAt(EXCLUSIVE))).doubleValue())
			maxGroupMean.setElementAt(new Double(tmpValue), EXCLUSIVE);
		
		    if ((tmpValue=aMeanSummary.getExclusivePercentage())>maxExclusivePercentage[MEAN])
			maxExclusivePercentage[MEAN] = tmpValue;

		    if (tmpValue>((Double)(maxGroupMean.elementAt(EXCLUSIVEPERCENTAGE))).doubleValue())
			maxGroupMean.setElementAt(new Double(tmpValue), EXCLUSIVEPERCENTAGE);

		    if ((tmpValue=aMeanSummary.getInclusive())>maxInclusive[MEAN])
			maxInclusive[MEAN] = tmpValue;

		    if (tmpValue>((Double)(maxGroupMean.elementAt(INCLUSIVE))).doubleValue())
			maxGroupMean.setElementAt(new Double(tmpValue), INCLUSIVE);

		    if ((tmpValue=aMeanSummary.getInclusivePercentage())>maxInclusivePercentage[MEAN])
			maxInclusivePercentage[MEAN] = tmpValue;

		    if (tmpValue>((Double)(maxGroupMean.elementAt(INCLUSIVEPERCENTAGE))).doubleValue())
			maxGroupMean.setElementAt(new Double(tmpValue), INCLUSIVEPERCENTAGE);

		    if ((tmpIntValue=aMeanSummary.getNumCalls())>maxCalls[MEAN])
			maxCalls[MEAN] = tmpIntValue;

		    if (tmpIntValue>((Integer)(maxGroupMean.elementAt(CALLS))).intValue())
			maxGroupMean.setElementAt(new Integer(tmpIntValue), CALLS);

		    if ((tmpIntValue=aMeanSummary.getNumSubroutines())>maxSubrs[MEAN])
			maxSubrs[MEAN] = tmpIntValue;

		    if (tmpIntValue>((Integer)(maxGroupMean.elementAt(SUBROUTINES))).intValue())
			maxGroupMean.setElementAt(new Integer(tmpIntValue), SUBROUTINES);

		    if ((tmpValue=aMeanSummary.getInclusivePerCall())>maxInclusivePerCall[MEAN])
			maxInclusivePerCall[MEAN] = tmpValue;

		    if (tmpValue>((Double)(maxGroupMean.elementAt(PERCALL))).doubleValue())
			maxGroupMean.setElementAt(new Double(tmpValue), PERCALL);
		
		    // total stuff
		    if ((tmpValue=aTotalSummary.getExclusive())>maxExclusive[TOTAL])
			maxExclusive[TOTAL] = tmpValue;
		
		    if (tmpValue>((Double)(maxGroupTotal.elementAt(EXCLUSIVE))).doubleValue())
			maxGroupTotal.setElementAt(new Double(tmpValue), EXCLUSIVE);
	
		    if ((tmpValue=aTotalSummary.getExclusivePercentage())>maxExclusivePercentage[TOTAL])
			maxExclusivePercentage[TOTAL] = tmpValue;

		    if (tmpValue>((Double)(maxGroupTotal.elementAt(EXCLUSIVEPERCENTAGE))).doubleValue())
			maxGroupTotal.setElementAt(new Double(tmpValue), EXCLUSIVEPERCENTAGE);

	
		    if ((tmpValue=aTotalSummary.getInclusive())>maxInclusive[TOTAL])
			maxInclusive[TOTAL] = tmpValue;

		    if (tmpValue>((Double)(maxGroupTotal.elementAt(INCLUSIVE))).doubleValue())
			maxGroupTotal.setElementAt(new Double(tmpValue), INCLUSIVE);

	
		    if ((tmpValue=aTotalSummary.getInclusivePercentage())>maxInclusivePercentage[TOTAL])
			maxInclusivePercentage[TOTAL] = tmpValue;
		
		    if (tmpValue>((Double)(maxGroupTotal.elementAt(INCLUSIVEPERCENTAGE))).doubleValue())
			maxGroupTotal.setElementAt(new Double(tmpValue), INCLUSIVEPERCENTAGE);

		    if ((tmpIntValue=aTotalSummary.getNumCalls())>maxCalls[TOTAL])
			maxCalls[TOTAL] = tmpIntValue;

		    if (tmpIntValue>((Integer)(maxGroupTotal.elementAt(CALLS))).intValue())
			maxGroupTotal.setElementAt(new Integer(tmpIntValue), CALLS);

		    if ((tmpIntValue=aTotalSummary.getNumSubroutines())>maxSubrs[TOTAL])
			maxSubrs[TOTAL] = tmpIntValue;

		    if (tmpIntValue>((Integer)(maxGroupTotal.elementAt(SUBROUTINES))).intValue())
			maxGroupTotal.setElementAt(new Integer(tmpIntValue), SUBROUTINES);
		
		    if ((tmpValue=aTotalSummary.getInclusivePerCall())>maxInclusivePerCall[TOTAL])
			maxInclusivePerCall[TOTAL] = tmpValue;

		    if (tmpValue>((Double)(maxGroupTotal.elementAt(PERCALL))).doubleValue())
			maxGroupTotal.setElementAt(new Double(tmpValue), PERCALL);

		}	
	    }
	}		
    }

    public void setSortKey(String key){ sortKey = key;}

    public String getSortKey() { return sortKey; }

    public void setSortOrder(String order){ sortOrder = order;}

    public String getSortOrder(){ return sortOrder; }
    
    
    // sort functions according to value or function name. If sort in values, then compare function groups according to the first value in the groups.
    public void sortBy(String key){
	Hashtable currentHT;
	String strKey, vecKey;
	Vector value, vecValue;
	ComparisonWindowIntervalEvent function, vecIntervalEvent;

	setSortKey(key);

	if (isShowMeanValueEnabled())
	    currentHT = getMeanValues();
	else if (isShowTotalValueEnabled())
	    currentHT = getTotalValues();
	else if (isShowGroupMeanValueEnabled())
	    currentHT = getMeanGroupValues(getSelectedGroupName());
	else 
	    currentHT = getTotalGroupValues(getSelectedGroupName());
	
	sortedHTKeys = new Vector();

	if (key.equals("value")){
	    
	    for(Enumeration e1=currentHT.keys(); e1.hasMoreElements();){
		strKey = (String) e1.nextElement();
		value = (Vector) currentHT.get(strKey);
		function = (ComparisonWindowIntervalEvent) value.elementAt(0);
	    
		if (sortedHTKeys.isEmpty()){
		    sortedHTKeys.add(strKey);
		    continue;
		}
	    
		int i=0;
		
		for(; i<sortedHTKeys.size(); i++){
		    vecKey = (String) sortedHTKeys.elementAt(i);
		    vecValue = (Vector) currentHT.get(vecKey);
		    vecIntervalEvent = (ComparisonWindowIntervalEvent) vecValue.elementAt(0);

		    if (isShowExclusiveEnabled()){
			if (sortOrder.equals("descending")){
			    if (vecIntervalEvent.getExclusive()<function.getExclusive()){
				sortedHTKeys.insertElementAt(strKey, i);
				break;
			    }
			}
			else
			    if (vecIntervalEvent.getExclusive()>function.getExclusive()){
				sortedHTKeys.insertElementAt(strKey, i);
				break;
			    }
		    }
		    else if (isShowInclusiveEnabled()){
			if (sortOrder.equals("descending")){
			    if (vecIntervalEvent.getInclusive()<function.getInclusive()){
				sortedHTKeys.insertElementAt(strKey, i);
				break;
			    }
			}
			else
			    if (vecIntervalEvent.getInclusive()>function.getInclusive()){
				sortedHTKeys.insertElementAt(strKey, i);
				break;
			    }
		    }
		    else if (isShowCallsEnabled()){
			if (sortOrder.equals("descending")){
			    if (vecIntervalEvent.getNumCalls()<function.getNumCalls()){
				sortedHTKeys.insertElementAt(strKey, i);
				break;
			    }
			}
			else 
			    if (vecIntervalEvent.getNumCalls()>function.getNumCalls()){
				sortedHTKeys.insertElementAt(strKey, i);
				break;
			    }
		    }
		    else{
			if (sortOrder.equals("descending")){
			    if (vecIntervalEvent.getInclusivePerCall()<function.getInclusivePerCall()){
				sortedHTKeys.insertElementAt(strKey, i);
				break;
			    }
			}
			else
			    if (vecIntervalEvent.getInclusivePerCall()>function.getInclusivePerCall()){
				sortedHTKeys.insertElementAt(strKey, i);
				break;
			    }
		    }
		}
		if (i>=sortedHTKeys.size())
		    sortedHTKeys.add(strKey);
	    }
	}
	else if (key.equals("function name")){
	    for(Enumeration e1=currentHT.keys(); e1.hasMoreElements();){
		strKey = (String) e1.nextElement();

		if (sortedHTKeys.isEmpty()){
		    sortedHTKeys.add(strKey);
		    continue;
		}

		int i=0;
		
		for(; i<sortedHTKeys.size(); i++){
		    vecKey = (String) sortedHTKeys.elementAt(i);
		    if (sortOrder.equals("descending")){
			if (strKey.compareTo(vecKey)>0){
			    sortedHTKeys.insertElementAt(strKey, i);
			    break;
			}
		    }
		    else 
			if (strKey.compareTo(vecKey)<0){
			    sortedHTKeys.insertElementAt(strKey, i);
			    break;
			}
		}
		if (i>=sortedHTKeys.size())
		    sortedHTKeys.add(strKey);
	    }
	}
	
    }

    public Enumeration getSortedHTKeys(){
	if (sortedHTKeys == null)
	    return null;
	else{	    
	    return sortedHTKeys.elements();
	}
    }

    public void resort(){
	sortBy(getSortKey());
    }

    private void clearMaxValues(){
	for (int i=0;i<2;i++){
	    maxExclusive[i] = 0;
	    maxExclusivePercentage[i] = 0;
	    maxInclusive[i] = 0;
	    maxInclusivePercentage[i] = 0;
	    maxCalls[i] = 0;
	    maxSubrs[i] = 0;
	    maxInclusivePerCall[i] = 0;
	}	    
    } 

	// open a comparison window displaying comparison results.
    public void displayComparisonResults(){
	
	ComparisonWindow cWindow = new ComparisonWindow(this);
	cWindow.setVisible(true);	
    }


    /*public void test(){
	int counter = 0;
	for(Enumeration e1=groupMeanHashtable.keys(); e1.hasMoreElements();){
	    String strKey = (String) e1.nextElement();
	    System.out.println(counter++ + "   " + strKey+":");

	    Vector aGroup = (Vector) groupMeanHashtable.get(strKey);
	    Hashtable groupHT = (Hashtable) aGroup.elementAt(1);

	    for(Enumeration e2=groupHT.keys(); e2.hasMoreElements();){
		String funcKey = (String) e2.nextElement();
		System.out.println(funcKey);
	    }	    
	    System.out.println("**********************");
	}
    }*/
        
}
