package edu.uoregon.tau.perfdmf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * XML Handler for cube data
 *
 * 
 * <P>CVS $Id: HPCToolkitXMLHandler.java,v 1.2 2006/12/28 03:05:59 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 * @see HPCToolkitDataSource.java
 */
public class HPCToolkitXMLHandler extends DefaultHandler {

    private HPCToolkitDataSource dataSource;

    private Function currentFunction;
    private int numMetrics = 0;

    private String currentFile;

    private Map<String, Metric> metricMap = new HashMap<String, Metric>();
    private Map<String, HPCMetric> hpcMetricMap = new HashMap<String, HPCMetric>();

    private Thread theThread;
    private ArrayList<Thread> threads;

    private Stack<String> nameStack = new Stack<String>();

    private Group defaultGroup;
    private Group callpathGroup;

    private String version=null;

    private HashMap<String,String> procedures;


    public HPCToolkitXMLHandler(HPCToolkitDataSource dataSource) {
	super();

	this.dataSource = dataSource;
	this.procedures = new HashMap<String, String>();


    }

    public void startDocument() throws SAXException {
	threads = new ArrayList<Thread>();
	theThread = dataSource.addThread(0, 0, 0);
	threads.add(theThread);

	defaultGroup = dataSource.addGroup("HPC_DEFAULT");
	callpathGroup = dataSource.addGroup("HPC_CALLPATH");

    }

    public void endDocument() throws SAXException {
    }


    private FunctionProfile createFunctionProfile(Thread thread, Function function) {
	FunctionProfile fp = thread.getFunctionProfile(function);
	if (fp == null) {
	    fp = new FunctionProfile(function, numMetrics);
	    thread.addFunctionProfile(fp);
	}

	return fp;

    }


    @SuppressWarnings("unchecked")
    private void stackName(String name) {
	String origName = name;


	Stack<String> stackCopy = (Stack<String>) nameStack.clone();
	while (stackCopy.size() != 0) {
	    name = stackCopy.pop() + " => " + name;
	}
	nameStack.push(origName);

	Function f = dataSource.addFunction(name);
	currentFunction = f;

	if (name.indexOf("=>") != -1) {
	    f.addGroup(callpathGroup);
	} else {
	    f.addGroup(defaultGroup);
	}

	// create the flat profile now
	//FunctionProfile flat = 
	getFlatFunctionProfile(theThread, f);

    }


    // given A => B => C, this retrieves the FP for C
    private FunctionProfile getFlatFunctionProfile(Thread thread, Function function) {
//	if (!function.isCallPathFunction()) {
//	    return null;
//	}
	Function childFunction;
if(function.getName().lastIndexOf("=>")!= -1){
	String childName = function.getName().substring(function.getName().lastIndexOf("=>") + 2).trim();
	 childFunction = dataSource.addFunction(childName);
}else{
	 childFunction = dataSource.addFunction(function.getName());
    
}
	childFunction.addGroup(defaultGroup);

	FunctionProfile childFP = thread.getFunctionProfile(childFunction);
	if (childFP == null) {
	    childFP = new FunctionProfile(childFunction, dataSource.getNumberOfMetrics());
	    thread.addFunctionProfile(childFP);
	}
	return childFP;

    }

    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
	if(version == null){
	    if(localName.equalsIgnoreCase("HPCToolkitExperiment")){
		version= attributes.getValue("version");
	    }else if(localName.equalsIgnoreCase("CSPROFILE")){
		version= attributes.getValue("version");
	    }
	}else if(version.startsWith("1")){

	    if (localName.equalsIgnoreCase("METRIC")) {



		String displayName = attributes.getValue("displayName");
		String shortName = attributes.getValue("shortName");
		if (displayName == null){
		    displayName = attributes.getValue("nativeName");
		}

		Metric metric = dataSource.addMetric(displayName);

		metricMap.put(shortName, metric);
		numMetrics++;

	    } else if (localName.equalsIgnoreCase("PGM")) {
		stackName(attributes.getValue("n"));
	    } else if (localName.equalsIgnoreCase("LM")) {
		stackName("Load module " + attributes.getValue("n"));
	    } else if (localName.equalsIgnoreCase("L")) {  // <L b="103" e="106">
		stackName("loop at " + currentFile + ": " + attributes.getValue("b") + "-" + attributes.getValue("e"));
	    } else if (localName.equalsIgnoreCase("LN")) { // <LN b="81" e="81">
		stackName(currentFile + ": " + attributes.getValue("b"));
	    } else if (localName.equalsIgnoreCase("F")) {
		stackName(attributes.getValue("n"));
		currentFile = attributes.getValue("n");
	    } else if (localName.equalsIgnoreCase("P")) {
		stackName(attributes.getValue("n"));
	    } else if (localName.equalsIgnoreCase("M")) {
		String metricID = attributes.getValue("n");
		Metric metric = metricMap.get(metricID);
		double value = Double.parseDouble(attributes.getValue("v"));

		FunctionProfile fp = createFunctionProfile(theThread, currentFunction);

		fp.setInclusive(metric.getID(), value);
		fp.setExclusive(metric.getID(), value);

	    }
	}else if(version.startsWith("2")){
	    if (localName.equalsIgnoreCase("METRIC")) {
		HPCMetric hpcMetric = new HPCMetric(attributes);
		 hpcMetricMap.put(hpcMetric.id,hpcMetric);

		 Matcher mat = Pattern.compile(".*(\\([IE]\\)).*").matcher(hpcMetric.getName());
			if (mat.matches()){
			    String newName =  hpcMetric.getName().substring(0, mat.start(1));
			    hpcMetric.setName(newName.trim());
			}
			
		//Is this a thread metric?
		Matcher m = Pattern.compile(".*(\\[(\\d*),(\\d*)\\]).*").matcher(hpcMetric.getName());
		if (m.matches()){
		   
		    String newName =  hpcMetric.getName().substring(0, m.start(1));
		    
		    hpcMetric.setName(newName.trim());
		    int thread = Integer.valueOf(m.group(2));
		    Thread newThread;
		    if(threads.size() <= thread){
			newThread = dataSource.addThread(thread, 0, 0);
			threads.add(newThread);
		    }else{
			newThread = threads.get(thread);
		    }
		   hpcMetric.setThread( newThread);
		
		}else{ //A summary method 
		    hpcMetric.setThread(theThread);
		}
		
		//Is this just inclusive or exclusive metric?
		String pID = hpcMetric.getPartnerID();
		if(pID != null && hpcMetricMap.containsKey(pID)){
		    HPCMetric partner = hpcMetricMap.get(pID);
		    hpcMetric.setPartner(partner);
		    hpcMetric.setMetric(partner.getMetric());
		    partner.setPartner(hpcMetric);
		   
		}else{
		    Metric metric = dataSource.addMetric(hpcMetric.name);
		    hpcMetric.setMetric(metric);
		    numMetrics++;
		}
		 

	    }else if(localName.equalsIgnoreCase("PF")){
		String id = attributes.getValue("n");
		if(procedures.containsKey(id)){
		    stackName(procedures.get(id));
		}else{
		    stackName(id);
		}
	    }else if(localName.equalsIgnoreCase("Procedure")){
		procedures.put(attributes.getValue("i"),attributes.getValue("n"));


	    } else if (localName.equalsIgnoreCase("M")) {
		String metricID = attributes.getValue("n");
		HPCMetric metric = hpcMetricMap.get(metricID);
		double value = Double.parseDouble(attributes.getValue("v"));
		metric.setValue(value);
		theThread = metric.getThread();
		if(metric.hasPartner()){
		    HPCMetric partner = metric.partner;
		    if(partner.hasValue()){
		
			FunctionProfile fp = createFunctionProfile(theThread, getCurrentFunction());
			partner.setFunctionValue(fp);
			metric.setFunctionValue(fp);
			metric.resetValue();
			fp.setNumCalls(fp.getNumCalls()+1);

		    }
		}else{
		    FunctionProfile fp = createFunctionProfile(theThread, getCurrentFunction());
		    metric.setFunctionValue(fp);
		    metric.resetValue();
			fp.setNumCalls(fp.getNumCalls()+1);

		}
	    }
	    
	}

    }


    private Function getCurrentFunction() {
	if(currentFunction == null){
	    String name ="Default";

		Function f = dataSource.addFunction(name);
		currentFunction = f;

		if (name.indexOf("=>") != -1) {
		    f.addGroup(callpathGroup);
		} else {
		    f.addGroup(defaultGroup);
		}
		getFlatFunctionProfile(theThread, f);
	}
	return currentFunction;
    }

    public void endElement(String uri, String localName, String qName) throws SAXException {
	if(version.startsWith("1")){
	    if (localName.equalsIgnoreCase("PGM")) {
		nameStack.pop();
	    } else if (localName.equalsIgnoreCase("LM")) {
		nameStack.pop();
	    } else if (localName.equalsIgnoreCase("P")) {
		nameStack.pop();
	    } else if (localName.equalsIgnoreCase("F")) {
		nameStack.pop();
	    } else if (localName.equalsIgnoreCase("L")) {
		nameStack.pop();
	    } else if (localName.equalsIgnoreCase("LN")) {
		nameStack.pop();
	    }
	}else if(version.startsWith("2")){
	    if(localName.equalsIgnoreCase("PF"))
	    {
		nameStack.pop();
	    }
	}
    }
}
class HPCMetric{
	/*
	<!ELEMENT Metric (MetricFormula*, Info?)>
	<!ATTLIST Metric
	i            CDATA #REQUIRED
	n            CDATA #REQUIRED
	v            (raw|final|derived-incr|derived) "raw"
	t            (inclusive|exclusive|nil) "nil"
	partner      CDATA #IMPLIED
	fmt          CDATA #IMPLIED
	show         (1|0) "1"
	show-percent (1|0) "1">*/
    String id,name,type,partnerID;
    HPCMetric partner;
    Metric metric;
    double value = -1;
    private Thread thread;
    public HPCMetric(String id, String name, String type, String partnerID) {
	super();
	this.id = id;
	this.name = name;
	this.type = type;
	this.partnerID = partnerID;
    }
    public Thread getThread() {
	return thread;
    }
    public void setThread(Thread thread) {
	this.thread = thread;
    }
    public void resetValue() {
	value = -1;
	if(partner != null)
	    partner.value = -1;
	
    }
    public void setFunctionValue(FunctionProfile fp) {
	if(type.equalsIgnoreCase("inclusive"))
	 fp.setInclusive(metric.getID(), value);
	else if(type.equalsIgnoreCase("exnclusive"))
	    fp.setExclusive(metric.getID(), value);
	else{
	    fp.setInclusive(metric.getID(), value);
	    fp.setExclusive(metric.getID(), value);
	}
	
    }
    public boolean hasPartner() {
	return partnerID != null;
    }
    public HPCMetric(Attributes attributes) {
	this.name = attributes.getValue("n");
	this.id = attributes.getValue("i");
	this.type = attributes.getValue("t");
	this.partnerID = attributes.getValue("partner");
    }
    public boolean hasValue(){
	return value != -1;
    }
    public double getValue() {
        return value;
    }
    public void setValue(double value) {
	if(value <0) value = 0;
        this.value = value;
    }
    public Metric getMetric() {
        return metric;
    }
    public void setMetric(Metric metric) {
        this.metric = metric;
    }
    public String getId() {
        return id;
    }
    public void setId(String id) {
        this.id = id;
    }
    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public String getType() {
        return type;
    }
    public void setType(String type) {
        this.type = type;
    }
    public String getPartnerID() {
        return partnerID;
    }
    public void setPartnerID(String partnerID) {
        this.partnerID = partnerID;
    }
    public HPCMetric getPartner() {
        return partner;
    }
    public void setPartner(HPCMetric partner) {
        this.partner = partner;
    }
    @Override
    public String toString() {
	return "HPCMetric [id=" + id + ", name=" + name + ", type=" + type + ", partnerID="
		+ partnerID + ", metric=" + metric + ", value=" + value
		+ "]";
    }
    
    
}