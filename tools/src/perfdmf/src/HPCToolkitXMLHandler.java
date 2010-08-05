package edu.uoregon.tau.perfdmf;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

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
    
    private Thread theThread;
    
    private Stack<String> nameStack = new Stack<String>();

    private Group defaultGroup;
    private Group callpathGroup;

    
    public HPCToolkitXMLHandler(HPCToolkitDataSource dataSource) {
        super();
        this.dataSource = dataSource;
    }

    public void startDocument() throws SAXException {
        theThread = dataSource.addThread(0, 0, 0);
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
        FunctionProfile flat = getFlatFunctionProfile(theThread, f);

    }
    
    
    // given A => B => C, this retrieves the FP for C
    private FunctionProfile getFlatFunctionProfile(Thread thread, Function function) {
        if (!function.isCallPathFunction()) {
            return null;
        }
        String childName = function.getName().substring(function.getName().lastIndexOf("=>") + 2).trim();
        Function childFunction = dataSource.addFunction(childName);

        childFunction.addGroup(defaultGroup);

        FunctionProfile childFP = thread.getFunctionProfile(childFunction);
        if (childFP == null) {
            childFP = new FunctionProfile(childFunction, dataSource.getNumberOfMetrics());
            thread.addFunctionProfile(childFP);
        }
        return childFP;

    }
    
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {

        if (localName.equalsIgnoreCase("METRIC")) {
            String displayName = attributes.getValue("displayName");
            String shortName = attributes.getValue("shortName");

            String nativeName = attributes.getValue("nativeName");

            // match PAPI_FP_INS-0, PAPI_FP_INS-1
            //RE r = new RE("*-[0-9]+");
            
            //if (r.match(nativeName)) {
                
            //}
            
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

    }

    public void endElement(String uri, String localName, String qName) throws SAXException {
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
    }

}
