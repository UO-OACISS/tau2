package edu.uoregon.tau.dms.dss;

import java.util.*;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * XML Handler for cube data
 *
 * @see <a href="http://www.fz-juelich.de/zam/kojak/">
 * http://www.fz-juelich.de/zam/kojak/</a> for more information about cube
 * 
 * <P>CVS $Id: CubeXMLHandler.java,v 1.1 2005/06/07 01:25:32 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class CubeXMLHandler extends DefaultHandler {


    private StringBuffer accumulator;

    private String metricID;
    private Stack metricIDStack = new Stack();

    private String regionID;
    private String csiteID;
    private String cnodeID;
    private String callee;

    private String uom;

    private String rank;
    private Stack rankStack = new Stack();
    
    private Map metricMap = new HashMap(); // map cube metricId Strings to PerfDMF Metric classes
    private Map regionMap = new HashMap(); // map cube regionId Strings to Function names (Strings)
    private Map csiteMap = new HashMap();  // map cube csiteId Strings to regionId Strings 
    private Map cnodeMap = new HashMap();  // map cube cnodeId Strings to csiteId Strings

    private CubeDataSource cubeDataSource;

    private String name;
    private Stack nameStack = new Stack();

    private Stack cnodeStack = new Stack();

    private List threads = new ArrayList();
    private Metric metric;

    private int nodeID = 0;
    private int threadID = 0;

    private Metric calls = new Metric();

    private List cubeProcesses = new ArrayList();
    private CubeProcess cubeProcess;
    
    private static class CubeProcess {
        public int rank;
        public List threads = new ArrayList();
    
        public CubeProcess(int rank) {
            this.rank = rank;
        }
    }
    
    private static class CubeThread {
        public int rank;
        public CubeThread(int rank) {
            this.rank = rank;
        }
    }
    
    
    public CubeXMLHandler(CubeDataSource cubeDataSource) {
        super();
        this.cubeDataSource = cubeDataSource;
    }

    public void startDocument() throws SAXException {
        accumulator = new StringBuffer();
        calls.setName("Number of Calls");
    }

    public void endDocument() throws SAXException {
    }

    private String getFunctionName(Object callSiteID) {
        return (String) regionMap.get(csiteMap.get(callSiteID));
    }

    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        accumulator.setLength(0);

        // after development is done here this should be changed to if/else if's at least
        
        if (localName.equalsIgnoreCase("metric")) {
            metricIDStack.push(metricID);
            metricID = attributes.getValue("id");
        }

        if (localName.equalsIgnoreCase("region")) {
            regionID = attributes.getValue("id");
        }

        if (localName.equalsIgnoreCase("csite")) {
            csiteID = attributes.getValue("id");
        }

        if (localName.equalsIgnoreCase("cnode")) {
            cnodeID = attributes.getValue("id");
            csiteID = attributes.getValue("csiteId");

            String functionName = getFunctionName(csiteID);

            Stack stackCopy = (Stack) cnodeStack.clone();

            while (stackCopy.size() != 0) {
                functionName = getFunctionName(stackCopy.pop()) + " => " + functionName;
            }

            System.out.println("Adding: " + functionName);

            Function function = cubeDataSource.addFunction(functionName);
            cnodeMap.put(cnodeID, function);

            cnodeStack.push(csiteID);
        }

        if (localName.equalsIgnoreCase("matrix")) {
            metricID = attributes.getValue("metricId");
            metric = (Metric) metricMap.get(metricID);
        }

        if (localName.equalsIgnoreCase("row")) {
            cnodeID = attributes.getValue("cnodeId");
        }

        if (localName.equalsIgnoreCase("process")) {
            cubeProcess = new CubeProcess(-1);
            cubeProcesses.add(cubeProcess);
        }

    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        accumulator.append(ch, start, length);
    }

    private void popName() {
        name = (String) nameStack.pop();
    }

    public void endElement(String uri, String localName, String qName) throws SAXException {

        // after development is done here this should be changed to if/else if's at least

        if (localName.equalsIgnoreCase("process")) {
            cubeProcess.rank = Integer.parseInt(rank);
            rank = (String) rankStack.pop();
        }

        if (localName.equalsIgnoreCase("thread")) {
            CubeThread cubeThread = new CubeThread(Integer.parseInt(rank));
            cubeProcess.threads.add(cubeThread);
            rank = (String) rankStack.pop();
        }

        if (localName.equalsIgnoreCase("locations")) {
            for (int i = 0; i < cubeProcesses.size(); i++) {
                CubeProcess cubeProcess = (CubeProcess) cubeProcesses.get(i);
                Node node = cubeDataSource.addNode(cubeProcess.rank);
                Context context = node.addContext(0);
                for (int j = 0; j < cubeProcess.threads.size(); j++) {
                    CubeThread cubeThread = (CubeThread) cubeProcess.threads.get(j);
                    Thread thread = context.addThread(cubeThread.rank, cubeDataSource.getNumberOfMetrics());
                    threads.add(thread);
                }
            }
        }
        
        if (localName.equalsIgnoreCase("rank")) {
            rankStack.push(rank);
            rank = accumulator.toString();
        }
        
        if (localName.equalsIgnoreCase("name")) {
            nameStack.push(name);
            name = accumulator.toString();
        }

        if (localName.equalsIgnoreCase("uom")) {
            uom = accumulator.toString();
        }

        if (localName.equalsIgnoreCase("metric")) {
            if (uom.equalsIgnoreCase("occ")) {
                metricMap.put(metricID, calls);

            } else {
                System.out.println("Got Metric: " + name);
                Metric metric = cubeDataSource.addMetric(name);
                metricMap.put(metricID, metric);
            }
            metricID = (String) metricIDStack.pop();

            popName();
        }

        if (localName.equalsIgnoreCase("region")) {
            regionMap.put(regionID, name);
            popName();
        }

        if (localName.equalsIgnoreCase("callee")) {
            callee = accumulator.toString();
        }

        if (localName.equalsIgnoreCase("csite")) {
            csiteMap.put(csiteID, callee);
        }

        if (localName.equalsIgnoreCase("cnode")) {
            cnodeStack.pop();
        }

        if (localName.equalsIgnoreCase("row")) {

            Function function = (Function) cnodeMap.get(cnodeID);

            String data = accumulator.toString();

            StringTokenizer tokenizer = new StringTokenizer(data, " \t\n\r");

            int index = 0;
            while (tokenizer.hasMoreTokens()) {
                String line = tokenizer.nextToken();

                Thread thread = (Thread) threads.get(index);
                FunctionProfile fp = thread.getFunctionProfile(function);
                if (fp == null) {
                    fp = new FunctionProfile(function, cubeDataSource.getNumberOfMetrics());
                    thread.addFunctionProfile(fp);
                }

                double value = Double.parseDouble(line);

                if (metric == calls) {
                    fp.setNumCalls(value);

                } else {
                    value = value * 1000 * 1000;
                    fp.setExclusive(metric.getID(), value);
                    fp.setInclusive(metric.getID(), 0);
                }

                String name = function.getName();
                if (name.lastIndexOf("=>") != -1) {

                    FunctionProfile flatFP = getTailFunctionProfile(thread, function.getName());

                    if (metric == calls) {
                        flatFP.setNumCalls(flatFP.getNumCalls() + value);
                    } else {
                        flatFP.setExclusive(metric.getID(), value + flatFP.getExclusive(metric.getID()));
                        //childFP.setInclusive(metric.getID(), value + childFP.getInclusive(metric.getID()));
                    }
                }

                if (metric == calls) {
                    FunctionProfile parent = getParent(thread, function.getName());
                    if (parent != null) {
                        parent.setNumSubr(parent.getNumSubr() + value);
                        FunctionProfile flat = getTailFunctionProfile(thread, parent.getName());
                        if (flat != null) {
                            flat.setNumSubr(flat.getNumSubr() + value);
                        }
                    }
                    //addToNumSubr(thread, fp, value);
                } else {
                    addToInclusive(thread, fp, value);
                }
                index++;
            }

        }

    }

    // given A => B => C, this retrieves the FP for C
    private FunctionProfile getTailFunctionProfile(Thread thread, String functionName) {
        if (functionName.lastIndexOf("=>") == -1) {
            return null;
        } else {
            String childName = functionName.substring(functionName.lastIndexOf("=>") + 2).trim();
            Function childFunction = cubeDataSource.getFunction(childName);
            if (childFunction == null) {
                childFunction = cubeDataSource.addFunction(childName);
            }
            FunctionProfile childFP = thread.getFunctionProfile(childFunction);
            if (childFP == null) {
                childFP = new FunctionProfile(childFunction, cubeDataSource.getNumberOfMetrics());
                thread.addFunctionProfile(childFP);
            }
            return childFP;
        }
    }

    private FunctionProfile getParent(Thread thread, String functionName) {
        if (functionName.lastIndexOf("=>") != -1) {
            String parentName = functionName.substring(0, functionName.lastIndexOf("=>"));
            FunctionProfile parent = thread.getFunctionProfile(cubeDataSource.getFunction(parentName));
            return parent;
        }
        return null;
    }

    private void addToInclusive(Thread thread, FunctionProfile fp, double value) {
        // add to this fp
        fp.setInclusive(metric.getID(), value + fp.getInclusive(metric.getID()));

        // add to our flat
        FunctionProfile flatFP = getTailFunctionProfile(thread, fp.getName());

        if (flatFP != null) {
            flatFP.setInclusive(metric.getID(), value + flatFP.getInclusive(metric.getID()));
        }

        // recurse to A => B if this is A => B => C
        FunctionProfile parent = getParent(thread, fp.getName());
        if (parent != null) {
            addToInclusive(thread, parent, value);
        }

    }

}
