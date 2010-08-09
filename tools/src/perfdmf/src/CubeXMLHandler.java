package edu.uoregon.tau.perfdmf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.StringTokenizer;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * XML Handler for cube data
 *
 * @see <a href="http://www.fz-juelich.de/zam/kojak/">
 * http://www.fz-juelich.de/zam/kojak/</a> for more information about cube
 * 
 * <P>CVS $Id: CubeXMLHandler.java,v 1.9 2009/10/27 00:13:17 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.9 $
 */
public class CubeXMLHandler extends DefaultHandler {

    private StringBuffer accumulator;

    private String metricID;
    private Stack<String> metricIDStack = new Stack<String>();

    private String regionID;
    private String csiteID;
    private String cnodeID;
    private int threadID = -1;
    private String callee;
    private String uom;

    private String rank;
    private Stack<String> rankStack = new Stack<String>();

    private Map<String, Metric> metricMap = new HashMap<String, Metric>(); // map cube metricId Strings to PerfDMF Metric classes
    private Map<String, String> regionMap = new HashMap<String, String>(); // map cube regionId Strings to Function names (Strings)
    private Map<String, String> csiteMap = new HashMap<String, String>(); // map cube csiteId Strings to regionId Strings 
    private Map<String, Function> cnodeMap = new HashMap<String, Function>(); // map cube cnodeId Strings to csiteId Strings
    private Map<Metric, String> uomMap = new HashMap<Metric, String>(); // map cube metricId Strings to uom (unit of measure) Strings

    private CubeDataSource cubeDataSource;

    private String name;
    private Stack<String> nameStack = new Stack<String>();

    private Stack<String> cnodeStack = new Stack<String>();

    private List<Thread> threads = new ArrayList<Thread>();
    private Metric metric;

    // changed to 3 if detected, some things are different
    private int version = 2;

    private Metric calls = new Metric();

    private List<CubeProcess> cubeProcesses = new ArrayList<CubeProcess>();
    private CubeProcess cubeProcess;

    private Group defaultGroup;
    private Group callpathGroup;

    // for progress only
    private volatile int numMetrics = 1;
    private volatile int currentMetric = 0;

    // these maps were added to speed up the lookup of flat and parent profiles
    private Map<Function, Function> parentMap = new HashMap<Function, Function>(); // map functions to their parent functions ("A=>B=>C" -> "A=>B")
    private Map<Function, Function> flatMap = new HashMap<Function, Function>(); // map functions to their flat functions ("A=>B=>C" -> "C")

    private static class CubeProcess {
        public int rank;
        public List<CubeThread> threads = new ArrayList<CubeThread>();

        public CubeProcess(int rank) {
            this.rank = rank;
        }
    }

    private static class CubeThread {
        public int rank;
        public int id;

        public CubeThread(int rank, int id) {
            this.rank = rank;
            this.id = id;
        }
    }

    public CubeXMLHandler(CubeDataSource cubeDataSource) {
        super();
        this.cubeDataSource = cubeDataSource;
    }

    public void startDocument() throws SAXException {
        accumulator = new StringBuffer();
        calls.setName("Number of Calls");

        defaultGroup = cubeDataSource.addGroup("CUBE_DEFAULT");
        callpathGroup = cubeDataSource.addGroup("CUBE_CALLPATH");
    }

    public void endDocument() throws SAXException {}

    private String getFunctionName(Object callSiteID) {
        if (version == 3) {
            return regionMap.get(callSiteID);
        } else {
            return regionMap.get(csiteMap.get(callSiteID));
        }
    }

    private String getInsensitiveValue(Attributes attributes, String key) {
        for (int i = 0; i < attributes.getLength(); i++) {
            if (attributes.getLocalName(i).equalsIgnoreCase(key)) {
                return attributes.getValue(i);
            }
        }
        return null;
    }

    /* (non-Javadoc)
     * @see org.xml.sax.ContentHandler#startElement(java.lang.String, java.lang.String, java.lang.String, org.xml.sax.Attributes)
     */
    @SuppressWarnings("unchecked")
	public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        accumulator.setLength(0);

        // after development is done here this should be changed to if/else if's at least

        if (localName.equalsIgnoreCase("cube")) {
            String version = (String) getInsensitiveValue(attributes, "version");
            //          if (!version.equals("2.0")) {
            //          throw new DataSourceException("PerfDMF only reads version 2.0 cube files (Found version " + version + ")");
            //      }
            if (version.equals("3.0")) {
                this.version = 3;
            }

        } else if (localName.equalsIgnoreCase("metric")) {
            metricIDStack.push(metricID);
            metricID = getInsensitiveValue(attributes, "id");
        } else if (localName.equalsIgnoreCase("thread")) {
            String tmp = getInsensitiveValue(attributes, "id");
            if (tmp != null) {
                threadID = Integer.parseInt(tmp);
            } else {
                threadID++;
            }
        } else if (localName.equalsIgnoreCase("region")) {
            regionID = getInsensitiveValue(attributes, "id");
        } else if (localName.equalsIgnoreCase("csite")) {
            csiteID = getInsensitiveValue(attributes, "id");
        } else if (localName.equalsIgnoreCase("cnode")) {
            cnodeID = getInsensitiveValue(attributes, "id");
            csiteID = getInsensitiveValue(attributes, "csiteid");
            callee = getInsensitiveValue(attributes, "calleeid");

            String functionName;

            if (version == 3) {
                functionName = getFunctionName(callee);
            } else {
                functionName = getFunctionName(csiteID);
            }

            Stack<String> stackCopy = (Stack<String>) cnodeStack.clone();

            while (stackCopy.size() != 0) {
                functionName = getFunctionName(stackCopy.pop()) + " => " + functionName;
            }

            Function function = cubeDataSource.addFunction(functionName);

            if (functionName.indexOf("=>") != -1) {
                function.addGroup(callpathGroup);
            } else {
                function.addGroup(defaultGroup);
            }

            cnodeMap.put(cnodeID, function);
            if (version == 3) {
                cnodeStack.push(callee);
            } else {
                cnodeStack.push(csiteID);
            }
        } else if (localName.equalsIgnoreCase("matrix")) {
            metricID = getInsensitiveValue(attributes, "metricid");
            metric = metricMap.get(metricID);
            currentMetric++;
        } else if (localName.equalsIgnoreCase("row")) {
            cnodeID = getInsensitiveValue(attributes, "cnodeid");
        } else if (localName.equalsIgnoreCase("process")) {
            cubeProcess = new CubeProcess(-1);
            cubeProcesses.add(cubeProcess);
        }

    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        accumulator.append(ch, start, length);
    }

    private void popName() {
        name = nameStack.pop();
    }

    /* (non-Javadoc)
     * @see org.xml.sax.ContentHandler#endElement(java.lang.String, java.lang.String, java.lang.String)
     */
    @SuppressWarnings("unchecked")
	public void endElement(String uri, String localName, String qName) throws SAXException {

        if (localName.equalsIgnoreCase("process")) {
            cubeProcess.rank = Integer.parseInt(rank);
            rank = rankStack.pop();
        } else if (localName.equalsIgnoreCase("thread")) {
            CubeThread cubeThread = new CubeThread(Integer.parseInt(rank), threadID);
            cubeProcess.threads.add(cubeThread);
            rank = rankStack.pop();
        } else if (localName.equalsIgnoreCase("locations") || localName.equalsIgnoreCase("system")) {
            for (int i = 0; i < cubeProcesses.size(); i++) {
                CubeProcess cubeProcess = cubeProcesses.get(i);
                Node node = cubeDataSource.addNode(cubeProcess.rank);
                Context context = node.addContext(0);
                for (int j = 0; j < cubeProcess.threads.size(); j++) {
                    CubeThread cubeThread = cubeProcess.threads.get(j);
                    Thread thread = context.addThread(cubeThread.rank, cubeDataSource.getNumberOfMetrics());

                    while (cubeThread.id >= threads.size()) {
                        threads.add(null);
                    }
                    threads.set(cubeThread.id, thread);
                }
            }
            numMetrics = cubeDataSource.getNumberOfMetrics();
        } else if (localName.equalsIgnoreCase("rank")) {
            rankStack.push(rank);
            rank = accumulator.toString();
        } else if (localName.equalsIgnoreCase("name")) {
            nameStack.push(name);
            name = accumulator.toString();
        } else if (localName.equalsIgnoreCase("disp_name")) {
            nameStack.push(name);
            name = accumulator.toString();
        } else if (localName.equalsIgnoreCase("uom")) {
            uom = accumulator.toString();
        } else if (localName.equalsIgnoreCase("metric")) {
            if (name.equalsIgnoreCase("Visits") || name.equalsIgnoreCase("Calls")) {
                metricMap.put(metricID, calls);

            } else {

                String treeName = name;
                Stack<String> stackCopy = (Stack<String>) nameStack.clone();
                while (stackCopy.size() > 0) {
                    String next = stackCopy.pop();
                    if (next != null) {
                        treeName = next + " => " + treeName;
                    }
                }

                String metricName = treeName.toUpperCase();
                if (metricName.indexOf("TIME") != -1 && uom.toUpperCase().equals("OCC")) {
                    treeName = treeName + "_count";
                }

                Metric metric = cubeDataSource.addMetric(treeName);
                metricMap.put(metricID, metric);

                // record the unit of measure for later use
                uomMap.put(metric, uom);
            }

            metricID = metricIDStack.pop();

            popName();
        } else if (localName.equalsIgnoreCase("region")) {
            regionMap.put(regionID, name);
            popName();
        } else if (localName.equalsIgnoreCase("callee")) {
            callee = accumulator.toString();
        } else if (localName.equalsIgnoreCase("csite")) {
            csiteMap.put(csiteID, callee);
        } else if (localName.equalsIgnoreCase("cnode")) {
            cnodeStack.pop();
        } else if (localName.equalsIgnoreCase("row")) {

            Function function = cnodeMap.get(cnodeID);

            String data = accumulator.toString();

            StringTokenizer tokenizer = new StringTokenizer(data, " \t\n\r");

            int index = 0;
            while (tokenizer.hasMoreTokens()) {
                String line = tokenizer.nextToken();

                Thread thread = threads.get(index);
                FunctionProfile fp = thread.getFunctionProfile(function);
                if (fp == null) {
                    fp = new FunctionProfile(function, cubeDataSource.getNumberOfMetrics());
                    thread.addFunctionProfile(fp);
                }

                double value = Double.parseDouble(line);
                if (value < 0) {
                    System.err.println("Warning: negative value found in cube file (" + value + ")");
                }

                if (metric == calls) {
                    fp.setNumCalls(value);

                } else {
                    String unitsOfMeasure = uomMap.get(metric);
                    if (unitsOfMeasure.equalsIgnoreCase("sec")) {
                        value = value * 1000 * 1000;
                    }
                    fp.setExclusive(metric.getID(), value);
                    //fp.setInclusive(metric.getID(), 0);
                }

                if (function.isCallPathFunction()) { // get the flat profile (C for A => B => C)

                    FunctionProfile flatFP = getFlatFunctionProfile(thread, function);

                    if (metric == calls) {
                        flatFP.setNumCalls(flatFP.getNumCalls() + value);
                    } else {
                        flatFP.setExclusive(metric.getID(), value + flatFP.getExclusive(metric.getID()));
                        //childFP.setInclusive(metric.getID(), value + childFP.getInclusive(metric.getID()));
                    }
                }

                if (metric == calls) {
                    // add numcalls to numsubr of parent (and its flat profile)
                    FunctionProfile parent = getParent(thread, function);
                    if (parent != null) {
                        parent.setNumSubr(parent.getNumSubr() + value);
                        FunctionProfile flat = getFlatFunctionProfile(thread, parent.getFunction());
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
    private FunctionProfile getFlatFunctionProfile(Thread thread, Function function) {
        if (!function.isCallPathFunction()) {
            return null;
        }

        Function childFunction = flatMap.get(function);

        if (childFunction == null) {
            String childName = function.getName().substring(function.getName().lastIndexOf("=>") + 2).trim();
            childFunction = cubeDataSource.addFunction(childName);
            childFunction.addGroup(defaultGroup);
            flatMap.put(function, childFunction);
        }
        FunctionProfile childFP = thread.getFunctionProfile(childFunction);
        if (childFP == null) {
            childFP = new FunctionProfile(childFunction, cubeDataSource.getNumberOfMetrics());
            thread.addFunctionProfile(childFP);
        }
        return childFP;

    }

    // retrieve the parent profile on a given thread (A=>B for A=>B=>C)
    private FunctionProfile getParent(Thread thread, Function function) {
        if (!function.isCallPathFunction()) {
            return null;
        }

        Function parentFunction = parentMap.get(function);
        if (parentFunction == null) {
            String functionName = function.getName();
            String parentName = functionName.substring(0, functionName.lastIndexOf("=>"));
            parentFunction = cubeDataSource.getFunction(parentName);
            parentMap.put(function, parentFunction);
        }
        FunctionProfile parent = thread.getFunctionProfile(parentFunction);
        return parent;
    }

    // recursively add a value to the inclusive amount (go up the tree)
    private void addToInclusive(Thread thread, FunctionProfile fp, double value) {
        // add to this fp
        fp.setInclusive(metric.getID(), value + fp.getInclusive(metric.getID()));

        // add to our flat
        FunctionProfile flatFP = getFlatFunctionProfile(thread, fp.getFunction());

        if (flatFP != null) {
            flatFP.setInclusive(metric.getID(), value + flatFP.getInclusive(metric.getID()));
        }

        // recurse to A => B if this is A => B => C
        FunctionProfile parent = getParent(thread, fp.getFunction());
        if (parent != null) {
            addToInclusive(thread, parent, value);
        }

    }

    public int getProgress() {
        int value = (int) (((float) currentMetric / (float) numMetrics) * 100);
        return value;
    }

}
