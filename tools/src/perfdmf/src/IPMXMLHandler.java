package edu.uoregon.tau.perfdmf;

import java.util.Hashtable;
import java.util.Map;
import java.util.Set;
import java.util.Iterator;
import java.util.TreeMap;
import java.util.List;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/*** SAX Handler which creates SQL to load a xml document into the database. ***/

public class IPMXMLHandler extends DefaultHandler {
    private StringBuffer accumulator = new StringBuffer();

    protected String currentElement = "";
	private String functionCount = "";
	private boolean firstTask = false;
    private IPMDataSource dataSource;
	private Function function = null;
	private FunctionProfile fp = null;
	private RegionData region = null;

    public IPMXMLHandler(IPMDataSource dataSource) {
        super();
        this.dataSource = dataSource;
    }

	/** Parse a value from a case-insensitive label */
    private String getInsensitiveValue(Attributes attributes, String key) {
        for (int i = 0; i < attributes.getLength(); i++) {
            if (attributes.getLocalName(i).equalsIgnoreCase(key)) {
                return attributes.getValue(i);
            }
        }
        return null;
    }

	/** For each attribute in the tag, add it to the metadata. */
	private void processAttributes(Attributes attrList) {
       	for (int i = 0; i < attrList.getLength(); i++) {
           	String key = attrList.getLocalName(i);
            String value = attrList.getValue(i);
			dataSource.getThread().getMetaData().put(currentElement + ":" + key, value);
		}
	}

    public void startElement(String url, String name, String qname, Attributes attrList) throws SAXException {
        accumulator.setLength(0);

        if (name.equalsIgnoreCase("ipm_job_profile")) {
			// do nothing
        } else if (name.equalsIgnoreCase("task")) {
            currentElement = name;
			// beginning of an MPI task 
			dataSource.initializeThread(firstTask);
			firstTask = true;
			// save all the metadata
			processAttributes(attrList);
        } else if (name.equalsIgnoreCase("job")) {
            currentElement = name;
			// save all the metadata
			processAttributes(attrList);
        } else if (name.equalsIgnoreCase("host")) {
            currentElement = name;
			// save all the metadata
			processAttributes(attrList);
        } else if (name.equalsIgnoreCase("perf")) {
            currentElement = name;
			// save all the metadata
			processAttributes(attrList);
        } else if (name.equalsIgnoreCase("switch")) {
            currentElement = name;
			// save all the metadata
			processAttributes(attrList);
        } else if (name.equalsIgnoreCase("cmdline")) {
            currentElement = name;
			// save all the metadata
			processAttributes(attrList);
        } else if ((name.equalsIgnoreCase("exec")) || (name.equalsIgnoreCase("exec_bin"))) {
            currentElement = name;
        } else if (name.equalsIgnoreCase("pre")) {
			// do nothing - handled at close
        } else if (name.equalsIgnoreCase("internal")) {
            currentElement = name;
			// save all the metadata
			processAttributes(attrList);
        } else if (name.equalsIgnoreCase("region")) {
			// here is a code region.
            currentElement = name;
			// get the region name
			String regionLabel = getInsensitiveValue(attrList, "label");
			this.region = new RegionData(regionLabel);
			// for each metric, save the value
			String[] values = {"nexits", "wtime", "utime", "stime", "mtime"};
			for (int i = 0 ; i < values.length ; i++) {
				String tmp = getInsensitiveValue(attrList, values[i]);
				this.region.measurements.put(values[i], tmp);
			}
        } else if (name.equalsIgnoreCase("counter")) {
			// save the HW counter name
			currentElement = getInsensitiveValue(attrList, "name");
        } else if (name.equalsIgnoreCase("func")) {
			// save the function name and number of calls
			currentElement = getInsensitiveValue(attrList, "name");
			functionCount = getInsensitiveValue(attrList, "count");
        }
    }

    public void characters(char[] ch, int start, int length) throws SAXException {
        accumulator.append(ch, start, length);
    }

    public void endElement(String url, String name, String qname) {
        if (name.equalsIgnoreCase("task")) {
			// do nothing - end of the MPI task data.
        } else if (name.equalsIgnoreCase("job")) {
			// save the value as metadata
            String value = accumulator.toString().trim();
			dataSource.getThread().getMetaData().put(name, value);
        } else if (name.equalsIgnoreCase("host")) {
			// save the value as metadata
            String value = accumulator.toString().trim();
			dataSource.getThread().getMetaData().put(name, value);
        } else if (name.equalsIgnoreCase("perf")) {
			// nothing?
        } else if (name.equalsIgnoreCase("switch")) {
			// nothing?
        } else if (name.equalsIgnoreCase("cmdline")) {
			// save the value as metadata
            String value = accumulator.toString().trim();
			dataSource.getThread().getMetaData().put(name, value);
        } else if (name.equalsIgnoreCase("pre")) {
			// save the value as metadata - include the exec/exec_bin label
            String value = accumulator.toString().trim();
			dataSource.getThread().getMetaData().put(currentElement + ":" + name, value);
        } else if (name.equalsIgnoreCase("env")) {
			// save the value as metadata
            String value = accumulator.toString().trim();
			StringTokenizer st = new StringTokenizer(value, "=");
			if (st.countTokens() == 2) {
				String key = st.nextToken();
				value = st.nextToken();
				dataSource.getThread().getMetaData().put("env:" + key, value);
			}
        } else if ((name.equalsIgnoreCase("ru_s_ti")) || (name.equalsIgnoreCase("ru_s_tf")) ||
                   (name.equalsIgnoreCase("ru_c_ti")) || (name.equalsIgnoreCase("ru_c_tf"))) {
			// save the value as metadata
            String value = accumulator.toString().trim();
			dataSource.getThread().getMetaData().put(name, value);
        } else if (name.equalsIgnoreCase("counter")) {
			// save the HW counter value
			String tmp = accumulator.toString().trim();
			this.region.measurements.put(currentElement, tmp);
        } else if (name.equalsIgnoreCase("func")) {
			// get the value
			String tmp = accumulator.toString().trim();
			// get the function name
			RegionData data = new RegionData(currentElement);
			// get the num calls
			data.numCalls = Integer.parseInt(functionCount);
			// store the time
			data.measurements.put("wtime",tmp);
			// add to the region
			this.region.functions.add(data);
        } else if (name.equalsIgnoreCase("region")) {
			// create the function
			createFunction(dataSource.getThread(), this.region, "");
    	}

    }

	/** THe order that these are done is very important. */
	private void createFunction(Thread thread, RegionData data, String callpath) {
		this.function = dataSource.addFunction(callpath + data.name);
		if (callpath.length() > 0) {
			this.function.addGroup(dataSource.addGroup("TAU_CALLPATH"));
		} else {
			this.function.addGroup(dataSource.addGroup("TAU_DEFAULT"));
		}
		if (data.name.startsWith("MPI_")) {
			this.function.addGroup(dataSource.addGroup("MPI"));
		}
		// always get it from the current parent region
		Set keys = this.region.measurements.keySet();
		this.fp = new FunctionProfile (function, keys.size());
		thread.addFunctionProfile(this.fp);
		fp.setNumCalls(data.numCalls);
		fp.setNumSubr(data.functions.size());
		for (Iterator iter = keys.iterator() ; iter.hasNext() ; ) {
			String metric = (String) iter.next();
			Metric m = dataSource.addMetric(metric, thread);
			String value = (String)data.measurements.get(metric);
			if (value != null) {
				double d = Double.parseDouble((String)data.measurements.get(metric));
				fp.setInclusive(m.getID(), d);
				fp.setExclusive(m.getID(), d);
			}
		}
		for (Iterator iter = data.functions.iterator() ; iter.hasNext() ; ) {
			RegionData func = (RegionData)iter.next();
			createFunction(dataSource.getThread(), func, "");
			createFunction(dataSource.getThread(), func, data.name + " => " );
		}

	}

	private class RegionData {
		public String name;

		// the metric name / value map
		public TreeMap measurements = new TreeMap();

		public int numCalls = 1;

		// the list of functions in this region
		public List functions = new ArrayList();

		public RegionData (String name) {
			this.name = name;
		}
	}
}
