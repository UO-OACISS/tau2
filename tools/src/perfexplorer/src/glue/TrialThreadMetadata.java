/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.io.Reader;
import java.io.StringReader;
import java.util.Set;
import java.util.Map;
import java.util.HashMap;
import java.util.TreeSet;


import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.server.TauNamespaceContext;

/**
 * @author khuck
 *
 */
public class TrialThreadMetadata extends AbstractResult {

    protected Map<Integer, Map<String, String>> stringData =
	            new HashMap<Integer, Map<String, String>>();
    protected Set<String> fields = new TreeSet<String>();

	/**
	 * 
	 */
	public TrialThreadMetadata(Trial trial) {
		getMetadata(trial);
	}

	public TrialThreadMetadata() {
		super();
	}
	
	/**
	 * @param input
	 */
	public TrialThreadMetadata(PerformanceResult input) {
		super(input);
		if (input instanceof TrialResult) {
			TrialResult tr = (TrialResult)input;
			Trial trial = tr.getTrial();
			getMetadata(trial);
		}
	}
	
	private void getMetadata(Trial trial) {
		try {
			// build a factory
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			// ask the factory for the document builder
			DocumentBuilder builder = factory.newDocumentBuilder();
	
			Reader reader = new StringReader(trial.getField(Trial.XML_METADATA));
			InputSource source = new InputSource(reader);
			Document metadata = builder.parse(source);
	
			NodeList names = null;
			NodeList values = null;
	
			// build the xpath object to jump around in that document
			javax.xml.xpath.XPath xpath = javax.xml.xpath.XPathFactory.newInstance().newXPath();
			xpath.setNamespaceContext(new TauNamespaceContext());
	
			// get the profile attributes from the metadata
			
			NodeList profileAttributes = (NodeList) xpath.evaluate("/metadata/ProfileAttributes", metadata, javax.xml.xpath.XPathConstants.NODESET);
	
			for (int i = 0 ; i < profileAttributes.getLength() ; i++) {
//				System.out.println("Got profile attributes..." + i);
				NamedNodeMap attributes = profileAttributes.item(i).getAttributes();
				String node = attributes.getNamedItem("node").getNodeValue();
				String context = attributes.getNamedItem("context").getNodeValue();
				String thread = attributes.getNamedItem("thread").getNodeValue();
				// TODO - calcluate the thread id properly from the node, context, thread values
				NodeList children = profileAttributes.item(i).getChildNodes();
				// TODO : this is a hack for sweep3d support - REMOVE IT!
				int neighbors = 0;
				for (int j = 0 ; j < children.getLength(); j++) {
					Node attributeElement = children.item(j);
					Node name = attributeElement.getFirstChild();
					while (name.getFirstChild() == null || name.getFirstChild().getNodeValue() == null) {
						name = name.getNextSibling();
					}
					Node value = name.getNextSibling();
					while (value.getFirstChild() == null || value.getFirstChild().getNodeValue() == null) {
						value = value.getNextSibling();
					}
					if (value == null) { // if there is no value
					} else {
						String tmp = value.getFirstChild().getNodeValue();
						String tmpName = name.getFirstChild().getNodeValue();
						if (tmp != null && tmpName != null && !tmpName.equals("pid") && !tmpName.toLowerCase().contains("time")) {
							try {
								Double tmpDouble = Double.parseDouble(tmp);
								// The metric name is "metadata"
								this.putExclusive(Integer.parseInt(node), tmpName, "METADATA", tmpDouble.doubleValue());
//								System.out.println(tmpName + node + " " + tmp);
								// TODO : this is a hack for sweep3d support - REMOVE IT!
								if (tmpName.startsWith("processor neighbor") && tmpDouble > 0.0) {
									neighbors++;
								}
							} catch (NumberFormatException e) { 
								putNameValue(Integer.parseInt(node), tmpName, tmp);
							}
						}
					}
				}
				// TODO : this is a hack for sweep3d support - REMOVE IT!
				this.putExclusive(Integer.parseInt(node), "total Neighbors", "METADATA", neighbors);
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
			// no metadata?  ok, at least put the node, context, thread ids in the metadata
			int nodes = Integer.parseInt(trial.getField("node_count"));
			int contexts = Integer.parseInt(trial.getField("contexts_per_node"));
			int threads = Integer.parseInt(trial.getField("threads_per_context"));
			int index = 0;
			for (int n = 0 ; n < nodes ; n++) {
				for (int c = 0 ; c < contexts ; c++) {
					for (int t = 0 ; t < threads ; t++) {
						this.putExclusive(index, "node", "METADATA", n);
						this.putExclusive(index, "context", "METADATA", c);
						this.putExclusive(index, "thread", "METADATA", t);
						this.putExclusive(index, "MPI Rank", "METADATA", index);
						index++;
					}
				}
			}
		}

	}

    public void putNameValue(Integer thread, String field, String value) {
        if (!threads.contains(thread)) {
            threads.add(thread);
        }
        if (!fields.contains(field)) {
            fields.add(field);
        }
        if (!stringData.containsKey(thread)) {
            stringData.put(thread, new HashMap<String, String>());
        }
        stringData.get(thread).put(field, value);
    }

	public String getNameValue(Integer thread, String field) {
		String value = null;
		try {
			value = stringData.get(thread).get(field);
		} catch (NullPointerException e) {
			value = "";
		}
		return value;
	}

    public Set<String> getFields() {
		return fields;
	}

    public int getThreadCount() {
		return stringData.size();
	}
}
