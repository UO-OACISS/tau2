/**
 * 
 */
package glue;

import java.io.Reader;
import java.io.StringReader;
import java.sql.PreparedStatement;
import java.util.Hashtable;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import common.PerfExplorerOutput;

import edu.uoregon.tau.perfdmf.Trial;
import server.PerfExplorerServer;
import server.TauNamespaceContext;

/**
 * @author khuck
 *
 */
public class TrialMetadata {
	private Hashtable<String,String> commonAttributes = new Hashtable<String,String>();
	private Hashtable<String,Double> accumulator = new Hashtable<String,Double>();
	private Trial trial = null;

	public TrialMetadata (int id) {
		this.trial = PerfExplorerServer.getServer().getSession().setTrial(id);
		getMetadata();
	}
	
	public TrialMetadata (Trial trial) {
		this.trial = trial;
		getMetadata();
	}
	
	private void getMetadata() {
		try {
			// do the trial attributes
			commonAttributes.put("Trial.Name", trial.getName());
			String[] columns = Trial.getFieldNames(PerfExplorerServer.getServer().getDB());
			for (int index = 0 ; index < columns.length ; index++) {
				if (columns[index].equalsIgnoreCase("XML_METADATA") || columns[index].equalsIgnoreCase("XML_METADATA_GZ")) {
					continue;
				}
				if (trial.getField(index) == null) {
					commonAttributes.put(columns[index].toLowerCase(), new String(""));
				} else {
					commonAttributes.put(columns[index].toLowerCase(), trial.getField(index));
				}
			}
			
			// build a factory
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			// ask the factory for the document builder
			DocumentBuilder builder = factory.newDocumentBuilder();
	
			Reader reader = new StringReader(trial.getField(Trial.XML_METADATA));
			InputSource source = new InputSource(reader);
			Document metadata = builder.parse(source);
	
			NodeList names = null;
			NodeList values = null;
	
	//		try {
	//		/* this is the 1.3 through 1.4 way */
	//		names = org.apache.xpath.XPathAPI.selectNodeList(metadata, 
	//			"/metadata/CommonProfileAttributes/attribute/name");
	//		values = org.apache.xpath.XPathAPI.selectNodeList(metadata, 
	//			"/metadata/CommonProfileAttributes/attribute/value");
	//		} catch (NoClassDefFoundError e) {
	
			/* this is the 1.5 way */
			// build the xpath object to jump around in that document
			javax.xml.xpath.XPath xpath = javax.xml.xpath.XPathFactory.newInstance().newXPath();
			xpath.setNamespaceContext(new TauNamespaceContext());
	
			// get the common profile attributes from the metadata
			names = (NodeList) 
				xpath.evaluate("/metadata/CommonProfileAttributes/attribute/name", 
				metadata, javax.xml.xpath.XPathConstants.NODESET);
	
			values = (NodeList) 
				xpath.evaluate("/metadata/CommonProfileAttributes/attribute/value", 
				metadata, javax.xml.xpath.XPathConstants.NODESET);
	//		}
	
			for (int i = 0 ; i < names.getLength() ; i++) {
				Node name = (Node)names.item(i).getFirstChild();
				Node value = (Node)values.item(i).getFirstChild();
				if (value == null) { // if there is no value
					commonAttributes.put(name.getNodeValue(), "");
				} else {
					commonAttributes.put(name.getNodeValue(), value.getNodeValue());
				}
			}
			
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
								Double total = accumulator.get(tmpName);
								if (total == null)
									accumulator.put(tmpName, tmpDouble);
								else 
									accumulator.put(tmpName, total + tmpDouble);
							} catch (NumberFormatException e) { 
								commonAttributes.put(tmpName, tmp);
							}
						}
					}
				}
				for (String key : accumulator.keySet()) {
					commonAttributes.put(key, Double.toString(accumulator.get(key) / profileAttributes.getLength()));
				}
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}

	/**
	 * @return the commonAttributes
	 */
	public Hashtable<String, String> getCommonAttributes() {
		return commonAttributes;
	}

	/**
	 * @param commonAttributes the commonAttributes to set
	 */
	public void setCommonAttributes(Hashtable<String, String> commonAttributes) {
		this.commonAttributes = commonAttributes;
	}

	/**
	 * @return the trial
	 */
	public Trial getTrial() {
		return trial;
	}

	/**
	 * @param trial the trial to set
	 */
	public void setTrial(Trial trial) {
		this.trial = trial;
	}
	
	
}
