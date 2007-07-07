/**
 * 
 */
package client;

import java.io.File;
import java.util.List;

import javax.swing.tree.DefaultMutableTreeNode;
import org.xml.sax.Attributes;

/**
 * This is the abstract tree node type for the XML tree table.
 *
 */
class XMLNode extends DefaultMutableTreeNode {
	protected String value = null;
	
	public XMLNode (String name){
		super(name);
	}
	
	public XMLNode() {
		super();
	}
	
	public String getValue() {
		return this.value;
	}
	
	public void setValue(String value) {
		//something funny is happening in the SAX parser... so append the
		// text onto the existing text if it is already there.
		if (this.value == null) {
			this.value = value;
		} else {
			this.value = this.value + value;
		}
	}
	
	public List getChildren() {
		return children;
	}
}

/**
 * This is the default tree node type for the XML tree table.
 *
 */
class XMLElementNode extends XMLNode {
	protected String namespaceURI = null;
	protected String localName = null;
	protected String qName = null;
	protected Attributes attributes = null;
	protected String prefix = null;
	public XMLElementNode (String namespaceURI, String localName,
                           String qName, Attributes attributes, String prefix) {
		super();
		this.namespaceURI = namespaceURI;
		this.localName = localName;
		this.qName = qName;
		this.attributes = attributes;
    	this.prefix = prefix;
    	this.allowsChildren = true;
    	
        // Process attributes
        for (int i=0; i<attributes.getLength(); i++) {
            XMLAttributeNode attribute =
              new XMLAttributeNode(attributes.getLocalName(i),
            		  attributes.getQName(i),
            		  attributes.getType(i),
            		  attributes.getValue(i),
            		  attributes.getURI(i));
            this.add(attribute);
        }
	}

	protected XMLElementNode() {
		super();
	}
	
	public String toString() {
		return "<" + this.prefix + ":" + this.localName + ">";
	}
}

/**
 * This is a special kind of node for displaying node-specific
 * profile attributes.  This node type is different because it
 * collapses the node, context and thread id values into one
 * line to save space.
 *
 */
class XMLProfileNode extends XMLElementNode {
	private String nodeID = null;
	private String contextID = null;
	private String threadID = null;
	public XMLProfileNode (String namespaceURI, String localName,
                           String qName, Attributes attributes, String prefix) {
		super();
		this.namespaceURI = namespaceURI;
		this.localName = localName;
		this.qName = qName;
		this.attributes = attributes;
    	this.prefix = prefix;
    	this.allowsChildren = true;
    	
		if (attributes.getLength() == 3) {
        	for (int i=0; i<attributes.getLength(); i++) {
				if (attributes.getLocalName(i).equals("node")) {
					nodeID = attributes.getValue(i);
				} else if (attributes.getLocalName(i).equals("context")) {
					contextID = attributes.getValue(i);
				} else if (attributes.getLocalName(i).equals("thread")) {
					threadID = attributes.getValue(i);
				} 
			}
			StringBuffer buf = new StringBuffer();
			buf.append("NCT: ");
			buf.append(nodeID);
			buf.append(", ");
			buf.append(contextID);
			buf.append(", ");
			buf.append(threadID);
			this.value = buf.toString();
			/*
            XMLAttributeNode attribute =
            	new XMLAttributeNode("node, context, thread",
            	  	attributes.getQName(0),
            	  	attributes.getType(0),
            	  	buf.toString(),
            	  	attributes.getURI(0));
            this.add(attribute);
			*/
		} else {
        	// Process attributes
        	for (int i=0; i<attributes.getLength(); i++) {
            	XMLAttributeNode attribute =
              	new XMLAttributeNode(attributes.getLocalName(i),
            		  	attributes.getQName(i),
            		  	attributes.getType(i),
            		  	attributes.getValue(i),
            		  	attributes.getURI(i));
            	this.add(attribute);
        	}
		}
	}
	
	public String toString() {
		return "<" + this.prefix + ":" + this.localName + ">";
	}
}

/**
 * This is a special kind of node for displaying tau
 * attributes in a compact way.  This node type is different 
 * because it collapses the name and value nodes below this
 * node, identifies itself as a leaf, and shows the values from
 * the name and value child nodes as this node's name and value.
 * This is all done to save space.
 *
 */
class XMLTAUAttributeElementNode extends XMLElementNode {
	public XMLTAUAttributeElementNode (String namespaceURI, String localName,
                           String qName, Attributes attributes, String prefix) {
		super(namespaceURI, localName, qName, attributes, prefix);
	}
	
	public String toString() {
		XMLNode nameNode = (XMLNode)this.getFirstChild();
		return "@ " + nameNode.value;
	}

	public String getValue() {
		XMLNode valueNode = (XMLNode)this.getLastChild();
		return valueNode.getValue();
	}

	public boolean isLeaf() {
		return true;
	}
}

/**
 * This is a special kind of node for displaying an XML attribute.
 * The TreePortionCellRenderer will output this node's text as green.
 * 
 */
class XMLAttributeNode extends XMLNode {
	private String localName = null;
	private String qName = null;
	private String type = null;
	private String uri = null;
	
	public XMLAttributeNode (String localName, String qName, String type, String value, String uri) {
		super();
		this.localName = localName;
		this.qName = qName;
		this.type = type;
		this.value = value;
		this.uri = uri;
		this.allowsChildren = false;
	}
	
	public XMLAttributeNode (String localName, String value) {
		super();
		this.localName = localName;
		this.value = value;
		this.allowsChildren = false;
	}
	
	public String toString() {
		return "@ " + this.localName;
	}
	
}

/**
 * This is a special kind of node for displaying an XML comment.
 * The TreePortionCellRenderer will output this node's text as dark red.
 * 
 */
class XMLCommentNode extends XMLNode {
	public XMLCommentNode (String value) {
		this.value = value;
	}
	public String toString() {
		return "#COMMENT ";
	}
}
