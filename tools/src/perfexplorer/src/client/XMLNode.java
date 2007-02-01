/**
 * 
 */
package client;

import java.io.File;
import java.util.List;

import javax.swing.tree.DefaultMutableTreeNode;
import org.xml.sax.Attributes;

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
		this.value = value;
	}
	
	public List getChildren() {
		return children;
	}
}

class XMLElementNode extends XMLNode {
	private String namespaceURI = null;
	private String localName = null;
	private String qName = null;
	private Attributes attributes = null;
	private String prefix = null;
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
//            String attURI = attributes.getURI(i);
//            if (attURI.length() > 0) {
//                String attPrefix = 
//                    (String)namespaceMappings.get(namespaceURI);
//                DefaultMutableTreeNode attNamespace =
//                    new DefaultMutableTreeNode("Namespace: prefix = '" +
//                        attPrefix + "', URI = '" + attURI + "'");
//                attribute.add(attNamespace);            
//            }
            this.add(attribute);
        }
	}
	
	public String toString() {
		return "<" + this.prefix + ":" + this.localName + ">";
	}
}

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
		return "@" + this.localName;
	}
	
}

class XMLCommentNode extends XMLNode {
	public XMLCommentNode (String value) {
		this.value = value;
	}
	public String toString() {
		return "#COMMENT";
	}
}
