/*-- 

 Copyright (C) 2001 Brett McLaughlin.
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions, and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions, and the disclaimer that follows 
    these conditions in the documentation and/or other materials 
    provided with the distribution.

 3. The name "Java and XML" must not be used to endorse or promote products
    derived from this software without prior written permission.  For
    written permission, please contact brett@newInstance.com.
 
 In addition, we request (but do not require) that you include in the 
 end-user documentation provided with the redistribution and/or in the 
 software itself an acknowledgement equivalent to the following:
     "This product includes software developed for the
      'Java and XML' book, by Brett McLaughlin (O'Reilly & Associates)."

 THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED.  IN NO EVENT SHALL THE JDOM AUTHORS OR THE PROJECT
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 SUCH DAMAGE.

 */

package edu.uoregon.tau.perfexplorer.client;

import java.awt.BorderLayout;
import java.io.IOException;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.tree.DefaultMutableTreeNode;

import org.xml.sax.Attributes;
import org.xml.sax.ContentHandler;
import org.xml.sax.ErrorHandler;
import org.xml.sax.InputSource;
import org.xml.sax.Locator;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;
import org.xml.sax.XMLReader;
import org.xml.sax.ext.LexicalHandler;
import org.xml.sax.helpers.XMLReaderFactory;

import edu.uoregon.tau.common.treetable.AbstractTreeTableModel;
import edu.uoregon.tau.common.treetable.JTreeTable;

/**
 * <b><code>SAXTreeViewer</code></b> uses Swing to graphically
 *   display an XML document.
 */
public class SAXTreeViewer extends JFrame {

    /**
	 * 
	 */
	private static final long serialVersionUID = -176395314854266217L;

	/** Default parser to use */
    private String vendorParserClass = 
        "org.apache.xerces.parsers.SAXParser";

    /** The base tree to render */
    private JTreeTable jTreeTable;

    /** Tree model to use */
    AbstractTreeTableModel xmlModel;

    /**
     * <p> This initializes the needed Swing settings. </p>
     */
    public SAXTreeViewer() {
        // Handle Swing setup
        super("SAX Tree Viewer");
        setSize(600, 450);
    }

    /**
     * <p> This will construct the tree using Swing. </p>
     *
     * @param filename <code>String</code> path to XML document.
     */
    public void init(String xml) throws IOException, SAXException {
        //DefaultMutableTreeNode base = 
        	new DefaultMutableTreeNode("XML String");
        
        // Construct the tree hierarchy
        buildTree(xml);

        // Display the results
        getContentPane().add(new JScrollPane(jTreeTable), 
            BorderLayout.CENTER);
    }

    public JTreeTable getTreeTable(String xml) throws IOException, SAXException {
        //DefaultMutableTreeNode base = null; 
            // new DefaultMutableTreeNode("XML Metadata");
        
        // Construct the tree hierarchy
        jTreeTable = buildTree(xml);
     
        jTreeTable.expandAll(true);

        return jTreeTable;
    }

    /**
     * <p>This handles building the Swing UI tree.</p>
     *
     * @param treeModel Swing component to build upon.
     * @param base tree node to build on.
     * @param xmlURI URI to build XML document from.
     * @throws <code>IOException</code> - when reading the XML URI fails.
     * @throws <code>SAXException</code> - when errors in parsing occur.
     */
    public JTreeTable buildTree(String xml) 
        throws IOException, SAXException {

        // Create instances needed for parsing
    	
    	XMLReader reader; //= XMLReaderFactory.createXMLReader(vendorParserClass);
    	
    	 try { // Xerces
    		 reader = XMLReaderFactory.createXMLReader(
    				 vendorParserClass
    		    );
    		  }
    		  catch (SAXException e1) {
    		    try { // Crimson
    		    	 reader = XMLReaderFactory.createXMLReader(
    		       "org.apache.crimson.parser.XMLReaderImpl"
    		      );
    		    }
    		    catch (SAXException e2) { 
    		      try { // Ælfred
    		    	  reader= XMLReaderFactory.createXMLReader(
    		         "gnu.xml.aelfred2.XmlReader"
    		        );
    		      }
    		      catch (SAXException e3) {
    		        try { // Piccolo
    		          reader = XMLReaderFactory.createXMLReader(
    		            "com.bluecast.xml.Piccolo"
    		          );
    		        }
    		        catch (SAXException e4) {
    		          try { // Oracle
    		        	  reader = XMLReaderFactory.createXMLReader(
    		              "oracle.xml.parser.v2.SAXParser"
    		            );
    		          }
    		          catch (SAXException e5) {
    		            try { // default
    		            	 reader = XMLReaderFactory.createXMLReader();
    		            }
    		            catch (SAXException e6) {
    		              throw new NoClassDefFoundError(
    		                "No SAX parser is available");
    		            }
    		          }
    		        }
    		      }
    		    } 
    		  }
    	
        
        JTreeContentHandler jTreeTableContentHandler = new JTreeContentHandler();
        ErrorHandler jTreeTableErrorHandler = new JTreeErrorHandler();
        JTreeLexicalHandler lexicalHandler = new JTreeLexicalHandler(jTreeTableContentHandler);

        // Register content handler
        reader.setContentHandler(jTreeTableContentHandler);

        // Register error handler
        reader.setErrorHandler(jTreeTableErrorHandler);
        
        // register handler for comments
        reader.setProperty("http://xml.org/sax/properties/lexical-handler",lexicalHandler);

        // Parse
        StringReader stringReader = new StringReader(xml);
        InputSource inputSource = new InputSource(stringReader);
        reader.parse(inputSource);

        // Build the tree model
        xmlModel = new XMLModel(jTreeTableContentHandler.getRoot());
        JTreeTable jTreeTable = new JTreeTable(xmlModel, true, false);
        jTreeTable.getTree().setCellRenderer(new TreePortionCellRenderer());
        
        return jTreeTable;
    }

    /**
     * <p> Static entry point for running the viewer. </p>
     */
    public static void main(String[] args) {
        try {
            if (args.length != 1) {
                System.out.println(
                    "Usage: java SAXTreeViewer " +
                    "[XML Document URI]");
                System.exit(0);
            }
            SAXTreeViewer viewer = new SAXTreeViewer();
            viewer.init(args[0]);
            viewer.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

/**
 * <b><code>JTreeContentHandler</code></b> implements the SAX
 *   <code>ContentHandler</code> interface and defines callback
 *   behavior for the SAX callbacks associated with an XML
 *   document's content, bulding up JTreeTable nodes.
 */
class JTreeContentHandler implements ContentHandler {

    /** Hold onto the locator for location information */
    //private Locator locator;

    /** Store URI to prefix mappings */
    private Map<String, String> namespaceMappings;

    /** Current node to add sub-nodes to */
    private XMLNode current = null;
    
    private XMLNode root = null;

	boolean doNamespace = false;

    /**
     * <p> Set up for working with the JTree. </p>
     *
     * @param treeModel tree to add nodes to.
     * @param base node to start adding sub-nodes to.
     */
    public JTreeContentHandler() {
        this.namespaceMappings = new HashMap<String, String>();
        this.current = new XMLNode("XML Document");
        this.root = this.current;
    }

    public XMLNode getRoot() {
    	return this.root;
    }
    /**
     * <p>
     *  Provide reference to <code>Locator</code> which provides
     *    information about where in a document callbacks occur.
     * </p>
     *
     * @param locator <code>Locator</code> object tied to callback
     *        process
     */
    public void setDocumentLocator(Locator locator) {
        // Save this for later use
        //this.locator = locator;
    }

    /**
     * <p>
     *  This indicates the start of a Document parse-this precedes
     *    all callbacks in all SAX Handlers with the sole exception
     *    of <code>{@link #setDocumentLocator}</code>.
     * </p>
     *
     * @throws <code>SAXException</code> when things go wrong
     */
    public void startDocument() throws SAXException {
        // No visual events occur here
    }

    /**
     * <p>
     *  This indicates the end of a Document parse-this occurs after
     *    all callbacks in all SAX Handlers.</code>.
     * </p>
     *
     * @throws <code>SAXException</code> when things go wrong
     */
    public void endDocument() throws SAXException {
        // No visual events occur here
    }

    /**
     * <p>
     *   This indicates that a processing instruction (other than
     *     the XML declaration) has been encountered.
     * </p>
     *
     * @param target <code>String</code> target of PI
     * @param data <code>String</code containing all data sent to the PI.
     *               This typically looks like one or more attribute value
     *               pairs.
     * @throws <code>SAXException</code> when things go wrong
     */
    public void processingInstruction(String target, String data)
        throws SAXException {

        DefaultMutableTreeNode pi = 
            new DefaultMutableTreeNode("PI (target = '" + target +
                                       "', data = '" + data + "')");
        current.add(pi);
    }

    /**
     * <p>
     *   This indicates the beginning of an XML Namespace prefix
     *     mapping. Although this typically occurs within the root element
     *     of an XML document, it can occur at any point within the
     *     document. Note that a prefix mapping on an element triggers
     *     this callback <i>before</i> the callback for the actual element
     *     itself (<code>{@link #startElement}</code>) occurs.
     * </p>
     *
     * @param prefix <code>String</code> prefix used for the namespace
     *                being reported
     * @param uri <code>String</code> URI for the namespace
     *               being reported
     * @throws <code>SAXException</code> when things go wrong
     */
    public void startPrefixMapping(String prefix, String uri) {
        // No visual events occur here.
        namespaceMappings.put(uri, prefix);
		doNamespace = true;
    }

    /**
     * <p>
     *   This indicates the end of a prefix mapping, when the namespace
     *     reported in a <code>{@link #startPrefixMapping}</code> callback
     *     is no longer available.
     * </p>
     *
     * @param prefix <code>String</code> of namespace being reported
     * @throws <code>SAXException</code> when things go wrong
     */
    public void endPrefixMapping(String prefix) {
        // No visual events occur here.
        for (Iterator<String> i = namespaceMappings.keySet().iterator(); 
             i.hasNext(); ) {

            String uri = i.next();
            String thisPrefix = namespaceMappings.get(uri);
            if (prefix.equals(thisPrefix)) {
                namespaceMappings.remove(uri);
                break;
            }
        }
    }

    /**
     * <p>
     *   This reports the occurrence of an actual element. It includes
     *     the element's attributes, with the exception of XML vocabulary
     *     specific attributes, such as
     *     <code>xmlns:[namespace prefix]</code> and
     *     <code>xsi:schemaLocation</code>.
     * </p>
     *
     * @param namespaceURI <code>String</code> namespace URI this element
     *               is associated with, or an empty <code>String</code>
     * @param localName <code>String</code> name of element (with no
     *               namespace prefix, if one is present)
     * @param qName <code>String</code> XML 1.0 version of element name:
     *                [namespace prefix]:[localName]
     * @param atts <code>Attributes</code> list for this element
     * @throws <code>SAXException</code> when things go wrong
     */
    public void startElement(String namespaceURI, String localName,
                             String qName, Attributes atts)
        throws SAXException {

    	String prefix = "";
        // Determine namespace
        if (namespaceURI.length() > 0) {
            prefix = namespaceMappings.get(namespaceURI);
        }

        XMLElementNode element = null;
		if (localName.equals("ProfileAttributes") && prefix.equals("tau")) {
			element = new XMLProfileNode(namespaceURI, 
				localName, qName, atts, prefix);
		} else if (localName.equals("attribute") && prefix.equals("tau")) {
			element = new XMLTAUAttributeElementNode(namespaceURI, 
				localName, qName, atts, prefix);
		} else {
        	element = new XMLElementNode(namespaceURI, 
				localName, qName, atts, prefix);
		}

        if (doNamespace && (prefix.length() > 0)) {
        	// add the namespace and style attributes
            XMLAttributeNode attribute = 
				new XMLAttributeNode("xmlns:" + prefix, namespaceURI);
            element.add(attribute);
			doNamespace = false;
        }
       	current.add(element);
        current = element;
    }

    /**
     * <p>
     *   Indicates the end of an element
     *     (<code>&lt;/[element name]&gt;</code>) is reached. Note that
     *     the parser does not distinguish between empty
     *     elements and non-empty elements, so this occurs uniformly.
     * </p>
     *
     * @param namespaceURI <code>String</code> URI of namespace this
     *                element is associated with
     * @param localName <code>String</code> name of element without prefix
     * @param qName <code>String</code> name of element in XML 1.0 form
     * @throws <code>SAXException</code> when things go wrong
     */
    public void endElement(String namespaceURI, String localName,
                           String qName)
        throws SAXException {

        // Walk back up the tree
        current = (XMLNode)current.getParent();
    }

    /**
     * <p>
     *   This reports character data (within an element).
     * </p>
     *
     * @param ch <code>char[]</code> character array with character data
     * @param start <code>int</code> index in array where data starts.
     * @param length <code>int</code> index in array where data ends.
     * @throws <code>SAXException</code> when things go wrong
     */
    public void characters(char[] ch, int start, int length)
        throws SAXException {

        String s = new String(ch, start, length);
        if (s.trim().length() > 0) {
	        current.setValue(s);
        }
    }

    /**
     * <p>
     *   This reports character data (within an element).
     * </p>
     *
     * @param ch <code>char[]</code> character array with character data
     * @param start <code>int</code> index in array where data starts.
     * @param length <code>int</code> index in array where data ends.
     * @throws <code>SAXException</code> when things go wrong
     */
    public void comment(char[] ch, int start, int length)
        throws SAXException {

        String s = new String(ch, start, length);
        if (s.trim().length() > 0) {
	        XMLCommentNode data =
	            new XMLCommentNode(trim(s));
	        current.add(data);
        }
    }

    /* remove leading whitespace */
    private static String ltrim(String source) {
        return source.replaceAll("^\\s+", "");
    }

    /* remove trailing whitespace */
    private static String rtrim(String source) {
        return source.replaceAll("\\s+$", "");
    }

    /* replace multiple whitespaces between words with single blank */
    private static String itrim(String source) {
        return source.replaceAll("\\s{2,}", " ");
    }

    /* remove all superfluous whitespaces in source string */
    private static String trim(String source) {
        return itrim(ltrim(rtrim(source)));
    }

//    private static String lrtrim(String source){
//        return ltrim(rtrim(source));
//    }

    /**
     * <p>
     * This reports whitespace that can be ignored in the
     * originating document. This is typically invoked only when
     * validation is ocurring in the parsing process.
     * </p>
     *
     * @param ch <code>char[]</code> character array with character data
     * @param start <code>int</code> index in array where data starts.
     * @param end <code>int</code> index in array where data ends.
     * @throws <code>SAXException</code> when things go wrong
     */
    public void ignorableWhitespace(char[] ch, int start, int length)
        throws SAXException {
        
        // This is ignorable, so don't display it
    }

    /**
     * <p>
     *   This reports an entity that is skipped by the parser. This
     *     should only occur for non-validating parsers, and then is still
     *     implementation-dependent behavior.
     * </p>
     *
     * @param name <code>String</code> name of entity being skipped
     * @throws <code>SAXException</code> when things go wrong
     */
    public void skippedEntity(String name) throws SAXException {
//        DefaultMutableTreeNode skipped =
//            new DefaultMutableTreeNode("Skipped Entity: '" + name + "'");
//        current.add(skipped);
    }
}

/**
 * <b><code>JTreeErrorHandler</code></b> implements the SAX
 *   <code>ErrorHandler</code> interface and defines callback
 *   behavior for the SAX callbacks associated with an XML
 *   document's warnings and errors.
 */
class JTreeErrorHandler implements ErrorHandler {

    /**
     * <p>
     * This will report a warning that has occurred; this indicates
     *   that while no XML rules were "broken", something appears
     *   to be incorrect or missing.
     * </p>
     *
     * @param exception <code>SAXParseException</code> that occurred.
     * @throws <code>SAXException</code> when things go wrong 
     */
    public void warning(SAXParseException exception)
        throws SAXException {
            
        System.out.println("**Parsing Warning**\n" +
                           "  Line:    " + 
                              exception.getLineNumber() + "\n" +
                           "  URI:     " + 
                              exception.getSystemId() + "\n" +
                           "  Message: " + 
                              exception.getMessage());        
        throw new SAXException("Warning encountered");
    }

    /**
     * <p>
     * This will report an error that has occurred; this indicates
     *   that a rule was broken, typically in validation, but that
     *   parsing can reasonably continue.
     * </p>
     *
     * @param exception <code>SAXParseException</code> that occurred.
     * @throws <code>SAXException</code> when things go wrong 
     */
    public void error(SAXParseException exception)
        throws SAXException {
        
        System.out.println("**Parsing Error**\n" +
                           "  Line:    " + 
                              exception.getLineNumber() + "\n" +
                           "  URI:     " + 
                              exception.getSystemId() + "\n" +
                           "  Message: " + 
                              exception.getMessage());
        throw new SAXException("Error encountered");
    }

    /**
     * <p>
     * This will report a fatal error that has occurred; this indicates
     *   that a rule has been broken that makes continued parsing either
     *   impossible or an almost certain waste of time.
     * </p>
     *
     * @param exception <code>SAXParseException</code> that occurred.
     * @throws <code>SAXException</code> when things go wrong 
     */
    public void fatalError(SAXParseException exception)
        throws SAXException {
    
        System.out.println("**Parsing Fatal Error**\n" +
                           "  Line:    " + 
                              exception.getLineNumber() + "\n" +
                           "  URI:     " + 
                              exception.getSystemId() + "\n" +
                           "  Message: " + 
                              exception.getMessage());        
        throw new SAXException("Fatal Error encountered");
    }
}

class JTreeLexicalHandler implements LexicalHandler {

	private JTreeContentHandler treeContentHandler = null;
	public JTreeLexicalHandler(JTreeContentHandler treeContentHandler) {
		this.treeContentHandler = treeContentHandler;
	}

	public void comment(char[] ch, int start, int length) throws SAXException {
		treeContentHandler.comment(ch, start, length);
	}

	public void endCDATA() throws SAXException {}
	public void endDTD() throws SAXException {}
	public void endEntity(String name) throws SAXException {}
	public void startCDATA() throws SAXException {}
	public void startDTD(String name, String publicId, String systemId) throws SAXException {}
	public void startEntity(String name) throws SAXException {}
}
