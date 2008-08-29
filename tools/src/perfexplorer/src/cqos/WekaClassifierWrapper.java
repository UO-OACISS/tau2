/**
 * 
 */
package cqos;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This class is a wrapper around the Weka classifier.  The intention is that PerfExplorer
 * will generate a classifier (using this Class), and serialize it out to disk.  Later, a runtime CQoS 
 * component will read the classifier (using this Class), and use it to make performance predictions for
 * runtime decision making.
 * 
 * The implementation is not the most efficient, but it works.
 * 
 * Step 1:  (the constructor)
 *   The constructor saves the List of incoming data, and builds an array of
 *   	metadata fields, A.K.A. attribute names.
 * Step 2:  (buildClassifier)
 *   Call "checkForNominalAttributes()", and see which attributes are nominal, and which
 *      are numeric.  This is easier(?) than just sticking them all in and filtering them
 *      later, which is another option with Weka.
 * Step 3:  (buildClassifier)
 *   Iterate over the training data, and build Weka data structures, and then 
 *      create the Weka Classifier.
 * Step 4:
 *   Call getClass() to classifiy a new test instance.   Optionally call "getConfidence" to
 *      get the probability that this instance is actually that class.    
 * 
 * @author khuck
 *
 */
public class WekaClassifierWrapper implements Serializable {

	/**
	 *  Here is the serialization ID. 
	 */
	private static final long serialVersionUID = -3288768059845773266L;
	
	// here are the classifiers which have been tested.
	private static final String SUPPORT_VECTOR_MACHINE = "weka.classifiers.functions.SMO";
	private static final String NAIVE_BAYES = "weka.classifiers.bayes.NaiveBayes";
    private static final String MULTILAYER_PERCEPTRON = "weka.classifiers.functions.MultilayerPerceptron";
	
    // member variables.
	private String classifierType = WekaClassifierWrapper.MULTILAYER_PERCEPTRON;
	private String[] metadataFields = null; 	// array of independent variables
	private List/*<Map<String,String>>*/ trainingData = null;  	// incoming map of training data
	private int fieldCount = 0;               // convenience variable <== metadataFields.length
	private String classLabel = null;         // the label of the "dependent" variable
	private Classifier cls = null;            // the Weka classifier.
	private Attribute classAttr = null;       // the Weka attribute which corresponds to the "dependent" variable
	private FastVector attributes = new FastVector();        // the Weka FastVector of attributes.
	private Attribute[] attArray = null;      // local array of attributes, so we don't have to cast (using java 1.4)
	private double confidence = 0.0;          // the confidence of the class prediction
	private String className = null;          // the class prediction
	private Set[]/*String*/ nominalAttributes = null;    // this helps us figure out what's a numeric variable and what isn't
	
	/**
	 * The one and only constructor.  The Constructor will use the trainingData passed in to build the
	 * classifier, and will classify based on the string in the classLabel.
	 * @param trainingData a List of Maps of Strings to Strings.
	 * @param classLabel The key in the maps which identifies the class of each instance
	 * @throws Exception If you don't pass in the right data...
	 */
	WekaClassifierWrapper (List/*<Map<String,String>>*/ trainingData, String classLabel) throws Exception {
		this.trainingData = trainingData;
		this.classLabel = classLabel;
		
		// get the first instance in the list of training data
		Object tmpMap = trainingData.get(0);

		// check the type, just to be sure...
		if (tmpMap instanceof HashMap) {
			HashMap hashMap = (HashMap) tmpMap;
			// get the set of keys from the hash map
			this.metadataFields = new String[hashMap.keySet().size()];
			
			// Weka works best if the first attribute is the class (dependent variable)
			this.metadataFields[this.fieldCount++] = classLabel;
			
			// save the keys from the map into our local array.
	        Iterator keyIter = hashMap.keySet().iterator();
	        while (keyIter.hasNext()) {
	        	String key = (String) keyIter.next();
	        	if (!key.equals(classLabel))
	        		this.metadataFields[this.fieldCount++] = key;
	        }
		} else {
			throw (new Exception("WekaClassifierWrapper only supports a training data in the form:\n\tList<Map<String,String>> "));
		}
	}
	
	/**
	 * This method will build the classifier, using the data passed in to the constructor.
	 *
	 */
	public void buildClassifier () {

		// inspect the attribute types.
		checkForNominalAttributes();
		
        // this array is for us locally, to iterate through the attributes when necessary
        attArray = new Attribute[this.fieldCount];
		
		// create all the attributes
		for (int index = 0 ; index < this.fieldCount ; index++) {
			Attribute attr = null;
			String key = this.metadataFields[index];
			
			// if this is a nominal attribute, we will have a set of possible values.
			// if not, it's a numeric attribute.
			if (this.nominalAttributes[index] != null) {

				// create a vector fo classes for the possible values
				FastVector classes = new FastVector();
				
				// iterate through them, and put them in the FastVector
				Iterator classIter = this.nominalAttributes[index].iterator();
				while(classIter.hasNext()) {
					classes.addElement(classIter.next());
				}
				
				// this is a nominal attribute.
				attr = new Attribute(key, classes);
				
				// if this is our class attribute, remember that.  It BETTER be the first one.
				if (key.equals(this.classLabel))
					classAttr = attr;
			} else {
				// this is a numeric attribute.
				attr = new Attribute(key);
			}
			// add our attributes to our local array, and to the Weka FastVector
			attArray[index] = attr;
			this.attributes.addElement(attr);
		}
		
		// create the set of training instances
        Instances train = new Instances("train", this.attributes, this.trainingData.size());
        
        // set the class for the instances to be the first attribute, the dependent variable
        train.setClass(classAttr);
        
        // iterate over the training data
        Iterator dataIter = trainingData.iterator();
        while (dataIter.hasNext()) {
        	
        	// get the set of name/value pairs for this instance
        	Map/*<String,String>*/ tmpMap = (Map)dataIter.next();
        	
        	// create the Weka Instance
			Instance inst = new Instance(this.fieldCount);
			
			// set the attributes in the Instance
			for (int i = 0 ; i < attArray.length ; i++) {
				String value = (String)tmpMap.get(attArray[i].name());
				try {
					inst.setValue(attArray[i], Double.parseDouble(value));
				} catch (Exception e) {
					inst.setValue(attArray[i], value);
				}
			}
			train.add(inst);
		}

		try {
	        // build the classifier!
	        this.cls = Classifier.forName(this.classifierType, null);
	        // train the classifier!
			this.cls.buildClassifier(train);
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}

		return;		
	}
	
	/**
	 * This method will iterate over the training data, and figure out which parameters
	 * (they all come in as Strings) are numeric.  Those that are numeric don't get a Set
	 * of potential values.
	 *
	 */
	private void checkForNominalAttributes() {
        this.nominalAttributes = new Set[this.fieldCount];
        
        // iterate over the training data, and test if the data can be parsed as a double
        Iterator dataIter = trainingData.iterator();
        while (dataIter.hasNext()) {
        	Map/*<String,String>*/ tmpMap = (Map)dataIter.next();
			for (int i = 0 ; i < this.metadataFields.length ; i++) {
				String value = (String)tmpMap.get(this.metadataFields[i]);
				try {
					double test = Double.parseDouble(value);
				} catch (Exception e) {
					nominalAttributes[i] = new HashSet();
				}
			}
		}
        
        // ok, we know which fields are nominal, so now get their potential values.
        dataIter = trainingData.iterator();
        while (dataIter.hasNext()) {
        	Map/*<String,String>*/ tmpMap = (Map)dataIter.next();
			for (int i = 0 ; i < this.metadataFields.length ; i++) {
				String value = (String)tmpMap.get(this.metadataFields[i]);
				if (nominalAttributes[i] != null) {
					nominalAttributes[i].add(tmpMap.get(this.metadataFields[i]));
				}
			}
		}
	}

	/**
	 * Once the classifier has been built, you can pass in test instances, and get them
	 * classified.
	 * 
	 * @param inputFields
	 * @return the class of the instance, based on the classification of the input fields.
	 */
	public String getClass(Map/*<String,String>*/ inputFields) {
		
		// create the weka data structures - this looks silly that we create an "Instances"
		// object with one instance, but initializing it with the array of Weka Attributes
		// makes everything work "better".
        Instances test = new Instances("test", this.attributes, 1);
    	Instance inst = new Instance(attArray.length);
    	
    	// iterate over the input fields, and set the Weka Attributes in the Weka Instance
    	int classIndex = 0;
		for (int i = 0 ; i < attArray.length ; i++) {
			String tmp = (String)inputFields.get(attArray[i].name());
			if (tmp != null) {
				if (attArray[i].isNumeric())
					inst.setValue(attArray[i], Double.parseDouble(tmp));
				else
					inst.setValue(attArray[i], tmp);
			}
			if (attArray[i].name().equals(this.classLabel))
				classIndex = i;
		}
		
		// put our one instance into the instances object
		test.add(inst);
		// tell the instances object which attribute is the "class"
		test.setClassIndex(classIndex);
        
		// get our instance back - it may have changed from the previous command (Weka side effects!)
		inst = test.firstInstance();
        try {
        	// get the classification, which comes back as probabilities for each class.
	        double[] dist = cls.distributionForInstance(inst);
	        
//	        System.out.print(" [");
//	        for (int i = 0 ; i < dist.length ; i++)
//	            System.out.print (dist[i] + ",");
//	        System.out.print("] ");
	        
	        // choose the class with the highest probability.
			int i = 0;
			for (int j = 1 ; j < dist.length; j++) {
				if (dist[j] > dist[i]) {
					i = j;
				}
			}
			
			// save our result 
			this.confidence = dist[i];
			this.className = classAttr.value(i);
        } catch (Exception e) {
        	// oopsie!
        	System.err.println(e.getMessage());
        	e.printStackTrace();
        }
		
		return this.className;
	}

	/**
	 * Get the confidence of the classification
	 * 
	 * @return
	 */
	public double getConfidence() {
		return confidence;
	}

	/**
	 * Get the class name (in case you didn't save it from the getClass() call)
	 * 
	 * @return
	 */
	public String getClassName() {
		return className;
	}

	/**
	 * Serialize the wrapper out to disk.
	 * 
	 * @param fileName
	 * @param wrapper
	 */
	public static void writeClassifier(String fileName, WekaClassifierWrapper wrapper) {
		try {
			OutputStream os = new FileOutputStream(fileName);
			ObjectOutput oo = new ObjectOutputStream(os);
			oo.writeObject(wrapper);
			oo.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}

	/**
	 * Read the wrapper back in from disk.
	 * @param fileName
	 * @return the (hopefully) trained classifier.
	 */
	public static WekaClassifierWrapper readClassifier(String fileName) {
		WekaClassifierWrapper classifier = null;
		try {
			InputStream is = new FileInputStream(fileName);
			ObjectInput oi = new ObjectInputStream(is);
			Object newObj = oi.readObject();
			oi.close();
			classifier = (WekaClassifierWrapper)newObj;
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		return classifier;
	}

	/**
	 * @return the classifierType
	 */
	public String getClassifierType() {
		return classifierType;
	}

	/**
	 * @param classifierType the classifierType to set
	 */
	public void setClassifierType(String classifierType) {
		this.classifierType = classifierType;
	}

	/**
	 * Main method for testing this wrapper.
	 * 
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		
		// create one of those there wrappers...
		WekaClassifierWrapper wrapper = null;
		
		// default behavior - do both the read and the write
		boolean read = true;
		boolean write = true;
		
		// if the user wants just reading or writing, let them do so... (testing purposes)
		if (args.length == 1) {
			if (args[0].equals("write")) {
				read = false;
			} else if (args[0].equals("read")) {
				write = false;
			}
		}
		
		// train that classifier!
		if (write) {
			System.out.println("Writing...");
			
			// create a list for our training data
			List trainingData = new ArrayList();
			
			// this is the attribute that we want as our dependent variable.  All the others
			// will be used as independent variables for the classification.
			String classLabel = "method name";
			
			// first instance
			Map myMap = new HashMap();
			myMap.put("method name", "method 1");   // the class for this instance
			myMap.put("sleep value", "0");          // the "real" independent variable
			myMap.put("bogus", "A");                // a "bogus" variable (noise, really)
			myMap.put("noise", "A");                // nominal noise
			myMap.put("more noise", "24.562");      // numeric noise
			trainingData.add(myMap);
			
			// second instance
			myMap = new HashMap();
			myMap.put("method name", "method 1");
			myMap.put("sleep value", "1");
			myMap.put("bogus", "A");
			myMap.put("noise", "A");
			myMap.put("more noise", "24.562");
			trainingData.add(myMap);
			
			// third instance
			myMap = new HashMap();
			myMap.put("method name", "method 1");
			myMap.put("sleep value", "2");
			myMap.put("bogus", "B");
			myMap.put("noise", "A");
			myMap.put("more noise", "24.562");
			trainingData.add(myMap);
			
			// fourth instance
			myMap = new HashMap();
			myMap.put("method name", "method 2");
			myMap.put("sleep value", "3");
			myMap.put("bogus", "B");
			myMap.put("noise", "A");
			myMap.put("more noise", "24.562");
			trainingData.add(myMap);
			
			// fifth instance
			myMap = new HashMap();
			myMap.put("method name", "method 2");
			myMap.put("sleep value", "4");
			myMap.put("bogus", "B");
			myMap.put("noise", "A");
			myMap.put("more noise", "24.562");
			trainingData.add(myMap);
			
			System.out.println(trainingData);
			
			// build an SVM classifier
			wrapper = new WekaClassifierWrapper (trainingData, classLabel);
			wrapper.setClassifierType(WekaClassifierWrapper.SUPPORT_VECTOR_MACHINE);
			wrapper.buildClassifier();

			// do some classifying with it
			System.out.println("\n" + wrapper.getClassifierType());
	        for (int i = 0 ; i < 5 ; i++) {
	    		Map/*<String,String>*/ inputFields = new HashMap/*<String,String>*/();
	        	inputFields.put("sleep value", Integer.toString(i));
	        	inputFields.put("bogus", (i<2?"A":"B"));
	        	// notice there's no noise fields!  We don't need ALL the training data...
		        System.out.println(inputFields + ", " + wrapper.getClass(inputFields) + 
		        		", confidence: " + wrapper.getConfidence());
	        }

	        // build a Naive Bayes classifier
			wrapper = new WekaClassifierWrapper (trainingData, classLabel);
			wrapper.setClassifierType(WekaClassifierWrapper.NAIVE_BAYES);
			wrapper.buildClassifier();

			// do some classifying with it
			System.out.println("\n" + wrapper.getClassifierType());
	        for (int i = 0 ; i < 5 ; i++) {
	    		Map/*<String,String>*/ inputFields = new HashMap/*<String,String>*/();
	        	inputFields.put("sleep value", Integer.toString(i));
	        	inputFields.put("bogus", (i<2?"A":"B"));
	        	// this test does have the noise variables
				inputFields.put("noise", "A");
				inputFields.put("more noise", "0.0");
		        System.out.println(inputFields + ", " + wrapper.getClass(inputFields) + 
		        		", confidence: " + wrapper.getConfidence());
	        }    

	        // build a multilayer perceptron classifier
	        wrapper = new WekaClassifierWrapper (trainingData, classLabel);
	        wrapper.setClassifierType(WekaClassifierWrapper.MULTILAYER_PERCEPTRON);
			wrapper.buildClassifier();

			// do some classifying with it
			System.out.println("\n" + wrapper.getClassifierType());
	        for (int i = 0 ; i < 5 ; i++) {
	    		Map/*<String,String>*/ inputFields = new HashMap/*<String,String>*/();
	        	inputFields.put("sleep value", Integer.toString(i));
	        	inputFields.put("bogus", (i<2?"A":"B"));
				inputFields.put("noise", "A");
				inputFields.put("more noise", "0.0");
		        System.out.println(inputFields + ", " + wrapper.getClass(inputFields) + 
		        		", confidence: " + wrapper.getConfidence());
	        }    
			
	        // serialize it to disk!
			WekaClassifierWrapper.writeClassifier("/tmp/pleasework", wrapper);
		}
		
		if (read) {
			System.out.println("Reading...");
			// read in our classifier
			wrapper = WekaClassifierWrapper.readClassifier("/tmp/pleasework");
			
			// do some classifying with it
			System.out.println("\n" + wrapper.getClassifierType());
	        for (int i = 0 ; i < 5 ; i++) {
	    		Map/*<String,String>*/ inputFields = new HashMap/*<String,String>*/();
	        	inputFields.put("sleep value", Integer.toString(i));
	        	inputFields.put("bogus", (i<2?"A":"B"));
				inputFields.put("noise", "A");
				inputFields.put("more noise", "0.0");
		        System.out.println(inputFields + ", " + wrapper.getClass(inputFields) + 
		        		", confidence: " + wrapper.getConfidence());
	        }    
		}
	}
}
