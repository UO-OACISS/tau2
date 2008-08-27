/**
 * 
 */
package glue;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectOutput;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Set;
import weka.core.*;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

/**
 * @author khuck
 *
 */
public class CQoSClassifierOperation extends AbstractPerformanceOperation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5899733736890853588L;
	private String metric = "Time";
	private Set<String> metadataFields = null;
	private String classLabel = null;
	private Classifier cls = null;
	private Attribute classAttr = null;
	private FastVector atts = null;
	private Attribute[] attArray = null;
	private double confidence = 0.0;
	private String className = null;
	
	/**
	 * @param inputs
	 */
	public CQoSClassifierOperation(List<PerformanceResult> inputs, String metric, 
			Set<String> metadataFields, String classLabel) {
		super(inputs);
		this.metric = metric;
		this.metadataFields = metadataFields;
		this.classLabel = classLabel;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		// create a map to store the UNIQUE tuples
		Map<Hashtable<String,String>,PerformanceResult> tuples = 
			new HashMap<Hashtable<String,String>,PerformanceResult>();
		
		// iterate through the inputs
		for (PerformanceResult input : this.inputs) {
			// create a local Hashtable 
			Hashtable<String,String> localMeta = new Hashtable<String,String>();
			// get the input's metadata
			TrialMetadata tmp = new TrialMetadata(input);
			Hashtable<String,String> meta = tmp.getCommonAttributes();
			// create a reduced hashtable
			if (this.metadataFields != null) {
				for (String key : this.metadataFields) {
					localMeta.put(key, meta.get(key));
				}
			// otherwise, if the user didn't specify a set of properties, use them all (?)
			} else {
				for (String key : meta.keySet()) {
					localMeta.put(key, meta.get(key));
				}				
			}
			// put the hashtable in the set: if its performance is "better" than the existing one
			// or if it doesn't exist yet.
			PerformanceResult result = tuples.get(localMeta);
			if (result == null) {
				tuples.put(localMeta,input);
			} else {
				if (input.getInclusive(0, input.getMainEvent(), metric) < 
						result.getInclusive(0, result.getMainEvent(), this.metric)) {
					tuples.put(localMeta,input);					
				}
			}
		}
		
		// some debugging output...
		
/*		System.out.println(tuples.size());
		for (Hashtable<String,String> tmp : tuples.keySet()) {
			System.out.print(tmp.toString() + ": " + tuples.get(tmp).getInclusive(0, tuples.get(tmp).getMainEvent(), metric)/1000000);
			System.out.println(": " + (new TrialMetadata(tuples.get(tmp)).getCommonAttributes().get(classLabel)));
		}
*/		
		
		// TODO: for each input parameter, check if it is numeric or categorical.  For now, assume numeric.
		
		// build the classifier!
        // this vector is for Weka
        this.atts = new FastVector();
        // this array is for us locally, to iterate through the attributes when necessary
        // the size is all the metadata fields, plus the class
        attArray = new Attribute[this.metadataFields.size()+1];
		// set the classes to null - we will convert them to nominal with a filter later.
        FastVector classes = null;
        // create the class attribute - the dependent variable, if you will.
		classAttr = new Attribute(this.classLabel, classes);
		// save the first attribute
		attArray[0] = classAttr;
		this.atts.addElement(classAttr);
		// create all the other attributes - the independent variables, if you will.
        int i = 1;
		for (String key : this.metadataFields) {
			attArray[i] = new Attribute(key);
			this.atts.addElement(attArray[i]);
			i++;
		}
		
		// create the set of training instances
        Instances train = new Instances("train", this.atts, tuples.size());
        // set the class for the instances to be the first attribute, the dependent variable
        train.setClass(attArray[0]);
        
        // ok, we have the set of "optimal" methods for each unique tuple.  Convert them to Instances.
		for (Hashtable<String,String> tmp : tuples.keySet()) {
			// the instance has one more field - the method name, don't forget...
			Instance inst = new Instance(tmp.size()+1);
			// set the method name (dependent parameter, AKA the class)
			inst.setValue(attArray[0], (new TrialMetadata(tuples.get(tmp)).getCommonAttributes().get(classLabel)));
			// set the independent parameters
			for (i = 1 ; i < attArray.length ; i++) {
				inst.setValue(attArray[i], Double.parseDouble(tmp.get(attArray[i].name())));
			}
			train.add(inst);
		}

		try {
			// tell the dataset which attribute is the class
			train.setClassIndex(0);
			// we need to filter the class to be a Nominal attribute, not just a string.
	        StringToNominal filter = new StringToNominal();
	        // For some reason, this is indexed at 1, not 0.  Nice consistency...
	        filter.setAttributeIndex("1");
	        filter.setInputFormat(train);
	        train = Filter.useFilter(train,filter);

	        // build the classifier!
//			this.cls = Classifier.forName("weka.classifiers.functions.SMO", null);
//			this.cls = Classifier.forName("weka.classifiers.bayes.NaiveBayes", null);
	        this.cls = Classifier.forName("weka.classifiers.functions.MultilayerPerceptron", null);
	        // train the classifier!
			this.cls.buildClassifier(train);
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}

		return null;
	}

	public Classifier getClassifier() {
		return cls;
	}

	public String getClass(Map<String,String> inputFields) {
        Instances test = new Instances("test", atts, 3);
    	Instance inst = new Instance(attArray.length);
		for (int i = 1 ; i < attArray.length ; i++) {
			inst.setValue(attArray[i], Double.parseDouble(inputFields.get(attArray[i].name())));
		}
		test.add(inst);
		test.setClassIndex(0);
        
		inst = test.firstInstance();
        try {
//	        System.out.print(" [");
	        double[] dist = cls.distributionForInstance(inst);
//	        for (int i = 0 ; i < dist.length ; i++) {
//	            System.out.print (dist[i] + ",");
//	        }
//	        System.out.print("] ");
			int i = 0;
			for (int j = 1 ; j < classAttr.numValues(); j++) {
				if (dist[j] > dist[i]) {
					i = j;
				}
			}
			this.confidence = dist[i];
			this.className = classAttr.value(i);
        } catch (Exception e) {
        	System.err.println(e.getMessage());
        	e.printStackTrace();
        }
		
		return this.className;
	}

	public double getConfidence() {
		return confidence;
	}

	public void setConfidence(double confidence) {
		this.confidence = confidence;
	}

	public String getClassName() {
		return className;
	}

	public void setClassName(String className) {
		this.className = className;
	}
	
	public static void writeClassifier(String fileName, CQoSClassifierOperation operation) {
		try {
			OutputStream os = new FileOutputStream(fileName);
			ObjectOutput oo = new ObjectOutputStream(os);
			oo.writeObject(operation);
			oo.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}

	public static CQoSClassifierOperation readClassifier(String fileName) {
		CQoSClassifierOperation classifier = null;
		try {
			InputStream is = new FileInputStream(fileName);
			ObjectInput oi = new ObjectInputStream(is);
			Object newObj = oi.readObject();
			oi.close();
			classifier = (CQoSClassifierOperation)newObj;
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		return classifier;
	}
}
