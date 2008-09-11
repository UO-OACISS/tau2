/**
 * 
 */
package cqos;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class CQoSClassifier {

	/**
	 * 
	 */
	public CQoSClassifier() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println("Reading...");
		String fileName = args[0];
		// read in our classifier
		WekaClassifierWrapper wrapper = WekaClassifierWrapper.readClassifier(fileName);

/*		for molecule_name in benzo_a_naphthacene benzo_b_triphenylene dibenz_ah_anthracene dibenz_aj_anthracene pentacene picene ; do
		    for basis_set in CCD N31-6-1-1 N31-6-1 N31-6 ; do
		        for run_type in ENERGY ; do
		            for scf_type in DIR CON ; do
		                for nodes in 1 2 4 ; do
		                    for cores in 1 2 4 ; do*/

		Set<String> molecules = new HashSet<String>();
		molecules.add("benzo_a_naphthacene");
		molecules.add("benzo_b_triphenylene");
		molecules.add("dibenz_ah_anthracene");
		molecules.add("dibenz_aj_anthracene");
		molecules.add("pentacene");
		molecules.add("picene");
		Set<String> bases = new HashSet<String>();
		bases.add("CCD");
		bases.add("N31-6-1-1");
		bases.add("N31-6-1");
		bases.add("N31-6");
		String runType = "ENERGY";
		Set<String> scf = new HashSet<String>();
		scf.add("DIR");
		scf.add("CON");
		
		// do some classifying with it
		System.out.println("\n" + wrapper.getClassifierType());
		for (String moleculeName : molecules) {
			for (String basisSet : bases) {
//			for (String scfType : scf) {
		        for (int n = 1 ; n < 9 ; n++) {
		            for (int c = 1 ; c < 9 ; c++) {
			    		Map/*<String,String>*/ inputFields = new HashMap/*<String,String>*/();
			        	inputFields.put("molecule name", moleculeName);
			        	inputFields.put("basis set", basisSet);
//			        	inputFields.put("scf type", scfType);
//			        	inputFields.put("run type", runType);
			        	inputFields.put("node count", Integer.toString(n));
			        	inputFields.put("core count", Integer.toString(c));
				        System.out.println(inputFields + ", " + wrapper.getClass(inputFields) + 
				        		", confidence: " + wrapper.getConfidence());
		            }
		        }    
			}
		}
	}

}
