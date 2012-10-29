package edu.uoregon.tau.perfexplorer.client;

import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;

import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;

import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.common.RMIPerformanceResults;

public class PerfExplorerCorrelationPane extends JScrollPane implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1776096417971557334L;

	private static PerfExplorerCorrelationPane thePane = null;

	private JPanel imagePanel = null;
	private Hashtable<String, RMIPerformanceResults> resultsHash = null;
	private RMIPerformanceResults results = null;

	public static PerfExplorerCorrelationPane getPane () {
		if (thePane == null) {
			JPanel imagePanel = new JPanel(new BorderLayout());
			//imagePanel.setPreferredScrollableViewportSize(new Dimension(400, 400));
			thePane = new PerfExplorerCorrelationPane(imagePanel);
		}
		thePane.repaint();
		return thePane;
	}

	private PerfExplorerCorrelationPane (JPanel imagePanel) {
		super(imagePanel);
		this.imagePanel = imagePanel;
		this.resultsHash = new Hashtable<String, RMIPerformanceResults>();
		JScrollBar jScrollBar = this.getVerticalScrollBar();
		jScrollBar.setUnitIncrement(35);
	}

	public JPanel getImagePanel () {
		return imagePanel;
	}

	private boolean sortByR=false;
	private ButtonGroup corSortGroup;
	private JRadioButton corsortR;
	private JRadioButton corsortN;
	private static final String RValue="RValue";
	private static final String DName="DName";
	private class CorSortListener implements ActionListener{

		public void actionPerformed(ActionEvent e) {
			//sortByR=((JCheckBox)e.getSource()).isSelected();
			if(e.getSource()==corsortR){
				
			}
			if(e.getActionCommand().equals(RValue)){
				sortByR=true;
			}else{
				sortByR=false;
			}
			//sortByR=corsort.isSelected();
			imagePanel.setVisible(false);
			updateImagePanel(false);
			
			imagePanel.repaint();
			imagePanel.setVisible(true);
		}
	}
	CorSortListener CSL = new CorSortListener();
	
	
	public void updateImagePanel(){
		updateImagePanel(true);
	}
	
	private void updateImagePanel (boolean reissue) {
		
		
		imagePanel.removeAll();
		PerfExplorerModel model = PerfExplorerModel.getModel();
		if ((model.getCurrentSelection() instanceof Metric) || 
			(model.getCurrentSelection() instanceof Trial)) {
			// check to see if we have these results already
			//results = (RMIPerformanceResults)resultsHash.get(model.toString());
			//if (results == null) {
			if(reissue){
				PerfExplorerConnection server = PerfExplorerConnection.getConnection();
				results = server.getCorrelationResults(model);
			}
			//}
			if (results.getResultCount() == 0) {
				return;
			}
			int iStart = 0;
			List<String> descriptions = results.getDescriptions();
			List<byte[]> thumbnails = results.getThumbnails();
			int imageCount = descriptions.size();
			resultsHash.put(model.toString(), results);
			JPanel imagePanel2 = null;
			corSortGroup=new ButtonGroup();
			corsortN = new JRadioButton();
			corsortN.setText("Alphabetical Ordering");
			corsortN.setSelected(!sortByR);
			corsortN.addActionListener(CSL);
			corsortN.setActionCommand(DName);
			corsortR = new JRadioButton();
			corsortR.setText("R Value Ordering");
			corsortR.setSelected(sortByR);
			corsortR.addActionListener(CSL);
			corsortR.setActionCommand(RValue);
			corSortGroup.add(corsortN);
			corSortGroup.add(corsortR);
			if(sortByR){
				List<DescribedImage> dlist =new ArrayList<DescribedImage>(imageCount);
				for(int i=0;i<imageCount;i++){
					dlist.add(new DescribedImage(descriptions.get(i),thumbnails.get(i),i));
				}
				Collections.sort(dlist);
				//System.out.println(results.getResultCount()+" "+(results.getResultCount()/5)+" "+(results.getResultCount()/5)*5);
				int cols=5;
				imagePanel2 = new JPanel(new GridLayout((results.getResultCount()/cols)+1, cols));
				imagePanel2.add(corsortN);
				imagePanel2.add(corsortR);
				for(int i=0;i<cols-1;i++){
					imagePanel2.add(new JLabel());
				}
				for (int i = iStart ; i < imageCount ; i++) {
					DescribedImage di = dlist.get(i);
					ImageIcon icon = new ImageIcon((di.getThumbnail()));
					String description = (di.getDescription());
					int id = di.getID();
					PerfExplorerImageButton button = new PerfExplorerImageButton(icon, id, description);
					button.addActionListener(this);
					imagePanel2.add(button);
				}
				
			}
			else{
			int sqrt = (int) java.lang.Math.sqrt(results.getResultCount());
			imagePanel2 = new JPanel(new GridLayout((results.getResultCount()/sqrt)+1, sqrt));
			imagePanel2.add(corsortN);
			imagePanel2.add(corsortR);
			for(int i=0;i<sqrt-2;i++){
				imagePanel2.add(new JLabel());
			}
			for (int i = iStart ; i < imageCount ; i++) {
				ImageIcon icon = new ImageIcon((thumbnails.get(i)));
				String description = (descriptions.get(i));
				PerfExplorerImageButton button = new PerfExplorerImageButton(icon, i, description);
				button.addActionListener(this);
				imagePanel2.add(button);
			}
			}
			imagePanel.add(imagePanel2, BorderLayout.SOUTH);
		}
		 //this.repaint();
	}
	private class DescribedImage implements Comparable<DescribedImage>{
		
		public String getDescription() {
			return description;
		}

		public byte[] getThumbnail() {
			return thumbnail;
		}

		public DescribedImage(String description, byte[] thumbnail, int id) {
			super();
			this.description = description;
			this.thumbnail = thumbnail;
			this.id=id;
		}

		public int getID(){
			return this.id;
		}
		
		private String description;
		private byte[] thumbnail;
		private int id;

		public int compareTo(DescribedImage arg0) {
			Double d0=getR(this.description);
			Double d1=getR(arg0.description);
			return Double.compare(d0, d1);
//			return d0 < d1 ? -1
//			         : d0 > d1 ? 1
//			         : 0;
		}
		
	}
//	private static class RComp implements Comparator<String>{
//
//		public int compare(String arg0, String arg1) {
//			double d0=getR(arg0);
//			double d1=getR(arg1);
//			return d0 < d1 ? -1
//			         : d0 > d1 ? 1
//			         : 0;
//		}
//		
//	}

	private static double getR(String description){
		
		int cdex=description.lastIndexOf("R:");
		double result=Double.NaN;
		if(cdex>=0){
			cdex+=2;
			String doub=description.substring(cdex);
			if(doub!=null&&doub.length()>0)
				result = Double.parseDouble(doub);
		}
		//System.out.println(description+" resolves to: "+result);
		return result;
	}
	
	public void actionPerformed(ActionEvent e) {
		int index = Integer.parseInt(e.getActionCommand());
		// create a new modal dialog with the big image

		String description = (String)(results.getDescriptions().get(index));
		ImageIcon icon = new ImageIcon((byte[])(results.getImages().get(index)));
		JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), null, description, JOptionPane.PLAIN_MESSAGE, icon);

	}
}
