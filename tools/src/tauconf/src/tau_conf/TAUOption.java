/*
 * TAUOption.java
 *
 * Created on July 27, 2007, 4:44 PM
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package tau_conf;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JOptionPane;

/**
 *
 * @author wspear
 */
public class TAUOption {
    
    protected static final int TOGGLE=0;
    protected static final int TEXT=1;
    protected static final int FILE=2;
    protected static final int DIR=3;
    protected static final int COMBO=4;
    protected static final int SPINNER=5;
    
    protected String[] comboOpts;
    
    protected int type;
    protected String optFlag;
    protected String helpText;
    protected String helpCommand;
    protected JCheckBox optToggle=null;
    protected JButton helpButton=null;
    
    
    
    /** Creates a new instance of TAUOption */
    public TAUOption(String optFlag, String helpCommand, String helpText) {
        
    int inc=80;
    int lendex=80;
    int freespace=0;
    while(lendex<helpText.length()){
        freespace = helpText.lastIndexOf(' ', lendex-1);
        helpText=helpText.substring(0,freespace+1)+'\n'+helpText.substring(freespace+1);
        lendex+=inc;
    }
        this.helpText=helpText;
        this.optFlag=optFlag;
        if(optFlag.endsWith("="))
            optFlag=optFlag.substring(0,optFlag.length()-2);
        this.helpCommand=optFlag;
        if(helpCommand.length()>0)
        {
            this.helpCommand+="="+helpCommand;
        }
        helpButton=new JButton();
        optToggle=new JCheckBox();
    }
    
    protected void QButton()
    {
        JOptionPane.showMessageDialog(null,helpText,helpCommand,JOptionPane.INFORMATION_MESSAGE);
    }
    /*
    protected void CheckChecked()
    {
        //String entry=" -mpi ";
	if(optToggle.isSelected()){
	    if(configureline.indexOf(optFlag)==-1){
		configureline+=optFlag;
	    }
	} else{
	    if(configureline.indexOf(optFlag)>-1)
		configureline = configureline.replaceFirst(optFlag,"");
	}
	commandTextArea.setText(configureline); updateITCommand();
    }*/
}
