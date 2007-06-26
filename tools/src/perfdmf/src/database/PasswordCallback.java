package edu.uoregon.tau.perfdmf.database;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;

import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
	

public interface PasswordCallback {

    String getPassword(ParseConfig config);

    public static PasswordCallback guiPasswordCallback = new PasswordCallback() {

        public String getPassword(ParseConfig config) {
            JPanel promptPanel = new JPanel();

            promptPanel.setLayout(new GridBagLayout());
            GridBagConstraints gbc = new GridBagConstraints();
            gbc.insets = new Insets(5, 5, 5, 5);

            JLabel label = new JLabel("<html>Enter password for user '" + config.getDBUserName() + "'<br> Database: '"
                    + config.getDBName() + "' (" + config.getDBHost() + ":" + config.getDBPort() + ")</html>");

            JPasswordField password = new JPasswordField(15);

            promptPanel.add(label);
            promptPanel.add(password);

            if (JOptionPane.showConfirmDialog(null, promptPanel, "Enter Password", JOptionPane.OK_CANCEL_OPTION) == JOptionPane.OK_OPTION) {
                return new String(password.getPassword());
            } else {
                return null;
            }
        }
    };
}

