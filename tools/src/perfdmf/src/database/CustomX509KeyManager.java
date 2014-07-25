package edu.uoregon.tau.perfdmf.database;

import java.net.Socket;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.Principal;
import java.security.PrivateKey;
import java.security.UnrecoverableKeyException;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.net.ssl.KeyManager;
import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.X509KeyManager;
import javax.net.ssl.X509TrustManager;

/**
 * Wrap the standard SunX509 key manager
 *
 * @author craig
 */
public class CustomX509KeyManager implements X509KeyManager {

    private final KeyStore keyStore;
    private final String keyAlias;
    private char[] keyStorePassword;
    private static final Logger logger = Logger.getLogger(CustomX509KeyManager.class.getName());
	private static String desiredAlias = null;

    public CustomX509KeyManager(KeyStore ks, char[] password) {
         keyStore = ks;
         keyStorePassword = password;
         // Does Key store have more than one key alias? If so,
         // reject it as we can't select between them.
         keyAlias = getKeyAlias(keyStore);
    }

    public String[] getClientAliases(String string, Principal[] prncpls) {
        return new String[] { keyAlias };
    }

    public String chooseClientAlias(String[] strings, Principal[] prncpls, Socket socket) {
        return keyAlias;
    }

    public String[] getServerAliases(String string, Principal[] prncpls) {
        throw new UnsupportedOperationException("This KeyManager only supports client mode");
    }

    public String chooseServerAlias(String string, Principal[] prncpls, Socket socket) {
        throw new UnsupportedOperationException("This KeyManager only supports client mode");
    }

    public X509Certificate[] getCertificateChain(String alias) {
        try {
            Certificate[] certs = keyStore.getCertificateChain(alias);
            X509Certificate[] xcerts = new X509Certificate[certs.length];
            System.arraycopy(certs, 0, xcerts, 0, certs.length);
            return xcerts;
        } catch (KeyStoreException ex) {
            logger.log(Level.SEVERE, "Unable to retrieve certificate chain for client certificate", ex);
            return null;
        }
    }

    public PrivateKey getPrivateKey(String alias) {
        try {
            return (PrivateKey) keyStore.getKey(alias, keyStorePassword);
        } catch (KeyStoreException ex) {
            logger.log(Level.SEVERE, "Unable to retrieve private key for client certificate", ex);
            return null;
        } catch (NoSuchAlgorithmException ex) {
            logger.log(Level.SEVERE, "Unable to retrieve private key for client certificate", ex);
            return null;
        } catch (UnrecoverableKeyException ex) {
            logger.log(Level.SEVERE, "Unable to retrieve private key for client certificate", ex);
            return null;
        }
    }

    private X509KeyManager getDefaultKeyManager(KeyStore ks, char[] password) {
        try {
            KeyManagerFactory f = KeyManagerFactory.getInstance("SunX509");
            f.init(ks, password);
            KeyManager[] kms = f.getKeyManagers();
            for (KeyManager km : kms) {
                if (km instanceof X509TrustManager) {
                    return (X509KeyManager)km;
                }
            }
            throw new CustomSSLError("No X509KeyManager found");
        } catch (NoSuchAlgorithmException ex) {
            throw new CustomSSLError(ex);
        } catch (KeyStoreException ex) {
            throw new CustomSSLError(ex);
        } catch (UnrecoverableKeyException ex) {
            throw new CustomSSLError(ex);
        }
    }

    private static String getKeyAlias(KeyStore keyStore) {
        String foundKeyAlias = null;
        try {
            Enumeration<String> aliases = keyStore.aliases();
            while (aliases.hasMoreElements()) {
                String alias = aliases.nextElement();
	            if (CustomX509KeyManager.desiredAlias.equals(alias)) {
                	if (keyStore.isKeyEntry(alias)) {
                    	if (foundKeyAlias == null) {
                        	foundKeyAlias = alias;
							break;
						}
                	}
				}
            }
            if (foundKeyAlias == null) {
                throw new CustomSSLError("Key store contains no keys, it is empty or contains only trusted certs");
            }
        } catch (KeyStoreException ex) {
            throw new CustomSSLError("Error reading key store to determine key alias", ex);
        }
        return foundKeyAlias;
    }

	public static void setClientAlias(String alias) {
	    CustomX509KeyManager.desiredAlias = alias;
	}
}
