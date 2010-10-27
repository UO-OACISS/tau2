package edu.uoregon.tau.perfexplorer.server;

import java.util.ArrayList;
import java.util.List;

/**
 * The DendrogramTree class exists to store the results of the hierarchical
 * clustering, and to find representative "centers" for each sub-cluster,
 * if the tree is cut at a particular height.  For example, if the tree
 * contains 32 leaves, and the tree is cut into two sub-trees, the methods
 * will find representative centers for each of the two sub-trees.
 *
 * <P>CVS $Id: DendrogramTree.java,v 1.3 2009/02/27 00:45:10 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class DendrogramTree {

	private DendrogramTree left = null;
	private DendrogramTree right = null;
	private int id = 0;
	private double height = 0.0;
	private int maxDepth = 0;

	/**
	 * Constructor.
	 * 
	 * @param id
	 * @param height
	 */
	public DendrogramTree (int id, double height) {
		this.id = id;
		this.height = height;
	}

	/**
	 * Set the input subtree as the left side of "this" tree.
	 * 
	 * @param left
	 */
	public void setLeft(DendrogramTree left) {
		this.left = left;
		if ((this.right == null) || (left.getDepth() > right.getDepth()))
			maxDepth = left.getDepth() + 1;
		else
			maxDepth = right.getDepth() + 1;
	}

	/** 
	 * Set the input subtree as the right side of "this" tree.
	 * 
	 * @param right
	 */
	public void setRight(DendrogramTree right) {
		this.right = right;
		if ((this.left == null) || (right.getDepth() > left.getDepth()))
			maxDepth = right.getDepth() + 1;
		else
			maxDepth = left.getDepth() + 1;
	}

	/**
	 * Convenience method to set both the left and right subtrees of
	 * this tree.
	 * 
	 * @param left
	 * @param right
	 */
	public void setLeftAndRight(DendrogramTree left, DendrogramTree right) {
		this.left = left;
		this.right = right;
		if (right.getDepth() > left.getDepth())
			maxDepth = right.getDepth() + 1;
		else
			maxDepth = left.getDepth() + 1;
	}

	/**
	 * Check whether this tree is a leaf.
	 * 
	 * @return
	 */
	public boolean isLeaf() {
		return (left == null && right == null) ? true : false;
	}

	/**
	 * Return the left sub-tree of this node in the tree.
	 * 
	 * @return
	 */
	public DendrogramTree getLeft() {
		return left;
	}

	/**
	 * Return the right sub-tree of this node in the tree.
	 * 
	 * @return
	 */
	public DendrogramTree getRight() {
		return right;
	}

	/**
	 * Get the height of this node in the tree.  The height
	 * is defined as the "distance" between the left and right
	 * sub-trees.
	 * 
	 * @return
	 */
	public double getHeight() {
		return height;
	}

	/**
	 * Find centers for this node in the tree.
	 * Centers for this node are found by decending the tree, and finding
	 * "representative" leaf nodes.  If one representative is needed, then
	 * it is selected from the left side of the sub-tree.  If two 
	 * representatives are needed, one each is selected from the left and
	 * right sub-trees.  If more than two representatives are needed, then
	 * The two sub-trees are traversed, to find the required number of 
	 * sub-tree nodes which have the greatest depth from the current node,
	 * that is, the sub-trees which are more "separated" from other nodes.
	 * 
	 * @param count
	 * @return
	 */
	public int[] findCenters(int count) {
		int[] centers = new int[count];
		if (count == 1) {
			// we only need one, so return the left representative
			centers[0] = getRepresentative();
		}else if (count == 2) {
			// return a representative from the left and right
			centers[0] = left.getRepresentative();
			centers[1] = right.getRepresentative();
		}else {
			// find "count" sub-trees which are the furthest away from
			// each other
			List<DendrogramTree> clusters = new ArrayList<DendrogramTree>(count);
			clusters.add(left);
			clusters.add(right);
			DendrogramTree current = null;
			while (clusters.size() < count) {
				DendrogramTree winner = clusters.get(0);
				for (int i = 0 ; i < clusters.size() ; i++) {
					current = clusters.get(i);
					if (current.getHeight() > winner.getHeight())
						winner = current;
				}
				clusters.remove(winner);
				clusters.add(winner.getLeft());
				clusters.add(winner.getRight());
			}
			// for each sub-tree, select a representative
			for (int i = 0 ; i < clusters.size() ; i++) {
				current = clusters.get(i);
				centers[i] = current.getRepresentative();
			}
		}
		return centers;
	}

	/**
	 * Find a representative for this sub-tree.
	 * The ideal representative is a node in the sub-tree which is a leaf
	 * node, and is one of the (if not the) deepest leaf in the tree and is
	 * closer to its parent node than its sibling leaf.  This method is called
	 * recursively to traverse sub-trees and find the representative leaf.
	 * 
	 * @return
	 */
	public int getRepresentative() {
		if (isLeaf())
			return java.lang.Math.abs(id);
		else if (left.getDepth() == right.getDepth())
			if (left.getHeight() < right.getHeight())
				return left.getRepresentative();
			else
				return right.getRepresentative();
		else if (left.getDepth() > right.getDepth())
			return left.getRepresentative();
		else
			return right.getRepresentative();
	}

	/**
	 * This method finds the "depth" of the tree, i.e. the greater of either
	 * the depth of the left side or the right side.  The depth is the number
	 * of nodes required to traverse to find the deepest leaf in the tree.
	 * 
	 * @return
	 */
	public int getDepth() {
		return maxDepth;
	}
}

