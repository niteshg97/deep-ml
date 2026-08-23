def compute_pruning_alphas(tree: dict) -> list:
    """
    Computes effective alpha values for cost-complexity pruning.

    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None

    Returns:
        List of effective alpha values for internal nodes, sorted ascending.
    """

    # Store alpha values of all internal nodes
    alphas = []

    # ------------------------------------------------
    # Recursive function to traverse the tree
    # ------------------------------------------------
    def traverse(node):

        # Step 1: Check if this node is a leaf
        if node['left'] is None and node['right'] is None:

            # A leaf has:
            #   - its own error
            #   - exactly 1 leaf
            return node['errors'], 1

        # Step 2: Process left subtree
        left_error, left_leaves = traverse(node['left'])

        # Step 3: Process right subtree
        right_error, right_leaves = traverse(node['right'])

        # Step 4: Total error of current subtree
        subtree_error = left_error + right_error

        # Step 5: Number of leaves in current subtree
        subtree_leaves = left_leaves + right_leaves

        # Step 6: Calculate effective alpha
        alpha = (
            node['errors'] - subtree_error
        ) / (
            subtree_leaves - 1
        )

        # Step 7: Store alpha
        alphas.append(alpha)

        # Step 8: Return information to parent
        return subtree_error, subtree_leaves

    # Start traversal from root
    traverse(tree)

    # Sort alpha values
    alphas.sort()

    return alphas


